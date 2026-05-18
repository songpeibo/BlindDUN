"""
BlindDUN: PALM-style deep unfolding — spectral + spatial residual linear updates,
then HeavyProxCompat U-Net proximal map. Optional per-stage Phi / k refinement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.degradation.operators import (
    normalize_kernel,
    normalize_srf,
    project_msi_to_hsi,
    spatial_degrade,
    spectral_degrade,
    upsample_hsi,
)
from models.prox.heavyprox import HeavyProxCompat
from models.prox.lwt_lite_prox import LWTLiteProxRefinement
from models.prox.lwt_prox import LWTProxRefinement
from models.unfolding.updaters import SpatialKernelUpdater, SpectralResponseUpdater


def inverse_softplus(x: float) -> float:
    """Inverse of softplus; x > 0. Used to init raw so softplus(raw) ≈ x."""
    ex = math.exp(float(x))
    return math.log(max(ex - 1.0, 1e-12))


class MGuidanceProjector(nn.Module):
    """
    Spectral MSI guidance for HeavyProx: ``m_guidance = M_backproj + guidance_scale * M_learned``.

    ``M_backproj = project_msi_to_hsi(M, Phi, C_hsi)``. ``M_learned`` is a small CNN on ``M_backproj``;
    the last conv is zero-initialized so ``M_learned ≡ 0`` at startup.

    ``guidance_scale`` is a scalar in ``(0, 1)``, learnable as ``sigmoid(raw)``, initialized to ``0.01``,
    with asymptotic maximum ``1.0``.
    """

    def __init__(self, hsi_channels: int, hidden_channels: int = 48, guidance_scale_init: float = 0.01) -> None:
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        h = max(int(hidden_channels), 16)
        self.m_learned = nn.Sequential(
            nn.Conv2d(self.hsi_channels, h, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(h, self.hsi_channels, kernel_size=3, padding=1, bias=True),
        )
        p0 = float(guidance_scale_init)
        if not (0.0 < p0 < 1.0):
            raise ValueError(f"guidance_scale_init must be in (0, 1) for sigmoid init, got {guidance_scale_init!r}")
        raw0 = math.log(p0 / (1.0 - p0))
        self.raw_guidance_scale = nn.Parameter(torch.tensor(raw0, dtype=torch.float32))
        self._init_m_learned_weights()

    def _init_m_learned_weights(self) -> None:
        last = self.m_learned[-1]
        for m in self.m_learned.modules():
            if isinstance(m, nn.Conv2d):
                if m is last:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def guidance_scale(self) -> torch.Tensor:
        """Scalar in (0, 1); init ≈ 0.01; approaches 1.0 as raw → +∞."""
        return torch.sigmoid(self.raw_guidance_scale)

    def forward(self, M: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        M_backproj = project_msi_to_hsi(M, Phi, self.hsi_channels)
        M_learned = self.m_learned(M_backproj)
        g = self.guidance_scale()
        return M_backproj + g * M_learned


class _InitRefineNet(nn.Module):
    """
    Non-linear warm start z0 = h_up + F(h_up). F is a small multi-block conv net (not identity).
    """

    def __init__(self, hsi_channels: int, hidden_channels: int) -> None:
        super().__init__()
        c = hsi_channels
        h = max(int(hidden_channels), 32)
        self.stem = nn.Conv2d(c, h, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(h)
        self.rb1 = nn.Sequential(
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
        )
        self.rb2 = nn.Sequential(
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
        )
        self.head = nn.Conv2d(h, c, 3, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Anchor-preserving: start as identity on H_up (residual branch off at init).
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, h_up: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn0(self.stem(h_up)))
        x = F.gelu(x + self.rb1(x))
        x = F.gelu(x + self.rb2(x))
        return h_up + self.head(x)


class _WaveletDecoupleBranch(nn.Module):
    """
    .. deprecated::
        Legacy placeholder branch for ``use_wavelet_decoupling``. Not used when ``use_lwt_prox`` is
        ``True``. Prefer :class:`~models.prox.lwt_prox.LWTProxRefinement` for learnable lifting-wavelet
        proximal refinement.

    Optional lightweight residual branch (off by default).

    Mixes a box low-pass ``lf`` and high-pass ``hf = z - lf`` into ``body`` input via learnable
    positive scalars ``gamma_L``, ``gamma_D`` (softplus), identity at init (``gamma = 1`` → ``z_mix = z``).
    Last conv of ``body`` is zero-init so ``delta ≡ 0`` at startup.

    ``collect_diag``: return ``(delta, diag_dict)`` with detached scalars for epoch logging (no large tensors).
    """

    def __init__(self, hsi_channels: int) -> None:
        super().__init__()
        c = int(hsi_channels)
        h = max(c, 32)
        self.body = nn.Sequential(
            nn.Conv2d(c, h, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(h, c, 3, padding=1, bias=True),
        )
        last = self.body[-1]
        assert isinstance(last, nn.Conv2d)
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
        for m in self.body.modules():
            if isinstance(m, nn.Conv2d) and m is not last:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        g0 = inverse_softplus(1.0)
        self.raw_gamma_l = nn.Parameter(torch.tensor(g0, dtype=torch.float32))
        self.raw_gamma_d = nn.Parameter(torch.tensor(g0, dtype=torch.float32))

    def forward(self, z: torch.Tensor, collect_diag: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        lf = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        hf = z - lf
        gl = F.softplus(self.raw_gamma_l)
        gd = F.softplus(self.raw_gamma_d)
        z_mix = z + (gl - 1.0) * (lf - z) + (gd - 1.0) * hf
        delta = self.body(z_mix)
        diag: Dict[str, float] = {}
        if collect_diag:
            with torch.no_grad():
                hf_gate = torch.sigmoid(hf.detach().abs().mean(dim=(1, 2, 3)))
                diag["diag_wv_gamma_L"] = float(gl.detach().item())
                diag["diag_wv_gamma_D"] = float(gd.detach().item())
                diag["diag_wv_lf_residual_abs_mean"] = float(lf.detach().abs().mean().item())
                diag["diag_wv_hf_residual_abs_mean"] = float(hf.detach().abs().mean().item())
                diag["diag_wv_hf_gate_mean"] = float(hf_gate.mean().item())
                diag["diag_wv_hf_gate_std"] = float(hf_gate.std(unbiased=False).item())
        return delta, diag


def _diag_da_fusion_energy(z_enter: torch.Tensor, h_up: torch.Tensor, m_guidance: torch.Tensor) -> Dict[str, float]:
    """
    Lightweight pseudo fusion weights: softmax over per-sample L2 energy of (Z, M_guidance, H_up).

    These are diagnostic energy allocations, not the internal fusion weights of HeavyProx.
    Detached scalars only (for JSONL / CSV diagnostics).
    """
    with torch.no_grad():
        z = z_enter.float()
        h = h_up.float()
        m = m_guidance.float()
        ez = z.pow(2).mean(dim=(1, 2, 3))
        em = m.pow(2).mean(dim=(1, 2, 3))
        eh = h.pow(2).mean(dim=(1, 2, 3))
        logits = torch.stack([ez, em, eh], dim=1)
        W = torch.softmax(logits, dim=1)
        wz, wm, wh = W[:, 0], W[:, 1], W[:, 2]
        ent = (-(W * (W + 1e-8).log()).sum(dim=1)).mean()
        return {
            "diag_da_W_Z_mean": float(wz.mean().item()),
            "diag_da_W_Z_std": float(wz.std(unbiased=False).item()),
            "diag_da_W_M_mean": float(wm.mean().item()),
            "diag_da_W_M_std": float(wm.std(unbiased=False).item()),
            "diag_da_W_H_mean": float(wh.mean().item()),
            "diag_da_W_H_std": float(wh.std(unbiased=False).item()),
            "diag_da_gate_entropy_mean": float(ent.item()),
            "diag_da_collapse_W_Z": float((wz.mean() > 0.9).item()),
            "diag_da_collapse_W_M": float((wm.mean() > 0.9).item()),
            "diag_da_collapse_W_H": float((wh.mean() > 0.9).item()),
        }


class BlindDUN(nn.Module):
    """
    Stages: PALM-type linear correction using MSI spectral pullback + upsampled LR residual,
    then HeavyProxCompat. Optional projected-gradient steps on Phi and k when enabled.

    Per-stage spectral / spatial / hub linear step sizes use ``tau = softplus(raw)``; all ``raw``
    tensors are initialized via ``inverse_softplus(tau_init)`` (default ``tau_init=0.001``).

    HeavyProx MSI guidance uses :class:`MGuidanceProjector`: ``m_guidance = M_backproj + g * M_learned``
    with bounded learnable ``g`` (``m_guidance_scale`` in aux); ``g`` init via ``guidance_scale_init``.

    HeavyProx capacity: ``hidden_channels``, ``prox_num_blocks`` (passed as ``HeavyProxCompat.num_blocks``),
    and ``prox_residual_scale_init`` (see :class:`~models.prox.heavyprox.HeavyProxCompat`).

    ``use_degradation_aware_fusion`` (default ``True``): when ``False``, skip the HeavyProx CNN and
    MSI guidance projector in the forward (parameters frozen): proximal step is plain ``clamp(z_lin)``.

    ``use_wavelet_decoupling`` (default ``False``): when ``True`` and neither ``use_lwt_prox`` nor
    ``use_lwt_lite_prox`` is enabled, apply deprecated :class:`_WaveletDecoupleBranch` to ``z_lin`` before HeavyProx.

    ``use_lwt_prox`` (default ``False``): when ``True``, run :class:`~models.prox.lwt_prox.LWTProxRefinement`
    after HeavyProx: ``Z_base = HeavyProxCompat(z_enter, …)``, then ``Z = LWTProxRefinement(Z_base, …)``
    (never replaces HeavyProx on ``z_enter``). Does not use ``_WaveletDecoupleBranch``.
    ``lwt_identity_mode`` (default ``False``): when ``True``, each LWT block is a strict no-op
    (returns ``Z_base``) for B3 + wrapper parity checks.

    ``lwt_init_seed`` (default ``3407``): seed used only inside ``torch.random.fork_rng`` when
    constructing ``LWTProxRefinement`` so enabling LWT does not perturb global RNG for base modules.

    ``use_lwt_lite_prox`` (default ``False``): optional :class:`~models.prox.lwt_lite_prox.LWTLiteProxRefinement`
    after HeavyProx (and after full ``lwt_prox`` when enabled — ``use_lwt_prox`` and ``use_lwt_lite_prox`` are
    mutually exclusive). ``lwt_lite_apply_stages`` is ``"last"`` (default) or ``"all"``. Built after all base
    modules with ``torch.random.fork_rng`` and ``lwt_lite_init_seed`` (default ``3407``).

    ``collect_lwt_inspect`` (forward kwarg, default ``False``): when ``True`` with ``return_aux=True``,
    stores last-stage ``lwt_inspect_z_base`` / ``lwt_inspect_z_next`` (post-LWT, pre-clamp) in the aux dict
    for offline tools (e.g. ``tools/inspect_lwt_forward.py``).

    With ``return_aux=True``, ``diag_vals`` may include legacy wavelet keys only when the deprecated branch
    is active; DA / degradation diagnostics unchanged when fusion is on.
    """

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        scale: int,
        stages: int = 2,
        hidden_channels: int = 96,
        prox_type: str = "heavyprox",
        update_phi: bool = False,
        update_k: bool = False,
        use_degradation_aware_fusion: bool = True,
        use_wavelet_decoupling: bool = False,
        use_lwt_prox: bool = False,
        lwt_levels: int = 1,
        lwt_hidden_channels: int = 64,
        lwt_learnable_transform: bool = True,
        lwt_use_lf_hsi_guidance: bool = True,
        lwt_use_hf_msi_guidance: bool = True,
        lwt_residual_scale_init: float = 1e-3,
        lwt_identity_mode: bool = False,
        lwt_init_seed: int = 3407,
        use_lwt_lite_prox: bool = False,
        lwt_lite_apply_stages: str = "last",
        lwt_lite_feature_channels: int = 32,
        lwt_lite_learnable_transform: bool = False,
        lwt_lite_use_lf_hsi_guidance: bool = True,
        lwt_lite_use_hf_msi_guidance: bool = True,
        lwt_lite_rho_init: float = 0.0,
        lwt_lite_rho_max: float = 0.001,
        lwt_lite_identity_mode: bool = False,
        lwt_lite_init_seed: int = 3407,
        lr_phi: float = 5e-3,
        lr_k: float = 1e-4,
        blur_kernel_hw: Tuple[int, int] = (5, 5),
        tau_init: float = 0.001,
        prox_num_blocks: int = 8,
        prox_residual_scale_init: float = 0.001,
        guidance_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        if prox_type != "heavyprox":
            raise ValueError(f"Unsupported prox_type {prox_type!r}; first version only supports 'heavyprox'.")
        if stages < 1:
            raise ValueError("stages must be >= 1")

        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)
        self.stages = int(stages)
        self.hidden_channels = int(hidden_channels)
        self.update_phi = bool(update_phi)
        self.update_k = bool(update_k)
        self.use_degradation_aware_fusion = bool(use_degradation_aware_fusion)
        self.use_wavelet_decoupling = bool(use_wavelet_decoupling)
        self.use_lwt_prox = bool(use_lwt_prox)
        self.lwt_levels = max(1, int(lwt_levels))
        self.lwt_hidden_channels = int(lwt_hidden_channels)
        self.lwt_learnable_transform = bool(lwt_learnable_transform)
        self.lwt_use_lf_hsi_guidance = bool(lwt_use_lf_hsi_guidance)
        self.lwt_use_hf_msi_guidance = bool(lwt_use_hf_msi_guidance)
        self.lwt_residual_scale_init = float(lwt_residual_scale_init)
        self.lwt_identity_mode = bool(lwt_identity_mode)
        self.lwt_init_seed = int(lwt_init_seed)
        self.use_lwt_lite_prox = bool(use_lwt_lite_prox)
        _las = str(lwt_lite_apply_stages).lower().strip()
        if _las not in ("last", "all"):
            raise ValueError(f"lwt_lite_apply_stages must be 'last' or 'all', got {lwt_lite_apply_stages!r}")
        self.lwt_lite_apply_stages = _las
        self.lwt_lite_feature_channels = int(lwt_lite_feature_channels)
        self.lwt_lite_learnable_transform = bool(lwt_lite_learnable_transform)
        self.lwt_lite_use_lf_hsi_guidance = bool(lwt_lite_use_lf_hsi_guidance)
        self.lwt_lite_use_hf_msi_guidance = bool(lwt_lite_use_hf_msi_guidance)
        self.lwt_lite_rho_init = float(lwt_lite_rho_init)
        self.lwt_lite_rho_max = float(lwt_lite_rho_max)
        self.lwt_lite_identity_mode = bool(lwt_lite_identity_mode)
        self.lwt_lite_init_seed = int(lwt_lite_init_seed)
        if self.use_lwt_prox and self.use_lwt_lite_prox:
            raise ValueError("use_lwt_prox and use_lwt_lite_prox cannot both be True.")
        self.lr_phi = float(lr_phi)
        self.lr_k = float(lr_k)
        self.blur_kernel_hw = (int(blur_kernel_hw[0]), int(blur_kernel_hw[1]))
        self.tau_init = float(tau_init)
        if self.tau_init <= 0.0:
            raise ValueError(f"tau_init must be > 0 (softplus domain), got {tau_init!r}")
        self.prox_num_blocks = int(prox_num_blocks)
        if self.prox_num_blocks < 1:
            raise ValueError(f"prox_num_blocks must be >= 1, got {prox_num_blocks!r}")

        self.init_refiner = _InitRefineNet(self.hsi_channels, self.hidden_channels)
        _guid_h = max(self.hidden_channels // 2, 32)
        self.m_guidance_projector = MGuidanceProjector(
            self.hsi_channels,
            hidden_channels=_guid_h,
            guidance_scale_init=float(guidance_scale_init),
        )
        self.prox = HeavyProxCompat(
            self.hsi_channels,
            hidden_channels=self.hidden_channels,
            num_blocks=self.prox_num_blocks,
            residual_scale_init=float(prox_residual_scale_init),
        )
        self.wavelet_branch: Optional[nn.Module] = (
            _WaveletDecoupleBranch(self.hsi_channels)
            if (self.use_wavelet_decoupling and not self.use_lwt_prox and not self.use_lwt_lite_prox)
            else None
        )
        self.lwt_prox: Optional[nn.ModuleList] = None
        if not self.use_degradation_aware_fusion:
            for p in self.prox.parameters():
                p.requires_grad_(False)
            for p in self.m_guidance_projector.parameters():
                p.requires_grad_(False)

        self.phi_updater: Optional[SpectralResponseUpdater] = (
            SpectralResponseUpdater(self.hsi_channels, self.msi_channels)
            if self.update_phi
            else None
        )
        self.kernel_updater: Optional[SpatialKernelUpdater] = (
            SpatialKernelUpdater(
                self.hsi_channels,
                self.blur_kernel_hw[0],
                self.blur_kernel_hw[1],
            )
            if self.update_k
            else None
        )

        # Learnable PALM step strengths: tau = softplus(raw); init raw = inverse_softplus(tau_init).
        _r0 = inverse_softplus(self.tau_init)
        _t0 = torch.tensor([_r0], dtype=torch.float32)
        self._raw_eta_spec = nn.ParameterList(
            [nn.Parameter(_t0.clone()) for _ in range(self.stages)]
        )
        self._raw_eta_spa = nn.ParameterList(
            [nn.Parameter(_t0.clone()) for _ in range(self.stages)]
        )
        self._raw_eta_hub = nn.ParameterList(
            [nn.Parameter(_t0.clone()) for _ in range(self.stages)]
        )

        # LWT after all base B3 modules so global RNG matches ``use_lwt_prox=False`` builds; fork isolates LWT init.
        if self.use_lwt_prox:
            lift_h = max(8, self.lwt_hidden_channels // 2)
            _fork_devs: List[torch.device] = []
            if torch.cuda.is_available():
                _fork_devs.extend(torch.device("cuda", i) for i in range(torch.cuda.device_count()))
            with torch.random.fork_rng(devices=_fork_devs, enabled=True):
                torch.manual_seed(self.lwt_init_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.lwt_init_seed)
                self.lwt_prox = nn.ModuleList(
                    [
                        LWTProxRefinement(
                            channels=self.hsi_channels,
                            hidden=self.lwt_hidden_channels,
                            lifting_hidden=lift_h,
                            learnable_transform=self.lwt_learnable_transform,
                            gamma_init=1e-3,
                            rho_init=float(self.lwt_residual_scale_init),
                            use_lf_hsi_guidance=self.lwt_use_lf_hsi_guidance,
                            use_hf_msi_guidance=self.lwt_use_hf_msi_guidance,
                            identity_mode=self.lwt_identity_mode,
                        )
                        for _ in range(self.lwt_levels)
                    ]
                )
            if not self.use_degradation_aware_fusion:
                for mod in self.lwt_prox:
                    for p in mod.parameters():
                        p.requires_grad_(False)

        self.lwt_lite_prox: Optional[LWTLiteProxRefinement] = None
        if self.use_lwt_lite_prox:
            _fork_lite: List[torch.device] = []
            if torch.cuda.is_available():
                _fork_lite.extend(torch.device("cuda", i) for i in range(torch.cuda.device_count()))
            with torch.random.fork_rng(devices=_fork_lite, enabled=True):
                torch.manual_seed(self.lwt_lite_init_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.lwt_lite_init_seed)
                self.lwt_lite_prox = LWTLiteProxRefinement(
                    channels=self.hsi_channels,
                    feature_channels=self.lwt_lite_feature_channels,
                    learnable_transform=self.lwt_lite_learnable_transform,
                    use_lf_hsi_guidance=self.lwt_lite_use_lf_hsi_guidance,
                    use_hf_msi_guidance=self.lwt_lite_use_hf_msi_guidance,
                    rho_init=float(self.lwt_lite_rho_init),
                    rho_max=float(self.lwt_lite_rho_max),
                    identity_mode=self.lwt_lite_identity_mode,
                )
            if not self.use_degradation_aware_fusion:
                for p in self.lwt_lite_prox.parameters():
                    p.requires_grad_(False)

        tqdm.write(
            "[LWT init] "
            f"use_lwt_prox={self.use_lwt_prox} "
            f"lwt_init_seed={self.lwt_init_seed} "
            f"lwt_identity_mode={self.lwt_identity_mode} "
            f"lwt_levels={self.lwt_levels} "
            f"lwt_learnable_transform={self.lwt_learnable_transform} "
            f"lwt_use_lf_hsi_guidance={self.lwt_use_lf_hsi_guidance} "
            f"lwt_use_hf_msi_guidance={self.lwt_use_hf_msi_guidance} "
            f"lwt_residual_scale_init={self.lwt_residual_scale_init}"
        )
        tqdm.write(
            "[LWT-Lite init] "
            f"use_lwt_lite_prox={self.use_lwt_lite_prox} "
            f"lwt_lite_init_seed={self.lwt_lite_init_seed} "
            f"lwt_lite_apply_stages={self.lwt_lite_apply_stages} "
            f"lwt_lite_feature_channels={self.lwt_lite_feature_channels} "
            f"lwt_lite_learnable_transform={self.lwt_lite_learnable_transform} "
            f"lwt_lite_use_lf_hsi_guidance={self.lwt_lite_use_lf_hsi_guidance} "
            f"lwt_lite_use_hf_msi_guidance={self.lwt_lite_use_hf_msi_guidance} "
            f"lwt_lite_rho_init={self.lwt_lite_rho_init} "
            f"lwt_lite_rho_max={self.lwt_lite_rho_max} "
            f"lwt_lite_identity_mode={self.lwt_lite_identity_mode}"
        )

    @staticmethod
    def _tau_from_raw(raw: torch.Tensor) -> torch.Tensor:
        """Strictly positive, learnable step size (per scalar parameter)."""
        return F.softplus(raw)

    def _apply_lwt_lite_this_stage(self, stage_idx: int) -> bool:
        if not self.use_lwt_lite_prox or self.lwt_lite_prox is None:
            return False
        if self.lwt_lite_apply_stages == "all":
            return True
        return int(stage_idx) == self.stages - 1

    def forward(
        self,
        M: torch.Tensor,
        H: torch.Tensor,
        H_up: torch.Tensor,
        Phi0: torch.Tensor,
        k0: torch.Tensor,
        return_aux: bool = False,
        collect_lwt_inspect: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if M.dim() != 4 or H.dim() != 4 or H_up.dim() != 4:
            raise ValueError("M, H, H_up must be 4D tensors [B,C,H,W].")
        b, c_hsi, h_hi, w_hi = H_up.shape
        if M.shape[0] != b or H.shape[0] != b:
            raise ValueError("Batch size mismatch among M, H, H_up.")
        if M.shape[1] != self.msi_channels or H.shape[1] != c_hsi or H_up.shape[1] != c_hsi:
            raise ValueError("Channel dimensions do not match model config.")
        if Phi0.shape != (self.msi_channels, self.hsi_channels):
            raise ValueError(
                f"Phi0 must be [{self.msi_channels}, {self.hsi_channels}], got {tuple(Phi0.shape)}"
            )
        kh_k, kw_k = int(self.blur_kernel_hw[0]), int(self.blur_kernel_hw[1])
        if k0.dim() == 3:
            if k0.shape != (1, kh_k, kw_k):
                raise ValueError(
                    f"k0 must be [1, {kh_k}, {kw_k}] (or batched [B, 1, kh, kw] / legacy [1, 1, kh, kw]); "
                    f"got {tuple(k0.shape)}"
                )
        elif k0.dim() == 4 and k0.shape[1] == 1:
            if tuple(k0.shape[-2:]) != self.blur_kernel_hw:
                raise ValueError(
                    f"k0 spatial {tuple(k0.shape[-2:])} must equal blur_kernel_hw {self.blur_kernel_hw}"
                )
            if k0.shape[0] not in (1, b):
                raise ValueError(
                    f"k0 batch must be 1 (broadcast) or B={b} matching M; got {k0.shape[0]}"
                )
        else:
            raise ValueError(
                f"k0 must be [1, kh, kw], [1, 1, kh, kw], or [B, 1, kh, kw]; got {tuple(k0.shape)}"
            )

        Phi = Phi0.to(device=M.device, dtype=M.dtype).clone()
        k = k0.to(device=M.device, dtype=M.dtype).clone()
        k = normalize_kernel(k)
        phi_t0 = Phi.detach().clone()
        k_t0 = k.detach().clone()

        Z = torch.clamp(self.init_refiner(H_up), 0.0, 1.0)
        stage_info: List[Dict[str, Any]] = []
        flat_aux: Dict[str, Any] = {}
        lwt_inspect_z_base: Optional[torch.Tensor] = None
        lwt_inspect_z_next: Optional[torch.Tensor] = None
        lwt_aux_sums: Dict[str, float] = {}
        lwt_aux_n = 0
        lwt_lite_aux_sums: Dict[str, float] = {}
        lwt_lite_aux_n = 0
        da_acc: Dict[str, float] = {}
        da_n = 0
        wv_acc: Dict[str, float] = {}
        wv_n = 0

        for s in range(self.stages):
            m_pred = spectral_degrade(Z, Phi)
            R_msi = M - m_pred
            r_spec_hsi = project_msi_to_hsi(R_msi, Phi, self.hsi_channels)

            h_pred = spatial_degrade(Z, k, self.scale)
            R_lr = H - h_pred
            r_spa_up = upsample_hsi(R_lr, target_size=(h_hi, w_hi))

            grad_spe = r_spec_hsi / (r_spec_hsi.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
            grad_spa = r_spa_up / (r_spa_up.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6)

            eta_s = self._tau_from_raw(self._raw_eta_spec[s])
            eta_p = self._tau_from_raw(self._raw_eta_spa[s])
            eta_h = self._tau_from_raw(self._raw_eta_hub[s])

            spe_spa = eta_s * grad_spe + eta_p * grad_spa
            z_lin = Z + spe_spa + eta_h * (H_up - Z)

            z_enter = z_lin
            if (
                self.use_wavelet_decoupling
                and self.wavelet_branch is not None
                and not self.use_lwt_prox
                and not self.use_lwt_lite_prox
            ):
                delta_w, wpart = self.wavelet_branch(z_lin, collect_diag=bool(return_aux))
                z_enter = z_lin + delta_w
                if return_aux and wpart:
                    for wk, wv in wpart.items():
                        wv_acc[wk] = wv_acc.get(wk, 0.0) + wv
                    wv_n += 1
            if self.use_degradation_aware_fusion:
                m_guidance = self.m_guidance_projector(M, Phi)
                if return_aux:
                    dpart = _diag_da_fusion_energy(z_enter, H_up, m_guidance)
                    for dk, dv in dpart.items():
                        da_acc[dk] = da_acc.get(dk, 0.0) + dv
                    da_n += 1
                    Z_base, paux = self.prox(
                        z_enter,
                        H_up,
                        m_guidance,
                        return_aux=True,
                    )
                    if self.lwt_prox is not None:
                        Z = Z_base
                        for lwt_blk in self.lwt_prox:
                            Z, lwt_d = lwt_blk(Z, H_up, m_guidance, return_diag=return_aux)
                            if return_aux and lwt_d is not None:
                                lwt_aux_n += 1
                                for _dk, _dv in lwt_d.items():
                                    lwt_aux_sums[_dk] = lwt_aux_sums.get(_dk, 0.0) + float(
                                        _dv.detach().float().cpu().reshape(()).item()
                                    )
                    else:
                        Z = Z_base
                    if self.lwt_lite_prox is not None and self._apply_lwt_lite_this_stage(s):
                        Z, lite_d = self.lwt_lite_prox(Z, H_up, m_guidance, return_diag=return_aux)
                        if return_aux and lite_d is not None:
                            lwt_lite_aux_n += 1
                            for _dk, _dv in lite_d.items():
                                lwt_lite_aux_sums[_dk] = lwt_lite_aux_sums.get(_dk, 0.0) + float(
                                    _dv.detach().float().cpu().reshape(()).item()
                                )
                    if (
                        collect_lwt_inspect
                        and s == self.stages - 1
                        and (
                            self.lwt_prox is not None
                            or (
                                self.lwt_lite_prox is not None
                                and self._apply_lwt_lite_this_stage(s)
                            )
                        )
                    ):
                        lwt_inspect_z_base = Z_base.detach().clone()
                        lwt_inspect_z_next = Z.detach().clone()
                    flat_aux[f"tau_stage{s}"] = torch.stack(
                        [
                            eta_s.detach().reshape(()),
                            eta_p.detach().reshape(()),
                            eta_h.detach().reshape(()),
                        ]
                    )
                    flat_aux[f"r_m_abs_mean_stage{s}"] = R_msi.detach().abs().mean()
                    flat_aux[f"r_h_abs_mean_stage{s}"] = R_lr.detach().abs().mean()
                    flat_aux[f"prox_residual_abs_mean_stage{s}"] = paux["prox_residual_abs_mean"].detach()
                    flat_aux[f"grad_spe_abs_mean_stage{s}"] = grad_spe.abs().mean().detach()
                    flat_aux[f"grad_spa_abs_mean_stage{s}"] = grad_spa.abs().mean().detach()
                    flat_aux[f"grad_abs_mean_stage{s}"] = spe_spa.abs().mean().detach()
                else:
                    Z_base = self.prox(
                        z_enter,
                        H_up,
                        m_guidance,
                        return_aux=False,
                    )
                    if self.lwt_prox is not None:
                        Z = Z_base
                        for lwt_blk in self.lwt_prox:
                            Z, _ = lwt_blk(Z, H_up, m_guidance, return_diag=False)
                    else:
                        Z = Z_base
                    if self.lwt_lite_prox is not None and self._apply_lwt_lite_this_stage(s):
                        Z, _ = self.lwt_lite_prox(Z, H_up, m_guidance, return_diag=False)
                    if (
                        collect_lwt_inspect
                        and s == self.stages - 1
                        and (
                            self.lwt_prox is not None
                            or (
                                self.lwt_lite_prox is not None
                                and self._apply_lwt_lite_this_stage(s)
                            )
                        )
                    ):
                        lwt_inspect_z_base = Z_base.detach().clone()
                        lwt_inspect_z_next = Z.detach().clone()
            else:
                Z = torch.clamp(z_enter, 0.0, 1.0)
                if return_aux:
                    flat_aux[f"tau_stage{s}"] = torch.stack(
                        [
                            eta_s.detach().reshape(()),
                            eta_p.detach().reshape(()),
                            eta_h.detach().reshape(()),
                        ]
                    )
                    flat_aux[f"r_m_abs_mean_stage{s}"] = R_msi.detach().abs().mean()
                    flat_aux[f"r_h_abs_mean_stage{s}"] = R_lr.detach().abs().mean()
                    zdev = z_enter.device
                    flat_aux[f"prox_residual_abs_mean_stage{s}"] = torch.zeros((), device=zdev, dtype=Z.dtype)
                    flat_aux[f"grad_spe_abs_mean_stage{s}"] = grad_spe.abs().mean().detach()
                    flat_aux[f"grad_spa_abs_mean_stage{s}"] = grad_spa.abs().mean().detach()
                    flat_aux[f"grad_abs_mean_stage{s}"] = spe_spa.abs().mean().detach()

            Z_before_projection = Z
            Z = torch.clamp(Z, 0.0, 1.0)
            if return_aux:
                flat_aux[f"pred_min_before_projection_stage{s}"] = Z_before_projection.detach().min()
                flat_aux[f"pred_max_before_projection_stage{s}"] = Z_before_projection.detach().max()
                flat_aux[f"pred_min_after_projection_stage{s}"] = Z.detach().min()
                flat_aux[f"pred_max_after_projection_stage{s}"] = Z.detach().max()
                flat_aux[f"clipped_fraction_low_stage{s}"] = (Z_before_projection < 0.0).float().mean().detach()
                flat_aux[f"clipped_fraction_high_stage{s}"] = (Z_before_projection > 1.0).float().mean().detach()

            if self.update_phi:
                assert self.phi_updater is not None
                r_m_after = M - spectral_degrade(Z, Phi)
                Phi, phi_aux = self.phi_updater(Z, M, r_m_after, Phi)

            if self.update_k:
                assert self.kernel_updater is not None
                r_h_after = H - spatial_degrade(Z, k, self.scale)
                k_in = k.unsqueeze(1) if k.dim() == 3 else k
                k, k_aux = self.kernel_updater(Z, H, r_h_after, k_in)

            if return_aux:
                entry: Dict[str, Any] = {
                    "R_msi": R_msi.detach(),
                    "R_lr": R_lr.detach(),
                    "r_spec_hsi": r_spec_hsi.detach(),
                    "r_spa_up": r_spa_up.detach(),
                }
                if self.update_phi:
                    entry["phi_updater_aux"] = {
                        ak: av.detach() if torch.is_tensor(av) else av for ak, av in phi_aux.items()
                    }
                if self.update_k:
                    entry["kernel_updater_aux"] = {
                        ak: av.detach() if torch.is_tensor(av) else av for ak, av in k_aux.items()
                    }
                stage_info.append(entry)

        if return_aux and self.lwt_prox is not None and lwt_aux_n > 0:
            inv = 1.0 / float(lwt_aux_n)
            for _lk, _sv in lwt_aux_sums.items():
                flat_aux[_lk] = _sv * inv

        if return_aux and self.lwt_lite_prox is not None and lwt_lite_aux_n > 0:
            inv_l = 1.0 / float(lwt_lite_aux_n)
            for _lk, _sv in lwt_lite_aux_sums.items():
                flat_aux[_lk] = _sv * inv_l

        if (
            return_aux
            and collect_lwt_inspect
            and lwt_inspect_z_base is not None
            and lwt_inspect_z_next is not None
        ):
            flat_aux["lwt_inspect_z_base"] = lwt_inspect_z_base
            flat_aux["lwt_inspect_z_next"] = lwt_inspect_z_next

        if return_aux:
            flat_aux["pred_min"] = Z.detach().min()
            flat_aux["pred_max"] = Z.detach().max()
            if self.use_degradation_aware_fusion:
                flat_aux["m_guidance_scale"] = self.m_guidance_projector.guidance_scale().detach().reshape(())
            else:
                flat_aux["m_guidance_scale"] = torch.zeros((), device=Z.device, dtype=Z.dtype)
            diag_vals: Dict[str, float] = {}
            if da_n > 0:
                for _k, _v in da_acc.items():
                    diag_vals[_k] = _v / float(da_n)
            if wv_n > 0:
                for _k, _v in wv_acc.items():
                    diag_vals[_k] = _v / float(wv_n)
            with torch.no_grad():
                diag_vals["diag_deg_phi_delta_l1_mean"] = float((Phi - phi_t0).abs().mean().item())
                diag_vals["diag_deg_k_delta_l1_mean"] = float((k - k_t0).abs().mean().item())
                ks = k.sum(dim=(-1, -2))
                diag_vals["diag_deg_k_sum_min"] = float(ks.min().item())
                diag_vals["diag_deg_k_sum_max"] = float(ks.max().item())
                diag_vals["diag_deg_phi_nonneg_ratio"] = float((Phi >= 0).float().mean().item())
                diag_vals["diag_deg_phi_min"] = float(Phi.min().item())
            for _fk, _fv in flat_aux.items():
                if isinstance(_fk, str) and _fk.startswith("lwt_") and isinstance(_fv, (int, float)):
                    diag_vals[_fk] = float(_fv)
            return Z, {
                **flat_aux,
                "stages": stage_info,
                "Phi": Phi.detach(),
                "k": k.detach(),
                "diag_vals": diag_vals,
            }
        return Z
