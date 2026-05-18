"""
BlindDUN — public preview pseudo-implementation.

This is a public preview pseudo-implementation.
It illustrates the algorithmic workflow only.
It does not contain the full model implementation.
The complete code and pretrained models will be released upon paper acceptance.

This file is NOT runnable and is NOT sufficient to reproduce any reported results.
All learnable blocks, operators, and hyperparameters are intentionally abstracted.
"""

from __future__ import annotations

from typing import Any, Tuple


# Symbolic placeholders only (no real values).
NUM_STAGES = ...
STEP_SIZE = ...
MODEL_WIDTH = ...


def _not_implemented(name: str) -> None:
    raise NotImplementedError(
        f"'{name}' is a workflow placeholder only. "
        "Full implementation will be released after paper acceptance."
    )


def apply_spectral_forward_model(latent: Any, srf: Any) -> Any:
    """Abstract: map HR-HSI latent to MSI domain via SRF."""
    _not_implemented("apply_spectral_forward_model")


def apply_spatial_forward_model(latent: Any, psf: Any, scale: Any) -> Any:
    """Abstract: map HR-HSI latent to LR-HSI domain via PSF and downsampling."""
    _not_implemented("apply_spatial_forward_model")


def refine_srf(latent: Any, srf: Any, srf_prior: Any, msi_residual: Any) -> Any:
    """Abstract: update SRF from MSI consistency (learnable block withheld)."""
    _not_implemented("refine_srf")


def refine_psf(latent: Any, psf: Any, psf_prior: Any, lr_residual: Any) -> Any:
    """Abstract: update PSF from LR-HSI consistency (learnable block withheld)."""
    _not_implemented("refine_psf")


def project_srf(srf: Any) -> Any:
    """Abstract: project SRF onto feasible set (constraints withheld)."""
    _not_implemented("project_srf")


def project_psf(psf: Any) -> Any:
    """Abstract: project PSF onto feasible set (constraints withheld)."""
    _not_implemented("project_psf")


def initialize_latent(lr_hsi: Any, scale: Any) -> Any:
    """Abstract: build initial HR-HSI estimate from LR-HSI."""
    _not_implemented("initialize_latent")


def linear_consistency_update(
    latent: Any,
    msi_residual: Any,
    lr_residual: Any,
    step_size: Any,
) -> Any:
    """Abstract: combine spectral and spatial residuals into a latent correction."""
    _not_implemented("linear_consistency_update")


def learned_fusion_refinement(latent: Any, observations: Any, width: Any) -> Any:
    """Abstract: fuse multi-source observations into latent (architecture withheld)."""
    _not_implemented("learned_fusion_refinement")


def learned_proximal_refinement(latent: Any, width: Any, step_size: Any) -> Any:
    """Abstract: learned proximal / regularization step on latent (details withheld)."""
    _not_implemented("learned_proximal_refinement")


class BlindDUNPseudo:
    """
    High-level unfolding workflow for blind HSI–MSI fusion (pseudo-code only).

    Inputs:  HR-MSI M, LR-HSI H, nominal SRF prior, nominal PSF prior.
    Outputs: reconstructed HR-HSI, estimated SRF, estimated PSF.
    """

    def __init__(
        self,
        *,
        num_stages: Any = NUM_STAGES,
        step_size: Any = STEP_SIZE,
        model_width: Any = MODEL_WIDTH,
        estimate_srf: bool = True,
        estimate_psf: bool = True,
    ) -> None:
        self.num_stages = num_stages
        self.step_size = step_size
        self.model_width = model_width
        self.estimate_srf = estimate_srf
        self.estimate_psf = estimate_psf

    def forward(
        self,
        msi: Any,
        lr_hsi: Any,
        srf_prior: Any,
        psf_prior: Any,
        *,
        scale: Any = ...,
    ) -> Tuple[Any, Any, Any]:
        """
        Illustrative inference workflow (documentation only; not executed).

        1. Initialize latent HR-HSI from LR-HSI; set SRF/PSF to priors.
        2. For each unfolding stage:
           - Form MSI and LR-HSI residuals via abstract forward models.
           - Optionally refine and project SRF / PSF.
           - Apply abstract linear consistency, fusion, and proximal updates.
        3. Return reconstructed HR-HSI, estimated SRF, and estimated PSF.
        """
        _not_implemented("BlindDUNPseudo.forward")
