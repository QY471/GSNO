"""Independent MSI-conditioned Gaussian rendering references and CUDA op."""

from .extension import (
    cuda_candidate_render,
    cuda_msi_conditioned_render,
    load_extension,
)
from .reference import (
    all_gaussian_full_reference,
    candidate_window_reference,
    gather_candidates,
    gaussian_logit,
    make_candidate_indices,
    make_query_coordinates,
)

__all__ = [
    "all_gaussian_full_reference",
    "candidate_window_reference",
    "cuda_candidate_render",
    "cuda_msi_conditioned_render",
    "gather_candidates",
    "gaussian_logit",
    "load_extension",
    "make_candidate_indices",
    "make_query_coordinates",
]
