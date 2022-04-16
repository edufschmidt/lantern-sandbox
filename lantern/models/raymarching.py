import torch
import warnings

from lantern.utils import RayBundle

from abc import ABC
from typing import Tuple


class Raymarcher(torch.nn.Module, ABC):
    def _check_inputs(
        self,
        sampled_densities: torch.Tensor,
    ):
        r"""Check the validity of the inputs to raymarching algorithms.
        """

        if not torch.is_tensor(sampled_densities):
            raise ValueError(
                "sampled_densities has to be an instance of torch.Tensor.")

        if sampled_densities.ndim < 1:
            raise ValueError(
                "sampled_densities have to have at least one dimension.")

        if sampled_densities.shape[-1] != 1:
            raise ValueError(
                "The size of the last dimension of sampled_densities " +
                "has to be one."
            )

    def _check_density_bounds(
        self,
        sampled_densities: torch.Tensor,
        bounds: Tuple[float, float] = (0.0, 1.0)
    ):
        r"""Checks whether the elements of `sampled_densities` range within `bounds`.
        If not issues a warning.
        """
        with torch.no_grad():
            if (sampled_densities.max() > bounds[1]) or \
                    (sampled_densities.min() < bounds[0]):
                warnings.warn(
                    "One or more elements of rays_densities are outside of "
                    + f"valid range {str(bounds)}"
                )



class ProbabilisticRaymarcher(Raymarcher):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        ray_bundle: RayBundle,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            sampled_densities: Per-ray density values represented with a tensor
                of shape `(..., num_samples_per_ray)` whose values range in [0,1].
            intersample_distance: The distance between samples taken along the ray.

        Returns:
            depths: A tensor of shape `(...,1)` containing estimated depths
            along each ray.
        """

        intersample_distances = ray_bundle.intersample_distances
        sampled_densities = ray_bundle.sampled_densities
        sampled_depths = ray_bundle.sampled_depths

        self._check_inputs(sampled_densities)
        self._check_density_bounds(sampled_densities)

        sampled_densities = sampled_densities[..., 0]

        occupancy_probabilities = torch.sub(
            1,
            torch.exp(
                -torch.mul(
                    sampled_densities,
                    intersample_distances
                )
            )
        )

        ray_termination_probabilities = torch.mul(
            occupancy_probabilities,
            torch.cumprod(
                1 - occupancy_probabilities + 1e-10, 
                dim=-1
            )
        )

        depths = torch.sum(
            torch.mul(
                ray_termination_probabilities,
                sampled_depths
            ),
            dim=-1, 
            keepdim=True
        )
        
        return depths


class AbsorptionOnlyRaymarcher(Raymarcher):
    """
    Raymarch using the Absorption-Only (AO) algorithm.

    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., num_samples_per_ray, 1)`, their values should range between [0, 1], 
    and represent the opaqueness of each point (the higher the less transparent).
    The algorithm only measures the total amount of light absorbed along each
    ray and outputs per-ray `opacity` values of shape `(...,)`.

    The algorithm simply computes `total_transmission = prod(1 - rays_densities)`
    of shape `(..., 1)` which, for each ray, measures the total amount of light
    that passed through the volume.
    It then returns `opacities = 1 - total_transmission`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        rays_densities: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., num_samples_per_ray)` whose values range in [0, 1].

        Returns:
            opacities: A tensor of per-ray opacity values of shape `(..., 1)`.
                Its values range between [0, 1] and denote the total amount
                of light that has been absorbed for each ray. E.g. a value
                of 0 corresponds to the ray completely passing through a
                volume.
        """

        self._check_inputs(rays_densities, None, features_can_be_none=True)

        rays_densities = rays_densities[..., 0]

        self._check_density_bounds(rays_densities)

        total_transmission = torch.prod(
            1 - rays_densities, dim=-1, keepdim=True)

        opacities = 1.0 - total_transmission

        return opacities
