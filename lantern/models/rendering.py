import torch
from torch.functional import Tensor

from pytorch3d.renderer import ImplicitRenderer
from pytorch3d.renderer import PerspectiveCameras, RayBundle

from lantern.models.implicit_scene import ImplicitSceneModel
from lantern.models.raymarching import ProbabilisticRaymarcher
from lantern.models.raysampling import LanternRaySampler

from typing import Any, Tuple, Callable


class LanternInput():
    def __init__(
        self,
        R: Tensor,
        T: Tensor,
        focal_length: int,
        principal_point: Tuple
    ):
        self.R = R
        self.T = T
        self.focal_length = focal_length
        self.principal_point = principal_point


class LanternOutput():
    def __init__(
        self,
        rays: RayBundle,
        rendered_disparity: Tensor,
        rendered_intensity: Tensor
    ):
        self.rays = rays
        self.rendered_disparity = rendered_disparity
        self.rendered_intensity = rendered_intensity
    
class Lantern(torch.nn.Module):
    """
    Implements an end-to-end neural implicit renderer.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_samples_per_ray: int,
        num_rays_per_image: int,
        min_depth: float,
        max_depth: float,
        build_input_fn: Callable[[Any], LanternInput] = lambda input: input,
        build_output_fn: Callable[[LanternOutput],
                                  Any] = lambda output: output,
        device: str = 'cpu',
    ) -> None:

        """
        Args:
            image_size: TODO
            num_samples: TODO
            num_rays_per_image: TODO
            min_depth: TODO
            max_depth: TODO
            build_input_fn: TODO
            build_output_fn: TODO
            device: TODO
        """

        super().__init__()

        image_height, image_width = image_size

        self.build_input_fn = build_input_fn
        self.build_output_fn = build_output_fn

        self._implicit_scene = ImplicitSceneModel()

        self._raysampler = LanternRaySampler(
            image_height=image_height,
            image_width=image_width,
            num_rays_per_image=num_rays_per_image,
            num_samples_per_ray=num_samples_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        self._raymarcher = ProbabilisticRaymarcher()

        self._renderer = ImplicitRenderer(
            raysampler=self._raysampler,
            raymarcher=self._raymarcher,
        )

        self.device = device

        self._implicit_scene.to(device)
        self._raysampler.to(device)
        self._raymarcher.to(device)        

    def forward(self, input, *args, **kwargs):
        """
        Performs the rendering passes of the implicit scene
        model from the viewpoint of the input `cameras`.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image.

        Args:            
            input: A tuple or dict containing the model inputs, which will
            be parse according to the function provided in the constructor.                

        Returns:
            out: `dict` containing the outputs of the rendering:
                `depths`: The result of the depth rendering pass.
                `intensities`: The result of the intensity rendering pass.

                The shape of `depths` and `intensities` depends on the
                `self.training` flag:
                    If `self.training==True`, these tensors the results 
                    corresponding to a random set of rays,
                    i.e., `(batch_size, num_rays_per_image, channels)`.                     
                    If `self.training==False`, this tensor is of shape
                    `(batch_size, image_size[0], image_size[1], channels)`
                    and contain the result of the all rays in a grid.            
        """

        lantern_input = self.build_input_fn(input)

        cameras = PerspectiveCameras(
            R=lantern_input.R,
            T=lantern_input.T,
            focal_length=10.0,
            principal_point=((0.0, 0.0),),
        ).to(self.device)

        ray_bundle = self._raysampler.forward(
            cameras
        )

        ray_bundle.sampled_densities = self._implicit_scene.forward(
            ray_bundle.samples()
        )

        disparity = 1 / self._raymarcher(ray_bundle)

        lantern_output = LanternOutput(
            rays=ray_bundle, 
            rendered_disparity=disparity,
            rendered_intensity=None
        )

        return self.build_output_fn(lantern_output)
