import torch
from torch import nn

from lantern.utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


class RaySampler(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _xy_to_ray_bundle(
        self,
        cameras: CamerasBase,
        xy_grid: torch.Tensor,
        min_depth: float,
        max_depth: float,
        num_samples_per_ray: int,
    ) -> RayBundle:
        """
        Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
        This adds to each xy location in the grid a vector of `num_pts_per_ray`
        depths uniformly spaced between `min_depth` and `max_depth`.

        The extended grid is then unprojected with `cameras` to yield
        ray origins, directions and depths.
        """
        batch_size = xy_grid.shape[0]
        spatial_size = xy_grid.shape[1:-1]
        num_rays_per_image = spatial_size.numel()

        # Generate Z coordinates for each ray
        depths = torch.linspace(
            min_depth,
            max_depth,
            num_samples_per_ray,
            dtype=xy_grid.dtype,
            device=xy_grid.device
        )

        rays_zs = depths[None, None].expand(
            batch_size,
            num_rays_per_image,
            num_samples_per_ray
        )

        # Generate two sets of points at a constant depth=1
        # and depth=2 which will be used to compute ray directions.
        to_unproject = torch.cat(
            (
                xy_grid.view(batch_size, 1, num_rays_per_image, 2)
                .expand(batch_size, 2, num_rays_per_image, 2)
                .reshape(batch_size, num_rays_per_image * 2, 2),
                torch.cat(
                    (
                        xy_grid.new_ones(
                            batch_size, num_rays_per_image, 1),
                        2.0 * xy_grid.new_ones(batch_size,
                                               num_rays_per_image, 1),
                    ),
                    dim=1,
                ),
            ),
            dim=-1,
        )

        # Unproject the points
        unprojected = cameras.unproject_points(to_unproject)

        # Split the two planes back
        rays_plane_1_world = unprojected[:, :num_rays_per_image]
        rays_plane_2_world = unprojected[:, num_rays_per_image:]

        # Compute ray directions as the difference between the two
        # sets of points.
        rays_directions_world = rays_plane_2_world - rays_plane_1_world

        # Ray origins are given by subtracting the directions from
        # the first plane.
        rays_origins_world = rays_plane_1_world - rays_directions_world

        origins_image = xy_grid
        origins_world = rays_origins_world.view(batch_size, *spatial_size, 3)
        directions = rays_directions_world.view(batch_size, *spatial_size, 3)
        sampled_depths = rays_zs.view(
            batch_size, *spatial_size, num_samples_per_ray)

        return RayBundle(
            origins_image,
            origins_world,
            directions,
            sampled_depths
        )

class GridRaySampler(RaySampler):
    """Samples a fixed number of points along rays *regularly distributed*
    in a batch of rectangular image grids. Points along each ray are uniformly-
    spaced with their z-coordinates ranging between a predefined minimum and
    maximum depth.

    The raysampler first generates a 3D coordinate grid of the following form:

    ```
       / min_x, min_y, max_depth -------------- / max_x, min_y, max_depth
      /                                        /|
     /                                        / |     ^
    / min_depth                    min_depth /  |     |
    min_x ----------------------------- max_x   |     | sample
    min_y                               min_y   |     | height
    |                                       |   |     |
    |                                       |   |     v
    |                                       |   |
    |                                       |   / max_x, max_y,   ^
    |                                       |  /  max_depth      /
    min_x                               max_y /                 /  num_samples
    max_y ----------------------------- max_x/ min_depth       v
              < --- sample_width --- >
    ```

    In order to generate ray points, `GridRaysampler` takes each 3D point of
    the grid (with coordinates `[x, y, depth]`) and unprojects it with
    `cameras.unproject_points([x, y, depth])`, where `cameras` are an
    additional input to the `forward` function.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        num_samples_x: int,
        num_samples_y: int,
        num_samples_per_ray: int,
        min_depth: float,
        max_depth: float,
        *args,
        **kwargs,
    ):
        """
        Args:
            min_x: Leftmost x-coordinate of each ray's source pixel's center.
            max_x: Rightmost x-coordinate of each ray's source pixel's center.
            min_y: Topmost y-coordinate of each ray's source pixel's center.
            max_y: Bottommost y-coordinate of each ray's source pixel's center.
            num_samples_x: Horizontal size of the image grid.
            num_samples_y: Vertical size of the image grid.
            num_samples_per_ray: Number of points sampled along each ray.
            min_depth: Minimum depth of a ray-point.
            max_depth: Maximum depth of a ray-point.
        """
        super().__init__(*args, **kwargs)

        self._num_samples_per_ray = num_samples_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth

        # Get the initial grid of image xy coords
        _xy_grid = torch.stack(
            tuple(
                reversed(
                    torch.meshgrid(
                        torch.linspace(min_y, max_y, num_samples_y + 1,
                                       dtype=torch.float32),
                        torch.linspace(min_x, max_x, num_samples_x + 1,
                                       dtype=torch.float32),
                    )
                )
            ),
            dim=-1,
        )

        self.register_buffer("_xy_grid", _xy_grid)

    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: Cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle containing (i) a tensor of shape
            `(batch_size, image_height, image_width, 3)` containing the
            3D coordinates of each ray origin in world coordinates; (ii)
            a tensor of shape `(batch_size, image_height, image_width, 3)`
            containing the directions of each ray in world coordinates;
            (iii) a tensor of shape `(batch_size, image_height, image_width,
            num_pts_per_ray)` containing the z-coords (=depth) of each ray in
            world units; and (iv) a tensor of shape `(batch_size, image_height,
            image_width, 2)` containing the 2D image coordinates of each ray.
        """
        
        batch_size = cameras.R.shape[0]

        device = cameras.device

        # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
        xy_grid = self._xy_grid.to(device)[None].expand(
            batch_size, *self._xy_grid.shape
        )

        return self._xy_to_ray_bundle(
            cameras,
            xy_grid,
            self._min_depth,
            self._max_depth,
            self._num_samples_per_ray,
        )


class NDCGridRaysampler(GridRaySampler):
    """
    Samples a fixed number of points along rays which are regularly distributed
    in a batch of rectangular image grids. Points along each ray
    have uniformly-spaced z-coordinates between a predefined minimum and maximum depth.

    `NDCGridRaysampler` follows the screen conventions of the `Meshes` and `Pointclouds`
    renderers. I.e. the border of the leftmost / rightmost / topmost / bottommost pixel
    has coordinates 1.0 / -1.0 / 1.0 / -1.0 respectively.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        num_samples_per_ray: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Args:
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            num_samples_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
        """
        half_pix_width = 1.0 / image_width
        half_pix_height = 1.0 / image_height
        super().__init__(
            min_x=1.0 - half_pix_width,
            max_x=-1.0 + half_pix_width,
            min_y=1.0 - half_pix_height,
            max_y=-1.0 + half_pix_height,
            num_samples_x=image_width,
            num_samples_y=image_height,
            num_samples_per_ray=num_samples_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )


class MonteCarloRaySampler(RaySampler):
    """
    Samples a fixed number of pixels within denoted xy bounds uniformly at random.
    For each pixel, a fixed number of points is sampled along its ray at uniformly-spaced
    z-coordinates such that the z-coordinates range between a predefined minimum
    and maximum depth.
    """

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        num_rays_per_image: int,
        num_samples_per_ray: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Args:
            min_x: The smallest x-coordinate of each ray's source pixel.
            max_x: The largest x-coordinate of each ray's source pixel.
            min_y: The smallest y-coordinate of each ray's source pixel.
            max_y: The largest y-coordinate of each ray's source pixel.
            n_rays_per_image: The number of rays randomly sampled in each camera.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of each ray-point.
            max_depth: The maximum depth of each ray-point.
        """
        super().__init__()
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._n_rays_per_image = num_rays_per_image
        self._n_pts_per_ray = num_samples_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth

    def forward(self, cameras: CamerasBase, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # get the initial grid of image xy coords
        # of shape (batch_size, n_rays_per_image, 2)
        rays_xy = torch.cat(
            [
                torch.rand(
                    size=(batch_size, self._n_rays_per_image, 1),
                    dtype=torch.float32,
                    device=device,
                )
                * (high - low)
                + low
                for low, high in (
                    (self._min_x, self._max_x),
                    (self._min_y, self._max_y),
                )
            ],
            dim=2,
        )

        return self._xy_to_ray_bundle(
            cameras, rays_xy, self._min_depth, self._max_depth, self._n_pts_per_ray
        )


class LanternRaySampler(RaySampler):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        min_depth: float,
        max_depth: float,
        num_rays_per_image: int,
        num_samples_per_ray: int,
    ):
        super().__init__()

        self._grid_raysampler = NDCGridRaysampler(
            image_width=image_width,
            image_height=image_height,
            num_samples_per_ray=num_samples_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,            
        )
        
        self._montecarlo_raysampler = MonteCarloRaySampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            num_rays_per_image=num_rays_per_image,
            num_samples_per_ray=num_samples_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        self._ray_cache = {}

    def _normalize_raybundle(self, ray_bundle: RayBundle):
        """
        Normalizes the ray directions of the input `RayBundle` to unit norm.
        """

        ray_bundle.directions = torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        
        return ray_bundle

    def forward(
        self,
        cameras: CamerasBase,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
        Returns:
            A `RayBundle` object.
        """
        
        if self.training:
            ray_bundle = self._montecarlo_raysampler(cameras)            
        else:
            ray_bundle = self._grid_raysampler(cameras)

        ray_bundle = self._normalize_raybundle(ray_bundle)

        return ray_bundle
