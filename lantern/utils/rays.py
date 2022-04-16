import torch


class RayBundle():
    """
    RayBundle parametrizes points along projection rays by storing ray
    `origins`, `directions` vectors and `lengths` at which the ray-points
    are sampled, together with the xy-locations (`xys`) of the ray pixels
    are stored as well.
    """

    def __init__(
        self,
        origins_image: torch.Tensor,
        origins_world: torch.Tensor,
        directions: torch.Tensor,
        sampled_depths: torch.Tensor,
    ) -> None:
        self.origins_image = origins_image
        self.origins_world = origins_world
        self.directions = directions
        self.sampled_depths = sampled_depths

        # This will be populated later on
        self.sampled_densities: torch.Tensor = torch.zeros_like(
            self.sampled_depths
        )

        # Assumes that each ray is sampled uniformly
        self.intersample_distances: torch.Tensor = torch.sub(
            sampled_depths[:, ..., 1],
            sampled_depths[:, ..., 0]
        )[:, ..., None]

    def samples(
        self,
    ) -> torch.Tensor:
        """
        Converts rays parametrized with origins and directions to
        3D samples in world coordinates by extending each ray according
        to the corresponding depth:

        Args:
            origins_world: A tensor of shape `(..., 3)`
            directions: A tensor of shape `(..., 3)`
            sampled_depths: A tensor of shape `(..., num_points_per_ray)`

        Returns:
            sampled_coordinates_world: A tensor of shape
            `(..., num_points_per_ray, 3)` containing the
             points sampled along each ray.
        """

        sampled_coordinates_world = (
            self.origins_world[..., None, :]
            + self.sampled_depths[..., :, None] * self.directions[..., None, :]
        )

        return sampled_coordinates_world

    def _validate(
        self,
    ):
        """
        Validate the shapes of RayBundle variables
        `rays_origins`, `rays_directions`, and `rays_lengths`.
        """
        ndim = self.origins_world.ndim
        if any(r.ndim != ndim for r in (self.directions, self.sampled_depths)):
            raise ValueError(
                "rays_origins, rays_directions and rays_lengths"
                + " have to have the same number of dimensions."
            )

        if ndim <= 2:
            raise ValueError(
                "rays_origins, rays_directions and rays_lengths"
                + " have to have at least 3 dimensions."
            )

        spatial_size = self.origins_world.shape[:-1]
        if any(spatial_size != r.shape[:-1] for r in (self.directions,
                                                      self.sampled_depths)):
            raise ValueError(
                "The shapes of origins_world, directions and sampled_depths"
                + " may differ only in the last dimension."
            )

        if any(r.shape[-1] != 3 for r in
               (self.origins_world, self.directions)):
            raise ValueError(
                "The size of the last dimension of origins_world/directions"
                + "has to be 3."
            )

def sample_images_at(
    images: torch.Tensor,
    coordinates: torch.Tensor,
):
    """
    Given a set of pixel locations `coordinates` this method samples the tensor
    `images` at the respective 2D locations.

    This function is used in order to extract the colors from ground truth images
    that correspond to the colors rendered using a Monte Carlo rendering.

    Args:
        images: A tensor of shape `(batch_size, ..., num_channels)`.
        coordinates: A tensor of shape `(batch_size, S_1, ..., S_N, 2)`.

    Returns:
        images_sampled: A tensor of shape `(batch_size, S_1, ..., S_N, num_channels)`
            containing `images` sampled at `coordinates`.
    """
    
    batch_size = images.shape[0]    
    num_channels = images.shape[-1]    
    image_size = coordinates.shape[1:-1]

    # The coordinate grid convention for grid_sample
    # has both x and y directions inverted.
    xy_grid = -coordinates.view(batch_size, -1, 1, 2).clone()

    images_sampled = torch.nn.functional.grid_sample(
        images.permute(0, 3, 1, 2),
        xy_grid,
        align_corners=True,
        mode="bilinear",
    )
    
    samples = images_sampled.permute(0, 2, 3, 1).view(batch_size, *image_size, num_channels)

    return samples
