from typing import Tuple

from numpy.core.numeric import Inf
import torch
import numpy as np

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings, 
    PointsRasterizationSettings,
    NormWeightedCompositor, 
    PointsRasterizer,
    MeshRasterizer,
    SoftPhongShader,
)

class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class PointsRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, compositor, **kwargs):
        super().__init__(**kwargs)
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)

        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        images = images.permute(0, 2, 3, 1)
        
        return images, fragments.zbuf


def render_meshes(
    meshes: Meshes,
    cameras: CamerasBase,
    image_size=(512,512),
    device='cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,        
        faces_per_pixel=1,
    )

    renderer = MeshRendererWithDepth(
        MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        SoftPhongShader(
            cameras=cameras,
            device=device,
        )
    )
    
    intensities, depths = renderer.forward(meshes)

    depths[depths < 0] = Inf

    return depths, intensities     

def render_point_clouds(
    clouds: Pointclouds,
    cameras: CamerasBase,
    image_size=(512,512),
    device='cpu'
):

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        points_per_pixel = 10,
        radius = 0.03,
    )
  
    renderer = PointsRendererWithDepth(
        rasterizer=PointsRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        compositor=NormWeightedCompositor(),
    )

    intensities, depths = renderer(
        clouds,
        gamma=(1e-2,),
    )

    return depths, intensities     

def generate_random_cameras(
    num_cameras,
    min_dist=0.5,
    max_dist=1.5,
    min_elev=-90,
    max_elev=90,
    min_azim=0,
    max_azim=360
    ):
    
    dist = np.random.uniform(low=min_dist, high=max_dist, size=num_cameras)
    azim = np.random.uniform(low=min_azim, high=max_azim, size=num_cameras)
    elev = np.random.uniform(low=min_elev, high=max_elev, size=num_cameras)

    _R,_T = [],[]

    for i in range(num_cameras):
        
        R,T = look_at_view_transform(
            dist=dist[i],
            elev=azim[i],
            azim=elev[i],
            degrees=True,
        )
        
        _R.append(R)        
        _T.append(T)

    return _R, _T