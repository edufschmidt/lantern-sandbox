import context

import torch
import matplotlib.pyplot as plt

from lantern.utils.io import load_point_clouds
from lantern.utils.visualization import prepare_depth_image
from lantern.utils.rendering import render_point_clouds

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import (
    look_at_view_transform,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__ == '__main__':

    image_size = (256,256)    
    paths = ['./data/pcl/bunny.xyz']

    R,T = look_at_view_transform(
        dist=1,
        elev=0,
        azim=0,
        degrees=True,
    )

    clouds = load_point_clouds(
        paths,
        device=device
    )

    cameras = PerspectiveCameras(        
        focal_length=10,
        principal_point=((0.0, 0.0),),
        R=R, T=T,
        device=device,
    )

    depths, intensities = render_point_clouds(
        clouds,
        cameras,
        device=device
    )

    f, axes = plt.subplots(1,2)
    
    depth, range = prepare_depth_image(depths[0, ..., 0].detach().cpu().numpy())

    axes[0].imshow(depth, cmap='jet')
    axes[1].imshow(intensities[0, ..., 0].detach().cpu().numpy(), cmap='gray')
    
    plt.show()