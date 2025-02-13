import context

import torch
from matplotlib import pyplot as plt

import lantern.datasets as datasets

from lantern.models import LanternRaySampler
from lantern.utils.rendering import generate_random_cameras

from pytorch3d.renderer.cameras import PerspectiveCameras

from torch.utils.data import DataLoader

import configargparse


p = configargparse.ArgumentParser()

p.add_argument('--id', type=str, required=False,
               help='Identifier for this experiment',
               default='mesh_fitting')

p.add_argument('--path', type=str, required=False,
               help='Path to the mesh that will be used for training the model',
               default='./data/mesh/bunny.obj')

p.add_argument('--num_cameras', type=int, required=False,
               help='Number of views that will be used to generate the training set',
               default=1)

p.add_argument('--image_size', type=int, required=False,
               help='Size of the images that will be synthesized for each view',
               default=100)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__ == '__main__':

    args = p.parse_args()

    path = args.path
    num_cameras = args.num_cameras
    image_size = (args.image_size,)*2

    R, T = generate_random_cameras(num_cameras)

    cameras = PerspectiveCameras(
        focal_length=10,
        principal_point=((0.0, 0.0),),
        R=torch.cat(R), T=torch.cat(T),
        device=device,
    )

    dataset = datasets.MeshDataset(
        path,
        cameras,
        image_size=image_size,
        device='cuda'
    )

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        pin_memory=False,
        num_workers=0
    )

    # Generate a RayBundle using the LanternRaySampler
    # object and performs depth computation with trimesh
    # raycasting, therefore replacing the implicit model
    # with a mesh.

    ray_sampler = LanternRaySampler(
        image_width=image_size[0],
        image_height=image_size[1],
        min_depth=0.1,
        max_depth=10,
        num_rays_per_image=100,
        num_samples_per_ray=10,
    )

    # Switch to inference model so that
    # our RaySampler outputs a dense RayBundle.
    ray_sampler.eval()

    ray_bundle = ray_sampler.forward(cameras=cameras)

    import trimesh
    mesh = trimesh.load(path)

    points = ray_bundle.origins_world.detach().cpu().numpy().reshape(-1, 3)
    directions = ray_bundle.directions.detach().cpu().numpy().reshape(-1, 3)

    depth_image = trimesh.proximity.longest_ray(mesh, points, directions)

    depths = depth_image.reshape(101, 101)

    # Show the depth map generated by trimesh
    # alongside the intensity image in the dataset.

    f, axes = plt.subplots(1, 2)

    input_dict, output_dict = dataset[0]
    axes[1].imshow(output_dict['intensity']
                   [...].detach().cpu().numpy(), cmap='gray')

    axes[0].imshow(depths, cmap='jet')

    plt.show()
