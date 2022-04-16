import context

import time
import torch

import lantern.losses as losses
import lantern.datasets as datasets
import lantern.training as training
from lantern.rendering.rendering import Lantern

from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from torch.utils.data import DataLoader

p = configargparse.ArgumentParser()

p.add_argument('--id', type=str, required=False,
               help='Identifier for this experiment',
               default='scene_fitting')

if __name__ == '__main__':

    args = p.parse_args()

    dataset = datasets.MeshDataset(path='../data/pcl/sphere.xyz')

    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=1, pin_memory=True, num_workers=0)
    
    model = Lantern(
        image_size=(100,100),
        num_rays_per_image=200,
        num_samples_per_ray=10,
        min_depth=0.1,
        max_depth=10.0,
        device='cuda',
    )    

    cameras = FoVPerspectiveCameras(
        fov=60,
        znear=1,
        zfar=100,
        aspect_ratio=1,
        degrees=True,
        R=torch.eye(3)[None],
        T=torch.zeros(1, 3),
        device='cuda'
    )

    model.train()

    # Train model
    trainer = training.ModelTrainer(
        model,
        dataloader,
        output_prefix=args.id,
    ).train(
        learning_rate=0.01,
        loss_fn=losses.dense_geometric_loss,
        steps_until_summary=100,
        num_epochs=3,
    )

    model.eval()
    with torch.no_grad():
        val_nerf_out, val_metrics = model(
            camera_idx if cfg.data.precache_rays else None,
            val_camera,
            val_image,
        )


    # depth_imshow(rendered_depth_image)
