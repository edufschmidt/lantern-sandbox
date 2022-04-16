import context

import torch
from matplotlib import pyplot as plt

import lantern.datasets as datasets
import lantern.training as training

from pytorch3d.renderer.cameras import PerspectiveCameras
from lantern.utils.rendering import look_at_view_transform
from lantern.models import Lantern, LanternInput, LanternOutput
from lantern.utils.visualization import SceneFittingSummaryWriter
from lantern.utils.rendering import generate_random_cameras
from lantern.utils.rays import sample_images_at

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
               default=64)

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
    
    R, T = generate_random_cameras(3)
    
    # R,T = look_at_view_transform(
    #     dist=1,
    #     elev=0,
    #     azim=0,
    #     degrees=True,
    # )
    # R, T = [R], [T]
    
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

    # ------------------

    def build_input(input_dict) -> LanternInput:
        return LanternInput(
            R=input_dict['R'],
            T=input_dict['T'],
            focal_length=input_dict['focal_length'],
            principal_point=input_dict['principal_point'],
        )

    def build_output(output: LanternOutput):
        return {
            'rays': output.rays,
            'disparity': output.rendered_disparity,
            'intensity': output.rendered_intensity,
        }

    model = Lantern(
        image_size=image_size,
        num_samples_per_ray=10,
        num_rays_per_image=100,
        min_depth=0.1, max_depth=1.0,
        build_input_fn=build_input,
        build_output_fn=build_output,
        device=device,
    )   

    def loss_fn(predicted_output_dict, expected_output_dict):            
        
        rays = predicted_output_dict['rays']        
        expected_disparity = expected_output_dict['disparity']
        predicted_disparity = predicted_output_dict['disparity']        
        
        # Sample the expected depth image at coordinates
        # defined by the ray origins so that we can compare
        # their values with those predicted by the model.
        sampled_expected_depth = sample_images_at(
            expected_disparity,
            rays.origins_image,
        )        
        
        y, y_hat = sampled_expected_depth, predicted_disparity        

        return {            
            'geometry_loss': torch.mean((y - y_hat) ** 2),
        }

    # Train model
    trainer = training.ModelTrainer(
        model,
        dataloader,
        output_prefix=args.id,
        summary_writer_class=SceneFittingSummaryWriter,
    )

    # ------------------

    trainer.train(
        learning_rate=0.01,
        loss_fn=loss_fn,
        steps_until_summary=100,
        num_epochs=10000,
    )
