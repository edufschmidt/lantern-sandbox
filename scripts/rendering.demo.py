import context

import time
import torch
from matplotlib import pyplot as plt

from lantern.models.rendering import Lantern, LanternInput, LanternOutput
from lantern.utils.visualization import depth_imshow
from lantern.utils.rendering import generate_random_cameras

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__ == '__main__':

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
            'depth': output.rendered_depth,
            'intensity': output.rendered_intensity,
        }
    
    model = Lantern(
        image_size=(128,128),
        num_samples_per_ray=10,
        num_rays_per_image=100,
        min_depth=0.1,
        max_depth=10.0,
        build_input_fn=build_input,
        build_output_fn=build_output,
        device=device,
    )

    model.eval()

    R, T = generate_random_cameras(1)

    start = time.time()

    predicted_output_dict = model.forward(
        {
            'R': torch.cat(R),
            'T': torch.cat(T),
            'focal_length': 10.0,
            'principal_point': ((0.0, 0.0),),
        }
    )
    
    print('Execution time in seconds: ' + str(time.time() - start))

    depth = predicted_output_dict['depth']

    f, axes = plt.subplots(1, 1)
  
    depth_imshow(depth) 

    plt.show()