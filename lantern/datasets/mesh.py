from pytorch3d.renderer.cameras import CamerasBase
import torch
from torch.utils.data import Dataset

from lantern.utils.io import load_meshes
from lantern.utils.rendering import render_meshes

from typing import List, Union

class MeshDataset(Dataset):
    def __init__(self,
                 path: Union[str, List[str]],
                 cameras: CamerasBase,
                 image_size=(100, 100),
                 device='cpu',
                 ):

        super().__init__()        

        self.cameras = cameras

        paths = [path]*len(cameras)

        self.meshes = load_meshes(
            paths,
            device=device
        )
             
        self.depth_images, self.intensity_images = render_meshes(
            self.meshes,
            self.cameras,
            image_size=image_size,
            device=device
        )

        self.disparity_images = 1 / self.depth_images

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):        
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_dict = {
            'R': self.cameras.R[idx],
            'T': self.cameras.T[idx],
            'focal_length': self.cameras.focal_length[idx],
            'principal_point': self.cameras.principal_point[idx],
        }
        
        output_dict = {
            'depth': self.depth_images[idx],
            'disparity': self.disparity_images[idx],            
            'intensity': self.intensity_images[idx],
        }
        
        return input_dict, output_dict
