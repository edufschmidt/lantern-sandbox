from pytorch3d.structures.pointclouds import Pointclouds
import torch
import numpy as np

from PIL import Image

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import Textures


def load_image(path):
    img = Image.open(path)

    print(img.format)
    print(img.mode)
    print(img.size)

    return img


def load_meshes(paths=[], device='cpu'):

    _verts, _faces, _verts_rgb = [], [], []    
    
    for path in paths:
        v, f, props = load_obj(path, device=device)
        _verts.append(v)
        _faces.append(f.verts_idx)
        _verts_rgb.append(torch.ones(1, v.size()[0], 3))        

    meshes = Meshes(
        verts=_verts,
        faces=_faces,
        textures = Textures(
            verts_rgb=torch.cat(_verts_rgb).to(device)
        ),
    )

    return meshes


def load_point_clouds(paths=[], device='cpu'):

    _points, _features = [], []    
    
    for path in paths:
        verts = np.loadtxt(path)
        p = torch.Tensor(verts[:,:3])
        f = torch.Tensor(verts[:,4:6])
        _points.append(p.to(device))
        _features.append(f.to(device))

    clouds = Pointclouds(
        points=_points,
        features=_features,
    )

    return clouds
