import torch
import numpy as np


def get_meshgrid(sidelength, dimensions=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    Input:
    - sidelength: tuple with the side lengths of the
    resulting tensor (n1,n2,...), or scalar in case
    lengths are the same along all dimensions.

    Return:
    - torch.Tensor (signal_dimension, sidelength, sidelength)
    """

    if isinstance(sidelength, tuple):
        dimensions = len(sidelength)

    if isinstance(sidelength, int):
        sidelength = dimensions * (sidelength,)

    coords = torch.meshgrid(
        *[torch.Tensor(np.arange(len)) for len in sidelength]
    )

    # Normalize coordinates between (-1,1)
    coords = [2*((coords[i] / max(sidelength[i] - 1, 1)) - 0.5)
              for i in range(dimensions)]

    coords = torch.stack(
        [coords[i].flatten() for i in range(dimensions)]
    )

    return coords.transpose(0, 1)
