import torch

from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from . import utils


class SingleImageDataset(Dataset):

    def __init__(self, filename, sidelength=256):
        super().__init__()

        if isinstance(sidelength, int):
            sidelength = 2*(sidelength,)

        self.sidelength = sidelength

        self.image = Image.open(filename)
        self.channels = len(self.image.mode)

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.sidelength = sidelength
        self.coordinates = utils.get_meshgrid(sidelength)

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        img = self.transform(self.image)
        img = img.permute(1, 2, 0).view(-1, self.channels)

        input_dict = {'idx': idx, 'coordinates': self.coordinates}
        output_dict = {'intensities': img}

        return input_dict, output_dict
