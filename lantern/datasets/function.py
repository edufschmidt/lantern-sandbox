import torch
from torch.utils.data import Dataset


class FunctionDataset(Dataset):
    """A dataset that emulates data coming from an
    arbitrary function.
    """

    def __init__(self, fcn=lambda x: 2*x+1, num_samples=100):
        super().__init__()

        num_samples = num_samples

        x_values = [i for i in range(num_samples)]
        y_values = [fcn(i) for i in x_values]

        self.samples = [
            (
                {'idx': i, 'coordinates': torch.tensor(
                    [x_values[i]], dtype=torch.float)},
                {'values': torch.tensor([y_values[i]])}
            )
            for i in range(num_samples)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
