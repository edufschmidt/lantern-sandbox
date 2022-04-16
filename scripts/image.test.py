import context

import torch

import lantern.models as models
import lantern.models.activations as activations

import configargparse

p = configargparse.ArgumentParser()

p.add_argument('--model_path', type=str, required=False,
               help='Path to the model to be loaded',
               default='/tmp/lantern/experiments/image_fitting/models/model.pth')

if __name__ == '__main__':

    args = p.parse_args()

    # Select model
    model = models.FullyConnectedBlock(
        in_features=1, out_features=1,
        num_hidden_layers=0, hidden_features=2,
        activation=activations.ReLU()
    )

    # Load model parameters
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    model.eval()

    # TODO: Test model
