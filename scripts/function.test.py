import context

import torch
from torch.utils.data import DataLoader

import lantern.losses as losses
import lantern.datasets as datasets
import lantern.models.activations as activations
import lantern.models as models

import configargparse

p = configargparse.ArgumentParser()

p.add_argument('--model_path', type=str, required=False,
               help='Path to the model to be loaded',
               default='/tmp/lantern/experiments/function_fitting/models/model.pth')

if __name__ == '__main__':

    args = p.parse_args()

    # Select model
    model = models.FullyConnectedBlock(
        in_features=1, out_features=1,
        num_hidden_layers=0, hidden_features=2,
        parse_input_fn=lambda input: input['coordinates'],
        build_output_fn=lambda input,output : {'values': output},
        activation=activations.ReLU()
    )

    # Load model parameters
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    model.eval()

    # Test model
    dataset = datasets.FunctionDataset(fcn=lambda x: 10 * x, num_samples=1000)
    dataloader = DataLoader(dataset)

    loss = 0.
    for input_dict, output_dict in dataloader:

        # Move tensors to the GPU
        input_dict = {key: value.cuda()
                      for key, value in input_dict.items()}

        output_dict = {key: value.cuda()
                       for key, value in output_dict.items()}

        predicted_output_dict = model(input_dict)

        loss += losses.function_mse_loss(predicted_output_dict, output_dict)['mse_loss']

    print('avg_mse = ', loss / len(dataloader))
