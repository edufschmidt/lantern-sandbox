import context

from torch.utils.data import DataLoader

import lantern.losses as losses
import lantern.datasets as datasets
import lantern.training as training
import lantern.models.activations as activations
import lantern.models as models

import configargparse

p = configargparse.ArgumentParser()

p.add_argument('--id', type=str, required=False,
               help='Identifier for this experiment',
               default='function_fitting')

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

    # Load dataset
    dataset = datasets.FunctionDataset(fcn=lambda x: 10 * x, num_samples=1000)

    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=1, pin_memory=True, num_workers=0)

    # Train model
    trainer = training.ModelTrainer(
        model,
        dataloader,
        output_prefix=args.id
    ).train(
        learning_rate=0.01,
        loss_fn=losses.function_mse_loss,
        steps_until_summary=100,
        num_epochs=10,
    )
