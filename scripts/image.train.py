import context

from torch.utils.data import DataLoader

import lantern.models as models
import lantern.models.activations as activations
from lantern.utils.visualization import ImageFittingSummaryWriter

import lantern.losses as losses
import lantern.datasets as datasets
import lantern.training as training

import configargparse

p = configargparse.ArgumentParser()

p.add_argument('--id', type=str, required=False,
               help='Identifier for this experiment',
               default='img_fitting')

p.add_argument('--img_path', type=str, required=False,
               help='Image to be used for training',
               default='./data/image/mug.jpeg')

if __name__ == '__main__':

    args = p.parse_args()

    resolution = (96, 128)

    # Load dataset
    dataset = datasets.SingleImageDataset(
        filename=args.img_path, sidelength=resolution)

    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=1, pin_memory=True, num_workers=0)

    input, output = dataset[0]

    # import matplotlib.pyplot as plt
    # plt.imshow(output['intensities'].reshape((*resolution, 3)))
    # plt.show()

    # sys.exit()

    # Select model
    model = models.FullyConnectedBlock(
        in_features=1, out_features=1,
        num_hidden_layers=0, hidden_features=2,
        activation=activations.Sine(),
        parse_input_dict_fn=lambda input_dict: input_dict['coordinates'],
        build_output_dict_fn=lambda input, output: {
            'coordinates': input,
            'intensities': output
        },
    )

    # Train model
    trainer = training.ModelTrainer(
        model,
        dataloader,
        output_prefix=args.id,
        summary_writer_class=ImageFittingSummaryWriter,
    ).train(
        learning_rate=0.01,
        loss_fn=losses.image_mse_loss,
        steps_until_summary=5,
        num_epochs=1000,
    )
