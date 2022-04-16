import torch

from .fully_connected_block import FullyConnectedBlock
from .embeddings import HarmonicEmbedding
from . import activations

class ImplicitSceneModel(torch.nn.Module):
    def __init__(self,
                 parse_input_fn=lambda input: input,
                 build_output_fn=lambda input, output: output,
                 device='cpu',
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        
        if parse_input_fn is None:
            raise Exception('undefined parse_input_fn')

        if build_output_fn is None:
            raise Exception('undefined build_output_fn')

        self.parse_input_dict_fn = parse_input_fn
        self.build_output_dict_fn = build_output_fn

        num_harmonic_functions = 6

        self.harmonic_encoder = HarmonicEmbedding(
            num_harmonic_functions=num_harmonic_functions,
            *args,
            **kwargs,
        )

        self.fully_connected_block = FullyConnectedBlock(
            in_features=num_harmonic_functions*6, out_features=1,
            num_hidden_layers=4, hidden_features=256,
            activation=activations.ReLU(),
            dropout_prob=0.0,
            *args,
            **kwargs
        )

        self.to(device)

    def forward(
        self,
        input_dict: dict,
        *args,
        **kwargs,
    ) -> dict:
        """
        The forward function accepts the parameterizations of 3D points
        sampled along projection rays. The forward pass is responsible
        for attaching a 1D scalar representing the point's opacity.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    with the ray direction vectors in world coords.
                sampled_depths: Tensor of shape
                `(minibatch, ..., num_points_per_ray)` containing the lengths
                 at which the rays are sampled.            

        Returns:
            A dict containing a tensor of shape
            `(minibatch, ..., num_points_per_ray, 1)`
            denoting the opacity of each ray point.

        """

        ray_points_world: torch.Tensor = self.parse_input_dict_fn(input_dict)

        # encoded_coordinates.shape = [minibatch,...,num_harmonic_functions*6]
        encoded_coordinates: torch.Tensor = self.harmonic_encoder.forward(
            ray_points_world,
            *args,
            **kwargs,
        )

        # output.shape = [minibatch, ... , self.n_hidden_neurons_xyz]
        output = self.fully_connected_block.forward(
            encoded_coordinates,
            *args,
            **kwargs,
        )

        output_dict = self.build_output_dict_fn(encoded_coordinates, output)

        return output_dict
