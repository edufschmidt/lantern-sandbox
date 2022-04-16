import torch

from torchsummary import summary

import lantern.models.activations as activations


class FullyConnectedBlock(torch.nn.Module):
    r"""Fully connected block with customizable activation.
    """

    def __init__(self,
                 in_features, out_features,
                 num_hidden_layers, hidden_features,
                 activation=activations.ReLU(),
                 linear_output_layer=False,
                 parse_input_fn=lambda input: input,
                 build_output_fn=lambda input, output: output,
                 dropout_prob=0.1,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)

        if parse_input_fn is None:
            raise Exception('undefined parse_input_fn')

        if build_output_fn is None:
            raise Exception('undefined build_output_fn')

        self.parse_input_fn = parse_input_fn
        self.build_output_fn = build_output_fn

        self.net = []

        self.init_weights_fcn = activation.init_weights_fcn
        self.init_first_layer_weights_fcn = \
            activation.init_first_layer_weights_fcn

        self.in_features = in_features
        self.hidden_features = hidden_features

        self.dropout_prob = dropout_prob

        self.net.append(torch.nn.Sequential(
            torch.nn.Linear(
                self.in_features,
                self.hidden_features,
                activation
            )
        ))

        for i in range(num_hidden_layers):
            self.net.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_features, hidden_features),
                    activation,
                    torch.nn.Dropout(p=self.dropout_prob),
                )
            )

        if linear_output_layer:
            self.net.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_features, out_features)
                )
            )
        else:
            self.net.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_features, out_features),
                    activation
                )
            )

        self.net = torch.nn.Sequential(*self.net)

        if self.init_weights_fcn is not None:
            self.net.apply(self.init_weights_fcn)

        if self.init_first_layer_weights_fcn is not None:
            self.net[0].apply(self.init_first_layer_weights_fcn)

    def forward(self, input, *args, **kwargs):
        input = self.parse_input_fn(input)
        return self.build_output_fn(input, self.net(input))

    def __str__(self):

        parse_input_fn = self.parse_input_fn
        build_output_fn = self.build_output_fn

        self.parse_input_fn = lambda x: x
        self.build_output_fn = lambda x, y: y

        s = summary(self, input_size=(self.in_features,), device='cuda')

        self.parse_input_fn = parse_input_fn
        self.build_output_fn = build_output_fn

        return s
