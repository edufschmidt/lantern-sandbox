import torch

# A loss function returns a dict containing one or more losses, which
# will be all summed up and taken into account when training.


def function_mse_loss(predicted_output_dict, expected_output_dict):

    y = expected_output_dict['values']
    y_hat = predicted_output_dict['values']

    return {'mse_loss': ((y_hat - y) ** 2).mean()}


def image_mse_loss(predicted_output_dict, expected_output_dict):

    y = expected_output_dict['values']
    y_hat = predicted_output_dict['values']

    return {'mse_loss': ((y_hat - y) ** 2).mean()}


def pcl_mse_loss(predicted_output_dict, expected_output_dict):

    y = expected_output_dict['values']
    y_hat = predicted_output_dict['values']

    return {'mse_loss': ((y_hat - y) ** 2).mean()}

def dense_geometric_loss(predicted_output_dict, expected_output_dict):

    # TODO: normalize based on depth variance

    y = expected_output_dict['depth']
    y_hat = predicted_output_dict['depth']

    loss = (torch.abs(y-y_hat)).mean()

    return {'geometric_loss': loss}


def dense_photometric_loss(predicted_output_dict, expected_output_dict):

    y = expected_output_dict['intensities']
    y_hat = predicted_output_dict['intensities']

    loss = (torch.abs(y-y_hat)).mean()

    return {'photometric_loss': loss}


def photogeometric_loss(predicted_output_dict, expected_output_dict):

    photo_loss_dict = dense_photometric_loss(
        predicted_output_dict, expected_output_dict)

    geo_loss_dict = dense_geometric_loss(
        predicted_output_dict, expected_output_dict)

    return {**photo_loss_dict, **geo_loss_dict}
