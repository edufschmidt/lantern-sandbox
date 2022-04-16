import context

import os
import numpy as np

import matplotlib.pyplot as plt
from lantern.utils.points import points_to_depth_image
from lantern.utils.visualization import depth_imshow

np.random.seed(0)

def save_depth_image(ranges, path):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(9, 0.64)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(ranges, aspect='equal')
    fig.savefig(os.path.join(path))
    plt.close()


if __name__ == '__main__':

    all_points = np.loadtxt('./data/pcl/house.xyz')

    scanned_points = all_points

    points_with_intensity = np.append(
        scanned_points, np.ones((len(scanned_points), 1)), axis=1)

    ranges, vertices, intensities, indices = points_to_depth_image(
        points_with_intensity,
        vertical_fov=60.0,
        image_size=(48,256),
        degrees=True
    )    

    f, axes = plt.subplots(1, 1)
    plt.imshow(ranges, cmap='jet')
    plt.show()
