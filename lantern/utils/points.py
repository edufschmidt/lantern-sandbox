import os
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(0)

# Credits: https://github.com/PRBonn/range-mcl

def points_to_depth_image(points, vertical_fov=np.pi/6, image_size=(64,900), max_range=50, degrees=False):
    
    """ Perform spherical projection to map points in a point 
    cloud onto a depth map.
      Args:
        points: 3D points with LIDAR intensity
        vertical_fov_deg
      Returns:
        depth: map of pixel coordinates to depth [u,v] -> depth
        coords: map of pixels coordinates to their corresponding 3D points [u,v] -> (x, y, z, 1)
        intensities: map of pixel coordinates to their corresponding LIDAR intensity [u,v] -> I
        indices: map of pixel coordinates to the index of the corresponding point in the point cloud [u,v] -> idx
    """

    if degrees == True:
      vertical_fov *= np.pi / 180.0

    # The depth of each point is the norm
    depth = np.linalg.norm(points[:, :3], 2, axis=1)
    points = points[(depth > 0) & (depth < max_range)]
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    intensity = points[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(vertical_fov)/2.0) / \
        vertical_fov  # in [0.0, 1.0]

    image_height, image_width = image_size

    # scale to image size using angular resolution
    proj_x *= image_width  # in [0.0, width]
    proj_y *= image_height  # in [0.0, height]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(image_width - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(image_height - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    range_img = np.full((image_height, image_width), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    
    points_img = np.full((image_height, image_width, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
                           
    intensity_img = np.full((image_height, image_width), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)

    index_img = np.full((image_size[1], image_width), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    
    range_img[proj_y, proj_x] = depth
    
    points_img[proj_y, proj_x] = np.array(
        [scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    
    index_img[proj_y, proj_x] = indices
    
    intensity_img[proj_y, proj_x] = intensity

    return range_img, points_img, intensity_img, index_img
