from sys import flags
from time import sleep
import numpy as np

import context

import torch
from visdom import Visdom, server
import open3d as o3d

from lantern.utils.visualization import plot_cameras, plot_mesh
from lantern.utils.rendering import generate_random_cameras

from pytorch3d.io.obj_io import load_obj
from pytorch3d.renderer.cameras import PerspectiveCameras

from threading import Thread
import configargparse

p = configargparse.ArgumentParser()

p.add_argument('--path', type=str, required=False,
               help='Path to the mesh that will be used for training the model',
               default='./data/mesh/bunny.obj')

p.add_argument('--num_cameras', type=int, required=False,
               help='Number of views that will be used to generate the training set',
               default=1)

p.add_argument('--image_size', type=int, required=False,
               help='Size of the images that will be synthesized for each view',
               default=100)

p.add_argument('--visdom_host', type=str, required=False,
               help='Server address of the target to run the script on',
               default='localhost')

p.add_argument('--visdom_port', type=int, required=False,
               help='Port the Visdom server is running on',
               default=8097)

p.add_argument('--visdom_base_url', type=str, required=False,
               help='Base URL',
               default='')

p.add_argument('--visdom_username', type=str, required=False,
               help='Username for authenticating in the Visdom server',
               default='')

p.add_argument('--visdom_password', type=str, required=False,
               help='Password for authenticating in the Visdom server',
               default='')

p.add_argument('--visdom_use_incoming_socket', type=bool, required=False,
               help='Use incoming socket',
               default=False)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def setup_visualization(viz):
    assert viz.check_connection(timeout_seconds=3), \
        'No connection could be formed quickly'

if __name__ == '__main__':    

    args = p.parse_args()
    
    path = args.path

    visdom_host = args.visdom_host
    visdom_port = args.visdom_port
    visdom_base_url = args.visdom_base_url
    visdom_username = args.visdom_username
    visdom_password = args.visdom_password
    visdom_use_incoming_socket = args.visdom_use_incoming_socket
    
    # server.start_server(
    #         port=visdom_port,
    #         hostname=visdom_host,
    #         base_url=visdom_base_url,
    #         use_frontend_client_polling=True,
    # )

    try:
        viz = Visdom(
            port=visdom_port,
            server='http://localhost',
            base_url='',
            username=visdom_username,
            password=visdom_password,
            use_incoming_socket=visdom_use_incoming_socket,
        )
        setup_visualization(viz)
    except Exception as e:
        print(e)

    vertices, faces, props = load_obj(path, device=device)
    plot_mesh(viz, vertices, faces, win='bunny')

    R, T = generate_random_cameras(
        30,
        min_dist=1.5,
        max_dist=2.0
    )

    cameras = PerspectiveCameras(
        focal_length=10,
        principal_point=((0.0, 0.0),),
        R=torch.cat(R), T=torch.cat(T),
        device=device,
    )

    # handle_cams = plot_cameras(ax, cameras, color="#0000ff")    

    # ax.set_xlim3d([-2, 2])
    # ax.set_ylim3d([-2, 2])
    # ax.set_zlim3d([-2, 2])
    
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")   

    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0),
    # )

    # plt.show()
