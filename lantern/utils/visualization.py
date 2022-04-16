import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pytorch3d.vis.plotly_vis import get_camera_wireframe

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter as TensorBoardSummaryWriter

from abc import ABC, abstractmethod

from visdom import Visdom


class SummaryWriter(TensorBoardSummaryWriter, ABC):
    r"""SummaryWriter
    """

    def __init__(self,
                 logger=None,
                 logs_dir='/tmp/lantern/experiments/latest/logs'):
                 
        super(SummaryWriter, self).__init__(log_dir=logs_dir)

        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', logs_dir])

        self.logger = logger

        if self.logger is not None:
            logger.success('TensorBoard started at:', self.tb.launch())

    @abstractmethod
    def write_summary(self):
        pass


class SimpleSummaryWriter(SummaryWriter):
    r"""SimpleLanternSummaryWriter
    """

    def __init__(self, trainer, logs_dir, logger=None):
        super().__init__(
            logger=logger,
            logs_dir=logs_dir
        )
        self.trainer = trainer

    def write_summary(self):
        summary = self.trainer.summary()
        self.add_scalar("loss", summary['loss'], summary['step'])

class SceneFittingSummaryWriter(SummaryWriter):
    r"""SceneFittingSummaryWriter
    """

    def __init__(self, trainer, logs_dir, logger=None):
        super().__init__(
            logger=logger,
            logs_dir=logs_dir
        )
        self.trainer = trainer

    def write_summary(self):
        
        summary = self.trainer.summary()

        input_dict, expected_output_dict = next(iter(self.trainer.dataloader))

        step = summary['step']

        self.trainer.model.eval()        
        predicted_output_dict = self.trainer.model.forward(input_dict)
        self.trainer.model.train()

        f1, ax = plt.subplots()
        ax.imshow(expected_output_dict['intensity']
                    [0,...].detach().cpu().numpy())

        f2, ax = plt.subplots()
        ax.imshow(expected_output_dict['disparity']
                    [0,...].detach().cpu().numpy(), cmap='jet')
        
        f3, ax = plt.subplots()
        ax.imshow(predicted_output_dict['disparity']
                    [0,...].detach().cpu().numpy(), cmap='jet')
                
        self.add_scalar("loss", summary['loss'], global_step=step)
        self.add_figure('(3) input_intensity', f1, global_step=step)
        self.add_figure('(2) input_disparity', f2, global_step=step)
        self.add_figure('(1) predicted_disparity', f3, global_step=step)

        plt.close(f1)
        plt.close(f2)
        plt.close(f3)
        
class ImageFittingSummaryWriter(SummaryWriter):
    r"""ImageFittingSummaryWriter
    """

    def __init__(self, trainer, logs_dir, logger=None):
        super().__init__(
            logger=logger,
            logs_dir=logs_dir
        )
        self.trainer = trainer

    def write_summary(self):
        summary = self.trainer.summary()
        self.add_scalar("loss", summary['loss'], summary['step'])
        print(summary)

def prepare_depth_image(img):   

    disp = 1/img
    disp[np.where(disp == -1)] = np.Inf
    
    vmin = np.min(disp[np.where(disp != 0)])
    vmax = np.max(disp)

    return disp, (vmin, vmax)

def depth_imshow(depth, ax=None):

    depth, range = prepare_depth_image(depth[0, ..., 0].detach().cpu().numpy())

    if ax is None:
        plt.imshow(depth, cmap='jet')
    else:
        ax.imshow(depth, cmap='jet')

def plot_mesh(viz: Visdom, vertices: torch.Tensor, faces: torch.Tensor, env=None, win=None, color='red', opacity=0.5):

    vertices = vertices
    faces = faces.verts_idx

    x = vertices[:,0].cpu().numpy()
    y = vertices[:,1].cpu().numpy()
    z = vertices[:,2].cpu().numpy()
    
    i = faces[:,0].cpu().numpy()
    j = faces[:,1].cpu().numpy()
    k = faces[:,2].cpu().numpy()
    
    X = np.c_[x, z, y]
    Y = np.c_[i, k, j]
    
    viz.mesh(X=X, Y=Y, env=env, win=win, opts={'color': color, 'opacity': opacity})

def plot_cameras(ax, cameras, color: str = "blue"):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """

    cam_wires_canonical = get_camera_wireframe().cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles
