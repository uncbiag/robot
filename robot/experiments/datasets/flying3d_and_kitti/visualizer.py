import os
import pyvista as pv
from robot.utils.visualizer import color_adaptive

def kitti_plot(color="source",rgb=False):
    def plot(plotter, cloud, visualfea, **kwargs):
        if color == "source":
            cmap = "Reds"
            clim = [-2, 2]
        elif color == "target":
            cmap = "Blues"
            clim = [-2, 2]
        else:
            raise ValueError(f"Unknown color: {color}.")
        # Normalize to [0, 1]:
        visualfea = color_adaptive(visualfea)
        plotter.add_mesh(
            pv.PolyData(cloud),
            scalars=visualfea,
            point_size=10,
            render_points_as_spheres=True,
            rgb = rgb,
            lighting=True,
            cmap=cmap,
            clim=clim,
            style="points",
            ambient=0.5,
            **kwargs,
        )
    return plot


