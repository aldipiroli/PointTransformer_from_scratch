import numpy as np
import pyvista as pv
import torch

from point_transformer.utils.colors import SHAPENET_PART_COLORS


def plot_cls_preds(point_clouds, output_path="tmp.png", point_size=10, return_figure=False, title=""):
    plotter = pv.Plotter(off_screen=True)
    colors = ["red", "green", "blue", "orange", "purple"]

    offset = 0
    for i, points in enumerate(point_clouds):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy().squeeze()
        points = np.asarray(points)
        points = points[:, [0, 2, 1]]
        points_offset = points + np.array([offset, 0, 0])
        point_cloud = pv.PolyData(points_offset)
        color = colors[i % len(colors)]
        plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=point_size, color=color)
        offset += points[:, 0].max() - points[:, 0].min() + 1

    plotter.add_text(title, position="upper_edge", font_size=20, color="black")
    plotter.set_background("white")

    if return_figure:
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img.transpose(2, 0, 1)
    else:
        plotter.show(screenshot=output_path)
        plotter.close()
        return None


def plot_semseg_preds(
    point_clouds,
    labels_list,
    output_path="tmp.png",
    point_size=10,
    return_figure=False,
):
    plotter = pv.Plotter(off_screen=True)
    COLOR_MAP = (SHAPENET_PART_COLORS * 255).astype(np.uint8)
    offset = 0
    for i, (points, labels) in enumerate(zip(point_clouds, labels_list)):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        points = np.asarray(points).squeeze()
        labels = np.asarray(labels).astype(int).squeeze()
        points = points[:, [0, 2, 1]]
        points_offset = points + np.array([0, 0, offset])

        label_colors = COLOR_MAP[labels]
        pdata = pv.PolyData(points_offset)
        pdata["colors"] = label_colors

        plotter.add_points(
            pdata,
            scalars="colors",
            rgb=True,
            render_points_as_spheres=True,
            point_size=point_size,
        )
        offset += points[:, 0].max() - points[:, 0].min() + 5
    plotter.set_background("white")
    if return_figure:
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img.transpose(2, 0, 1)
    else:
        plotter.show(screenshot=output_path)
        plotter.close()
        return None
