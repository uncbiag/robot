import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import numpy as np
import os
from scipy import misc
import robot.utils.finite_differences as fdt
from robot.utils.utils import t2np, point_to_grid


def read_png_into_numpy(file_path, name=None, visual=False):
    image = misc.imread(file_path, flatten=True)
    image = (image - image.min()) / (image.max() - image.min())
    if visual:
        plot_2d_img(image, name if name is not None else "image")
    return image


def read_png_into_standard_form(file_path, name=None, visual=False):
    image = read_png_into_numpy(file_path, name, visual)
    sz = [1, 1] + list(image.shape)
    image = image.reshape(*sz)
    spacing = 1.0 / (np.array(sz[2:]) - 1)
    return image, spacing


def save_3D_img_from_numpy(input, file_path, spacing=None, orgin=None, direction=None):
    output = sitk.GetImageFromArray(input)
    if spacing is not None:
        output.SetSpacing(spacing)
    if orgin is not None:
        output.SetOrigin(orgin)
    if direction is not None:
        output.SetDirection(direction)
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)
    sitk.WriteImage(output, file_path)


def transfer_point_to_grid_then_save(points, grid_size, file_path, spacing=None):
    img = point_to_grid(points, grid_size, return_np=True)
    img = np.squeeze(img)
    save_3D_img_from_numpy(img, file_path, spacing)


def compute_jacobi_map(map, spacing, record_path, fname_list, appendix="jacobi"):
    """
    compute determinant jacobi on transformatiomm map,  the coordinate should be canonical.

    :param map: the transformation map
    :param spacing: the map spacing
    :param use_01: infer the input map is in[0,1]  else is in [-1,1]
    :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels
    """

    if type(map) == torch.Tensor:
        map = map.detach().cpu().numpy()
    fd = fdt.FD_np(spacing)
    dfx = fd.dXc(map[:, 0, ...])
    dfy = fd.dYc(map[:, 1, ...])
    dfz = fd.dZc(map[:, 2, ...])
    jacobi_det = dfx * dfy * dfz
    jacobi_abs = -np.sum(jacobi_det[jacobi_det < 0.0])  #
    jacobi_num = np.sum(jacobi_det < 0.0)
    print(
        "print folds for each channel {},{},{}".format(
            np.sum(dfx < 0.0), np.sum(dfy < 0.0), np.sum(dfz < 0.0)
        )
    )
    print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
    print("the number of fold points for current batch is {}".format(jacobi_num))
    jacobi_abs_mean = jacobi_abs / map.shape[0]
    jacobi_num_mean = jacobi_num / map.shape[0]
    jacobi_abs_map = np.abs(jacobi_det)
    if record_path:
        jacobi_neg_map = np.zeros_like(jacobi_det)
        jacobi_neg_map[jacobi_det < 0] = 1
        for i in range(jacobi_abs_map.shape[0]):
            jacobi_img = sitk.GetImageFromArray(jacobi_abs_map[i])
            jacobi_neg_img = sitk.GetImageFromArray(jacobi_neg_map[i])
            spacing = spacing.astype(np.float64)
            jacobi_img.SetSpacing(np.flipud(spacing))
            jacobi_neg_img.SetSpacing(np.flipud(spacing))
            jacobi_saving = os.path.join(record_path, appendix)
            os.makedirs(jacobi_saving, exist_ok=True)
            pth = os.path.join(jacobi_saving, fname_list[i] + "_jacobi_img.nii")
            n_pth = os.path.join(jacobi_saving, fname_list[i] + "_jacobi_neg_img.nii")
            sitk.WriteImage(jacobi_img, pth)
            sitk.WriteImage(jacobi_neg_img, n_pth)
    return jacobi_abs_mean, jacobi_num_mean


def save_velocity(velocity, t, path=None):
    dim = len(velocity.shape) - 2
    velocity = velocity.detach()
    velocity = torch.sum(velocity ** 2, 1, keepdim=True)
    print(t)
    fname = str(t) + "velocity"
    if dim == 2:
        plot_2d_img(velocity[0, 0], fname, path)
    elif dim == 3:
        y_half = velocity.shape[3] // 2
        plot_2d_img(velocity[0, 0, :, y_half, :], fname, path)


def plot_2d_img(img, name, path=None):
    """
    :param img:  X x Y x Z
    :param name: title
    :param path: saving path
    :param show:
    :return:
    """
    sp = 111
    img = torch.squeeze(img)

    font = {"size": 10}

    plt.setp(plt.gcf(), "facecolor", "white")
    plt.style.use("bmh")

    plt.subplot(sp).set_axis_off()
    plt.imshow(t2np(img))  # ,vmin=0.0590, vmax=0.0604) #vmin=0.0590, vmax=0.0604
    plt.colorbar().ax.tick_params(labelsize=10)
    plt.title(name, font)
    if not path:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
        plt.clf()


def visualize_jacobi(phi, spacing, img=None, file_path=None, visual=True):
    """
    :param phi:  Bxdimx X xYxZ
    :param spacing: [sx,sy,sz]
    :param img: Bx1xXxYxZ
    :param file_path: saving path
    :return:
    """
    phi_sz = phi.shape
    n_batch = phi_sz[0]
    dim = phi_sz[1]
    phi_np = t2np(phi)
    if img is not None:
        assert phi.shape[0] == img.shape[0]
        img_np = t2np(img)
    fd = fdt.FD_np(spacing)
    dfx = fd.dXc(phi_np[:, 0, ...])
    dfy = fd.dYc(phi_np[:, 1, ...])
    dfz = 1.0
    if dim == 3:
        dfz = fd.dZc(phi_np[:, 2, ...])
    jacobi_det = dfx * dfy * dfz
    jacobi_neg = np.ma.masked_where(jacobi_det >= 0, jacobi_det)
    # jacobi_neg = (jacobi_det<0).astype(np.float32)
    jacobi_abs = -np.sum(jacobi_det[jacobi_det < 0.0])  #
    jacobi_num = np.sum(jacobi_det < 0.0)
    if dim == 3:
        print(
            "print folds for each channel {},{},{}".format(
                np.sum(dfx < 0.0), np.sum(dfy < 0.0), np.sum(dfz < 0.0)
            )
        )
    print("the jacobi_value of fold points for current map is {}".format(jacobi_abs))
    print("the number of fold points for current map is {}".format(jacobi_num))

    if visual:
        for i in range(n_batch):
            if dim == 2:
                sp = 111
                font = {"size": 10}
                plt.setp(plt.gcf(), "facecolor", "white")
                plt.style.use("bmh")
                plt.subplot(sp).set_axis_off()
                plt.imshow(t2np(img_np[i, 0]))
                plt.imshow(jacobi_neg[i], cmap="gray", alpha=1.0)
                plt.colorbar().ax.tick_params(labelsize=10)
                plt.title("img_jaocbi", font)
                if not file_path:
                    plt.show()
                else:
                    plt.savefig(file_path, dpi=300)
                    plt.clf()
            if dim == 3:
                if file_path:
                    jacobi_abs_map = np.abs(jacobi_det)
                    jacobi_img = sitk.GetImageFromArray(jacobi_abs_map[i])
                    pth = os.path.join(file_path)
                    sitk.WriteImage(jacobi_img, pth)
