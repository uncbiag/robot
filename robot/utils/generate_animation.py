"""
animation generation for deformation models

support displacement, spline and LDDMM, prealign
"""
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
from robot.experiments.datasets.lung.visualizer import lung_plot
from robot.utils.visualizer import visualize_source_flowed_target_overlap, visualize_point_pair_overlap, default_plot
from PIL import Image
from robot.datasets.data_utils import get_obj
from robot.global_variable import Shape
from robot.utils.obj_factory import obj_factory
import numpy as np
from scipy.linalg import logm, expm
import torch
import torch.nn as nn
from robot.utils.module_parameters import ParameterDict
from robot.datasets.vtk_utils import read_vtk
from robot.shape.shape_pair_utils import create_shape_pair
import pyvista as pv
from pygifsicle import optimize
import pykeops

# cache_path = "/home/zyshen/keops_cache"
# os.makedirs(cache_path, exist_ok=True)
# pykeops.set_bin_folder(cache_path)  # change the build folder
# print("change keops cache path into  {}".format(pykeops.config.bin_folder))


class FlowModel(nn.Module):
    def __init__(self, opt):
        super(FlowModel, self).__init__()
        self.dim = 3
        self.opt = opt
        self.model_type = opt[
            ("model_type", "liner_interp", "model type 'liner_interp'/'lddmm_shooting'")
        ]
        self.t_list = opt[("t_list", [1.0], "time list")]
        if self.model_type == "affine_interp":
            self.flow_model = self.affine_interp
        elif self.model_type == "linear_interp":
            self.flow_model = self.linear_interp
        elif self.model_type == "lddmm_shooting":
            self.init_lddmm(opt["lddmm"])
            self.flow_model = self.lddmm_interp

    def affine_interp(self, shape_pair):
        toflow = shape_pair.source
        toflow_points = toflow.points
        reg_param = shape_pair.reg_param
        reg_param_list = affine_interpolation(reg_param, self.t_list)
        interp_shape_list = []
        for reg_param in reg_param_list:
            points = torch.bmm(toflow_points, reg_param[:, : self.dim, :])
            points = reg_param[:, self.dim :, :].contiguous() + points
            interp_shape = Shape()
            interp_shape.set_data_with_refer_to(points, toflow)
            interp_shape_list.append(interp_shape)
        return interp_shape_list

    def linear_interp(self, shape_pair):
        interp_shape_list = []
        flowed = shape_pair.flowed
        toflow = shape_pair.source
        for t in self.t_list:
            interp_points = (flowed.points - toflow.points) * t + toflow.points
            interp_shape = Shape()
            interp_shape.set_data_with_refer_to(interp_points, toflow)
            interp_shape_list.append(interp_shape)
        return interp_shape_list

    def init_lddmm(self, lddmm_opt):
        from robot.modules_reg.ode_int import ODEBlock
        from robot.modules_reg.module_lddmm import LDDMMHamilton, LDDMMVariational

        self.module_type = lddmm_opt[
            ("module", "hamiltonian", "lddmm module type: hamiltonian or variational")
        ]
        assert self.module_type in ["hamiltonian", "variational"]
        self.lddmm_module = (
            LDDMMHamilton(lddmm_opt[("hamiltonian", {}, "settings for hamiltonian")])
            if self.module_type == "hamiltonian"
            else LDDMMVariational(
                lddmm_opt[("variational", {}, "settings for variational")]
            )
        )
        self.lddmm_kernel = self.lddmm_module.kernel
        self.integrator_opt = lddmm_opt[("integrator", {}, "settings for integrator")]
        self.integrator = ODEBlock(self.integrator_opt)
        self.integrator.set_func(self.lddmm_module)

    def lddmm_interp(self, shape_pair):
        toflow = shape_pair.source
        toflow_points = toflow.points
        momentum = shape_pair.reg_param
        momentum = momentum.clamp(
            -0.5, 0.5
        )  # todo  this is a temporal setting for data normalized into [0,1] should be set in class attribute
        if shape_pair.dense_mode:
            self.lddmm_module.set_mode("shooting")
            _, flowed_control_points_list = self.integrator.solve(
                (momentum, shape_pair.control_points)
            )
            flowed_points_list = flowed_control_points_list
        else:
            self.lddmm_module.set_mode("flow")
            _, flowed_control_points_list, flowed_points_list = self.integrator.solve(
                (momentum, shape_pair.control_points, toflow_points)
            )
        flowed_list = [
            Shape().set_data_with_refer_to(flowed_points, toflow)
            for flowed_points in flowed_points_list
        ]
        return flowed_list

    def forward(self, shape_pair):
        flowed_list = self.flow_model(shape_pair)
        return flowed_list


class ShapePair:
    def __init__(self, source_path, flowed_path, **kwargs):
        super(ShapePair, self).__init__()
        self.path_dict = {"source": source_path, "flowed": flowed_path}
        self.path_dict.update(kwargs)
        self.init_shape_pair()

    def init_shape_pair(self):
        pass





def init_reg_param(points_path, is_affine=False):
    if not is_affine:
        reg_param_dict = read_vtk(points_path)
        return torch.Tensor(reg_param_dict["reg_param_vector"])[None]


def affine_interpolation(input_affine, t_list):
    """

    :param affine_matrix: 4x3
    :param t_list:
    :return:
    """
    device = input_affine.device
    affine_matrix = input_affine.detach().cpu().numpy().squeeze()
    affine_matrix = np.transpose(affine_matrix)  # 3x4
    affine_matrix = np.concatenate([affine_matrix, np.array([[0, 0, 0, 1]])], 0)
    logB = logm(affine_matrix)
    affine_list = [expm(t * logB)[:3, :] for t in t_list]
    affine_list = [np.transpose(affine) for affine in affine_list]
    affine_list = [torch.Tensor(affine[None]).to(device) for affine in affine_list]
    return affine_list





def visualize_animation(
    shape1_list,
    shape2_list,
    title1_list,
    title2_list,
    saving_capture_path_list=None,
    camera_pos_list=None,
    shape1_plot_func = default_plot(cmap="magma"),
    shape2_plot_func = default_plot(cmap="viridis"),
    light_mode="none",
    show=False,
):
    from robot.utils.visualizer import format_input

    for shape1, shape2, saving_capture_path, title1, title2, camera_pos in zip(
        shape1_list,
        shape2_list,
        saving_capture_path_list,
        title1_list,
        title2_list,
        camera_pos_list,
    ):
        points1, points2 = shape1.points, shape2.points
        feas1, feas2 = shape1.weights, shape2.weights
        points1 = format_input(points1)
        points2 = format_input(points2)
        feas1 = format_input(feas1)
        feas2 = format_input(feas2)

        visualize_point_pair_overlap(
            points1,
            points2,
            feas1,
            feas2,
            title1,
            title2,
            shape1_plot_func,
            shape2_plot_func,
            saving_capture_path=saving_capture_path,
            light_mode=light_mode,
            camera_pos=camera_pos,
            show=show
        )



def generate_gif(img_capture_path_list, saving_path):
    import imageio

    images = []
    for img_capture_path in img_capture_path_list:
        image = imageio.imread(img_capture_path)
        image = Image.fromarray(image).resize((1250, 512))
        images.append(image)
    imageio.mimsave(saving_path, images)
    optimize(saving_path)  # For overwriting the original one



def camera_pos_interp(pos1, pos2, t_list):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos_interp_list = [(pos2 - pos1) * t + pos1 for t in t_list]
    pos_interp_list = [[tuple(sub_pos) for sub_pos in pos] for pos in pos_interp_list]
    return pos_interp_list





if __name__ == "__main__":
    import os, sys
    import subprocess

    os.environ["DISPLAY"] = ":99.0"
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.environ["PYVISTA_USE_IPYVTK"] = "true"
    bashCommand = "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & sleep 3"
    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
    process.wait()
    sys.path.insert(0, os.path.abspath("../.."))
    from robot.experiments.datasets.lung.lung_dataloader_utils import lung_reader


    def init_shpae(points_path):
        if points_path:
            shape = Shape()
            get_shape = get_obj(lung_reader())
            shape_dict, _ = get_shape(points_path)
            shape.set_data(**shape_dict)
            return shape
        else:
            return None

    COPD_ID = {
        "copd6": "12042G",
        "copd7": "12105E",
        "copd8": "12109M",
        "copd9": "12239Z",
        "copd10": "12829U",
        "copd1": "13216S",
        "copd2": "13528L",
        "copd3": "13671Q",
        "copd4": "13998W",
        "copd5": "17441T"
    }
    folder_path = "/playpen-raid1/zyshen/data/lung_expri/model_eval/draw/deep_flow_prealign_pwc_lddmm_4096_new_60000_8192_aniso_rerun2/records/3d/test_epoch_-1"
    # folder_path ="/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/draw/deep_flow_prealign_pwc_lddmm_4096_new_60000_8192_aniso_rerun2/records/3d/test_epoch_-1"
    case_id_list = ["copd{}".format(i) for i in range(1,11)]
    for case_id in case_id_list:
        output_folder_name = "gif21"
        output_folder = os.path.join(folder_path, output_folder_name, case_id)
        os.makedirs(output_folder, exist_ok=True)
        total_captrue_path_list = []

        source_path = os.path.join(folder_path, COPD_ID[case_id] + "_source.vtk")
        target_path = os.path.join(folder_path, COPD_ID[case_id] + "_target.vtk")
        control_path = os.path.join(folder_path, COPD_ID[case_id] + "_control.vtk")
        landmark_path = os.path.join(folder_path, COPD_ID[case_id] + "_landmark_gf_target.vtk")
        prealigned_path = os.path.join(folder_path, COPD_ID[case_id] + "__prealigned.vtk")
        reg_param_path = os.path.join(folder_path, COPD_ID[case_id] + "_reg_param.vtk")
        prealign_reg_param_path = os.path.join(
            folder_path, COPD_ID[case_id] + "_prealigned_reg_param.npy"
        )
        nonp_path = os.path.join(folder_path, COPD_ID[case_id] + "_flowed.vtk")
        nonp_gf_path = os.path.join(folder_path, COPD_ID[case_id] + "__gf_flowed.vtk")
        source = init_shpae(source_path)
        target = init_shpae(target_path)
        control = init_shpae(control_path)
        prealigned = init_shpae(prealigned_path)
        nonp = init_shpae(nonp_path)
        nonp_gf = init_shpae(nonp_gf_path)
        prealign_reg_param = torch.Tensor(np.load(prealign_reg_param_path))[None]

        # 0 preview
        output_folder = os.path.join(folder_path, output_folder_name, case_id, "preview")
        os.makedirs(output_folder, exist_ok=True)
        stage_name = "preview"
        camera_pos_start = [
            (2.195036914518257, 6.095982604001324, 1.93352845755372),
            (0.0, 0.0, 0.0),
            (-0.3904490200358431, -0.14763278801531093, 0.9087101422653299),
        ]
        #    camera_pos_start = [(-1.119039073715163, -6.374273410328297, 1.9580891967516285),
        # (0.0, 0.0, 0.0),
        # (0.6703830032209878, 0.10782789341316827, 0.7341387977722521)]
        camera_pos_end = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        t_list = list(np.linspace(0, 1.0, num=40))
        n_t = len(t_list)
        pos_interp_list = camera_pos_interp(camera_pos_start, camera_pos_end, t_list)
        saving_capture_path_list = [
            os.path.join(output_folder, "t_{:.2f}.png").format(t) for t in t_list
        ]
        title1_list = [stage_name] * n_t
        title2_list = ["target"] * n_t
        visualize_animation([source]*n_t, [target]*n_t, title1_list, title2_list,
                            saving_capture_path_list=saving_capture_path_list, camera_pos_list=pos_interp_list, shape1_plot_func=lung_plot("source"),
                            shape2_plot_func=lung_plot("target"), show=False)
        total_captrue_path_list += saving_capture_path_list
        total_captrue_path_list += [total_captrue_path_list[-1]] * 10

        # 1  affine
        output_folder = os.path.join(folder_path, output_folder_name, case_id, "prealign")
        os.makedirs(output_folder, exist_ok=True)
        stage_name = "stage1: affine"
        model_type = "affine_interp"
        flow_opt = ParameterDict()
        flow_opt["model_type"] = model_type
        flow_opt["t_list"] = list(np.linspace(0, 1.0, num=10))
        target_list = [target] * len(flow_opt["t_list"])
        flow_model = FlowModel(flow_opt)
        shape_pair = create_shape_pair(
            source, target, pair_name=case_id, n_control_points=-1
        )
        shape_pair.reg_param = prealign_reg_param
        flowed_list = flow_model(shape_pair)
        camera_pos_start = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        camera_pos_end = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        pos_interp_list = camera_pos_interp(
            camera_pos_start, camera_pos_end, flow_opt["t_list"]
        )
        saving_capture_path_list = [
            os.path.join(output_folder, "t_{:.2f}.png").format(t)
            for t in flow_opt["t_list"]
        ]
        title1_list = [stage_name] * len(flow_opt["t_list"])
        title2_list = ["target"] * len(flow_opt["t_list"])
        visualize_animation(flowed_list,target_list,title1_list,title2_list,saving_capture_path_list=saving_capture_path_list,camera_pos_list=pos_interp_list,shape1_plot_func=lung_plot("source"),
                            shape2_plot_func=lung_plot("target"),show=False)
        total_captrue_path_list += saving_capture_path_list
        total_captrue_path_list += [total_captrue_path_list[-1]] * 10

        # 2  nonp

        output_folder = os.path.join(folder_path, output_folder_name, case_id, "nonp")
        os.makedirs(output_folder, exist_ok=True)
        stage_name = "stage2: LDDMM"
        model_type = "lddmm_shooting"
        flow_opt = ParameterDict()
        flow_opt["model_type"] = model_type
        flow_opt["t_list"] = list(np.linspace(0, 1.0, num=20))
        target_list = [target] * len(flow_opt["t_list"])
        lddmm_opt = flow_opt[("lddmm", {}, "settings for lddmm")]
        lddmm_opt["module"] = "variational"
        lddmm_opt[("variational", {}, "settings for variational formulation")]
        lddmm_opt["variational"][
            "kernel"
        ] = "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.03,0.06,0.09],weight_list=[0.2,0.3,0.5])"
        lddmm_opt[("integrator", {}, "settings for integrator")]
        lddmm_opt["integrator"]["interp_mode"] = True
        lddmm_opt["integrator"]["integration_time"] = flow_opt["t_list"]
        shape_pair = create_shape_pair(
            prealigned, target, pair_name=case_id, n_control_points=-1
        )
        reg_param = init_reg_param(reg_param_path)
        shape_pair.set_reg_param(reg_param)
        shape_pair.set_control_points(control.points, control.weights)
        shape_pair.dense_mode = False
        flow_model = FlowModel(flow_opt)
        flowed_list = flow_model(shape_pair)
        camera_pos_start = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        camera_pos_end = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        pos_interp_list = camera_pos_interp(
            camera_pos_start, camera_pos_end, flow_opt["t_list"]
        )
        saving_capture_path_list = [
            os.path.join(output_folder, "t_{:.2f}.png").format(t)
            for t in flow_opt["t_list"]
        ]
        title1_list = [stage_name] * len(flow_opt["t_list"])
        title2_list = ["target"] * len(flow_opt["t_list"])
        visualize_animation(flowed_list,target_list,title1_list,title2_list,saving_capture_path_list=saving_capture_path_list,camera_pos_list=pos_interp_list,shape1_plot_func=lung_plot("source"),
                            shape2_plot_func=lung_plot("target"), show=False)
        total_captrue_path_list += saving_capture_path_list
        total_captrue_path_list += [total_captrue_path_list[-1]] * 10

        # 3  postprocess
        output_folder = os.path.join(folder_path, output_folder_name, case_id, "post")
        os.makedirs(output_folder, exist_ok=True)
        stage_name = "stage3: postprocessing"
        model_type = "linear_interp"
        flow_opt = ParameterDict()
        flow_opt["model_type"] = model_type
        flow_opt["t_list"] = list(np.linspace(0, 1.0, num=10))
        target_list = [target] * len(flow_opt["t_list"])
        flow_model = FlowModel(flow_opt)
        shape_pair = create_shape_pair(
            nonp, nonp_gf, pair_name=case_id, n_control_points=-1
        )
        shape_pair.flowed = nonp_gf
        flowed_list = flow_model(shape_pair)
        camera_pos_start = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        camera_pos_end = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        pos_interp_list = camera_pos_interp(
            camera_pos_start, camera_pos_end, flow_opt["t_list"]
        )
        saving_capture_path_list = [
            os.path.join(output_folder, "t_{:.2f}.png").format(t)
            for t in flow_opt["t_list"]
        ]
        title1_list = [stage_name] * len(flow_opt["t_list"])
        title2_list = ["target"] * len(flow_opt["t_list"])
        visualize_animation(flowed_list,target_list,title1_list,title2_list,saving_capture_path_list=saving_capture_path_list,camera_pos_list=pos_interp_list,shape1_plot_func=lung_plot("source"),
                            shape2_plot_func=lung_plot("target"), show=False)
        total_captrue_path_list += saving_capture_path_list
        total_captrue_path_list += [total_captrue_path_list[-1]] * 10

        gif_path = os.path.join(folder_path, output_folder_name, case_id, "reg2.gif")
        generate_gif(total_captrue_path_list, gif_path)

        """
         "lddmm": {
                "module": "variational",
                "hamiltonian": {
                    "kernel": "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.05,0.08,0.1],weight_list=[0.2,0.3,0.5])"
                    },
                "variational": {
                    "kernel": "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.03,0.06,0.09],weight_list=[0.2,0.3,0.5])"
                    },
                "integrator":{}
            },
        """
