import os
from copy import deepcopy
from robot.modules_reg.optimizer import optimizer_builder
from robot.modules_reg.scheduler import scheduler_builder
from robot.global_variable import SHAPE_SAMPLER_POOL
from robot.shape.shape_pair_utils import create_shape_pair
from robot.utils.shape_visual_utils import save_shape_pair_into_files
from robot.utils.obj_factory import obj_factory


def build_multi_scale_solver(opt, model):
    """
    :param opt:
    :param model:
    :param optimizer:
    :return:
    """
    sampler_scale_list = opt[
        (
            "point_grid_scales",
            [-1],
            "a list of scales that parameterizes the voxel-grid sampling,"
            " the scale is from rough to fine resolution, -1 refers to the original resolution",
        )
    ]
    sampler_npoints_list = opt[
        (
            "point_uniform_npoints",
            [-1],
            "a list of scales that parameterizes the uniform sampling,"
            " the scale is from rough to fine resolution, -1 refers to the original resolution",
        )
    ]
    scale_iter_list = opt[("iter_per_scale", [1], "number of iterations per scale")]
    scale_iter_list = (
        scale_iter_list
        if not model.call_thirdparty_package
        else [1 for _ in scale_iter_list]
    )
    scale_rel_ftol_list = opt[
        ("rel_ftol_per_scale", [1e-9], "rel_ftol threshold for each scale")
    ]
    init_lr_list = opt[
        ("init_lr_per_scale", [1e-9], "inital learning rate for each scale")
    ]
    shape_sampler_type = opt[
        (
            "shape_sampler_type",
            "point_grid",
            "shape sampler type: 'point_grid' or 'uniform'",
        )
    ]
    assert shape_sampler_type in [
        "point_grid",
        "point_uniform",
    ], "currently only point sampling {} is supported".format(
        ["point_grid", "point_uniform"]
    )
    scale_args_list = (
        sampler_scale_list
        if shape_sampler_type == "point_grid"
        else sampler_npoints_list
    )
    scale_shape_sampler_list = [
        SHAPE_SAMPLER_POOL[shape_sampler_type](scale) for scale in scale_args_list
    ]
    num_scale = len(scale_iter_list)
    stragtegy = opt[
        ("stragtegy", "use_optimizer_defined_from_model','use_optimizer_defined_here")
    ]
    build_single_scale_solver = (
        build_single_scale_general_solver
        if stragtegy == "use_optimizer_defined_here"
        else build_single_scale_model_embedded_solver
    )
    reg_param_initializer = model.init_reg_param
    param_updater = model.update_reg_param_from_low_scale_to_high_scale
    update_shape_pair_after_upsampling = model.flow
    single_scale_solver_list = [
        build_single_scale_solver(
            opt,
            model,
            scale_iter_list[i],
            scale_args_list[i],
            init_lr_list[i],
            scale_rel_ftol_list[i],
        )
        for i in range(num_scale)
    ]
    print("Multi-scale solver initialized!")
    print(
        "The optimization works on the strategy '{}' with setting {}".format(
            shape_sampler_type, scale_args_list
        )
    )

    def solve(shape_pair):
        source, target = shape_pair.source, shape_pair.target
        output_shape_pair = None
        model.clean()
        for i in range(num_scale):
            print(
                "{} th scale optimization begins, with  the strategy '{}' with setting {}".format(
                    i, shape_sampler_type, scale_args_list[i]
                )
            )
            if scale_args_list[i] > 0:
                scale_source = scale_shape_sampler_list[i](source)
                scale_target = scale_shape_sampler_list[i](target)
                toinput_shape_pair = create_shape_pair(
                    scale_source, scale_target, pair_name=shape_pair.get_pair_name()
                )
            else:
                toinput_shape_pair = shape_pair
            reg_param_initializer(toinput_shape_pair)
            # save_shape_pair_into_files(opt["record_path"], "debugging".format(iter), toinput_shape_pair)
            if i != 0:
                toinput_shape_pair = param_updater(
                    output_shape_pair, toinput_shape_pair
                )
                del output_shape_pair
            output_shape_pair = single_scale_solver_list[i](toinput_shape_pair)
        if scale_args_list[-1] != -1:
            output_shape_pair = param_updater(
                output_shape_pair,
                create_shape_pair(
                    source, target, pair_name=output_shape_pair.get_pair_name()
                ),
            )
            output_shape_pair = update_shape_pair_after_upsampling(output_shape_pair)
        return output_shape_pair

    return solve


def build_single_scale_general_solver(
    opt, model, num_iter, scale=-1, lr=1e-4, rel_ftol=1e-4, patient=5
):
    """
    general solver where the param needs to optimize iteratively
    this is typically required by native displacement method, lddmm method,

    :param opt:
    :param model:
    :param num_iter:
    :param scale:
    :param rel_ftol:
    :param patient:
    :return:
    """
    save_res = opt[("save_res", True, "save 2d/3d result")]
    save_3d_shape_every_n_iter = opt[
        ("save_3d_shape_every_n_iter", 20, "save output every n iteration")
    ]
    save_2d_capture_every_n_iter = opt[
        (
            "save_2d_capture_every_n_iter",
            -1,
            "save 2d screen capture of the plot every n iteration",
        )
    ]
    capture_plotter_obj = opt[
        (
            "capture_plot_obj",
            "visualizer.capture_plotter()",
            "factory object for 2d capture plot",
        )
    ]
    capture_plotter = obj_factory(capture_plotter_obj)
    record_path = opt[("record_path", "", "record path")]
    record_path = os.path.join(record_path, "scale_{}".format(scale))
    model.set_record_path(record_path)
    os.makedirs(record_path, exist_ok=True)
    shape_folder_3d = os.path.join(record_path, "3d")
    os.makedirs(shape_folder_3d, exist_ok=True)
    shape_folder_2d = os.path.join(record_path, "2d")
    os.makedirs(shape_folder_2d, exist_ok=True)
    opt_optim = opt[("optim", {}, "setting for the optimizer")]
    opt_optim = deepcopy(opt_optim)
    """settings for the optimizer"""
    opt_optim["lr"] = lr
    opt_scheduler = opt[("scheduler", {}, "setting for the scheduler")]
    """settings for the scheduler"""

    def solve(shape_pair):
        model.init_reg_param(shape_pair)
        ######################################3
        shape_pair.reg_param.register_hook(grad_hook)
        ############################################3
        optimizer = optimizer_builder(opt_optim)([shape_pair.reg_param])
        lr_scheduler = scheduler_builder(opt_scheduler)(optimizer)
        """initialize the optimizer and scheduler"""
        last_energy = 0.0
        patient_count = 0
        previous_converged_iter = 0.0

        def closure():
            optimizer.zero_grad()
            cur_energy = model(shape_pair)
            cur_energy.backward()
            return cur_energy

        for iter in range(num_iter):
            cur_energy = optimizer.step(closure)
            lr_scheduler.step(iter)
            cur_energy = cur_energy.item()
            rel_f = abs(last_energy - cur_energy) / (abs(cur_energy))
            last_energy = cur_energy
            if (
                save_res
                and shape_pair.dimension == 3
                and save_3d_shape_every_n_iter > 0
                and iter % save_3d_shape_every_n_iter == 0
            ):
                save_shape_pair_into_files(
                    shape_folder_3d,
                    "iter_{}".format(iter),
                    shape_pair.get_pair_name(),
                    shape_pair,
                )
            if (
                save_res
                and save_2d_capture_every_n_iter > 0
                and iter % save_2d_capture_every_n_iter == 0
            ):
                capture_plotter(
                    shape_folder_2d,
                    "iter_{}".format(iter),
                    shape_pair.get_pair_name(),
                    shape_pair,
                )
            if rel_f < rel_ftol:
                print("the converge rate: {} is too small".format(rel_f))
                patient_count = (
                    patient_count + 1 if (iter - previous_converged_iter) == 1 else 0
                )
                previous_converged_iter = iter
                if patient_count > patient:
                    print("Reached relative function tolerance of = " + str(rel_ftol))
                    break
        if save_res:
            save_shape_pair_into_files(
                record_path, "iter_last", shape_pair.get_pair_name(), shape_pair
            )
        model.reset()
        return shape_pair

    return solve


def build_single_scale_model_embedded_solver(
    opt, model, num_iter=1, scale=-1, lr=1e-4, rel_ftol=1e-4, patient=2
):
    """
    the optimizer and scheduler are not defined
    the model take responsibility to solve the optimal solution
    this is typically designed for calling the thirdparty package, least square regression, gradient flow,
     or approaches needs inner iteration ,e.g, coherent point drift
    :param opt:
    :param model:
    :param num_iter: typically the iteration here should be set to 1, but here we leave it for potential usage
    :param scale:
    :param rel_ftol:
    :param patient:
    :return:
    """
    save_res = opt[("save_res", True, "save 2d/3d result")]
    save_3d_shape_every_n_iter = opt[
        ("save_3d_shape_every_n_iter", 1, "save output every n iteration")
    ]
    save_2d_capture_every_n_iter = opt[
        (
            "save_2d_capture_every_n_iter",
            1,
            "save 2d screen capture of the plot every n iteration",
        )
    ]
    capture_plotter_obj = opt[
        (
            "capture_plot_obj",
            "visualizer.capture_plotter()",
            "factory object for 2d capture plot",
        )
    ]
    capture_plotter = obj_factory(capture_plotter_obj)
    record_path = opt[("record_path", "", "record path")]
    record_path = os.path.join(record_path, "scale_{}".format(scale))
    model.set_record_path(record_path)
    os.makedirs(record_path, exist_ok=True)
    shape_folder_3d = os.path.join(record_path, "3D")
    os.makedirs(shape_folder_3d, exist_ok=True)
    shape_folder_2d = os.path.join(record_path, "2d")
    os.makedirs(shape_folder_2d, exist_ok=True)

    def solve(shape_pair):
        model.init_reg_param(shape_pair)
        last_energy = 0.0
        patient_count = 0
        previous_converged_iter = 0.0
        for iter in range(num_iter):
            cur_energy = model(shape_pair)
            cur_energy = cur_energy.item()
            rel_f = abs(last_energy - cur_energy) / (abs(cur_energy))
            last_energy = cur_energy
            if (
                save_res
                and save_3d_shape_every_n_iter > 0
                and iter % save_3d_shape_every_n_iter == 0
            ):
                save_shape_pair_into_files(
                    shape_folder_3d,
                    "iter_{}".format(iter),
                    shape_pair.get_pair_name(),
                    shape_pair,
                )
            if (
                save_res
                and save_2d_capture_every_n_iter > 0
                and iter % save_2d_capture_every_n_iter == 0
            ):
                capture_plotter(
                    shape_folder_2d,
                    "iter_{}".format(iter),
                    shape_pair.get_pair_name(),
                    shape_pair,
                )
            if rel_f < rel_ftol:
                print("the converge rate: {} is too small".format(rel_f))
                patient_count = (
                    patient_count + 1 if (iter - previous_converged_iter) == 1 else 0
                )
                previous_converged_iter = iter
                if patient_count > patient:
                    print("Reached relative function tolerance of = " + str(rel_ftol))
                    break
        if save_res:
            save_shape_pair_into_files(
                record_path, "iter_last", shape_pair.get_pair_name(), shape_pair
            )
        model.reset()
        return shape_pair

    return solve


###################


def grad_hook(grad):
    # import pydevd
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    print("debugging info, the grad_norm is {} ".format(grad.norm()))
    return grad


############################3
