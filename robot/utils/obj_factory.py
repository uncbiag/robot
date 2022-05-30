import os
import importlib
from functools import partial

KNOWN_MODULES = {
    # Torch
    "lung_dataloader_utils": "robot.experiments.datasets.lung.lung_dataloader_utils",
    "body_dataset_utils": "robot.experiments.datasets.body.body_dataset_utils",
    "toy_dataset_utils": "robot.experiments.datasets.toy.toy_dataset_utils",
    "face_dataset_utils": "robot.experiments.datasets.unpub_face.face_dataset_utils",
    "dhface_dataset_utils": "robot.experiments.datasets.dhface.dataset_utils",
    "flying3d_hasocc_utils": "robot.experiments.datasets.flying3d_and_kitti.flying3d_hasocc.dataset_utils",
    "flyingkitti_nonocc_utils": "robot.experiments.datasets.flying3d_and_kitti.flyingkitti_nonocc.dataset_utils",
    "modelnet40_dataset_utils": "robot.experiments.datasets.modelnet40.modelnet40_dataset_utils",
    "lung_shape_pair": "robot.experiments.datasets.lung.lung_shape_pair",
    "lung_feature_extractor": "robot.experiments.datasets.lung.lung_feature_extractor",
    "dhface_feature_extractor": "robot.experiments.datasets.dhface.feature_extractor",
    "lung_data_analysis": "robot.experiments.datasets.lung.lung_data_analysis",
    "lung_data_aug": "robot.experiments.datasets.lung.lung_data_aug",
    "lung_visualizer": "robot.experiments.datasets.lung.visualizer",
    "dhface_data_aug": "robot.experiments.datasets.dhface.data_aug",
    "lung_data_eval": "robot.experiments.datasets.lung.evaluation_utils",
    "dhface_eval": "robot.experiments.datasets.dhface.evaluation_utils",
    "flyingkitti_nonocc_aug": "robot.experiments.datasets.flying3d_and_kitti.flyingkitti_nonocc.data_aug",
    "flyingkitti_nonocc_eval": "robot.experiments.datasets.flying3d_and_kitti.flyingkitti_nonocc.evaluation_utils",
    "modelnet40_eval": "robot.experiments.datasets.modelnet40.evaluation_utils",
    "modelnet_feature_extractor": "robot.experiments.datasets.modelnet40.modelnet_feature_extractor",
    "2d_toy_local_feature_extractor": "robot.experiments.datasets.toy.toy_feature_extractor_2d",
    "local_feature_extractor": "robot.utils.local_feature_extractor",
    "hybird_data": "robot.datasets.mixed_pair_generator",
    "aug_shape": "robot.datasets.shape_augmentor",
    "visualizer": "robot.utils.visualizer",
    "probreg_module": "robot.modules_reg.module_probreg",
    "deep_feature_module": "shapmgan.modules_reg.deep_feature_module",
    "shape_pair_utils": "robot.shape.shape_pair_utils",
    "torch_kernels": "robot.kernels.torch_kernels",
    "keops_kernels": "robot.kernels.keops_kernels",
    "point_interpolator": "robot.shape.point_interpolator",
    "geomloss": "geomloss",
    "nn": "torch.nn",
    "optim": "torch.optim",
    "lr_scheduler": "torch.optim.lr_scheduler",
    "probreg": "probreg",
    "features": "probreg.features",
    "utils": "robot.utils.utils",
    "knn_utils": "robot.utils.knn_utils"
    #'probreg.filterreg':'probreg.filterreg'
}


def extract_args(*args, **kwargs):
    return args, kwargs


def obj_factory(obj_exp, *args, **kwargs):
    """Creates objects from strings or partial objects with additional provided arguments.

    In case a sequence is provided, all objects in the sequence will be created recursively.
    Objects that are not strings or partials be returned as they are.

    Args:
        obj_exp (str or partial): The object string expresion or partial to be converted into an object. Can also be
            a sequence of object expressions
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        object or object list: Created object or list of recursively created objects
    """
    if isinstance(obj_exp, (list, tuple)):
        return [obj_factory(o, *args, **kwargs) for o in obj_exp]
    if isinstance(obj_exp, partial):
        return obj_exp(*args, **kwargs)
    if not isinstance(obj_exp, str):
        return obj_exp

    # Handle arguments
    if "(" in obj_exp and ")" in obj_exp:
        args_exp = obj_exp[obj_exp.find("(") :]
        obj_args, obj_kwargs = eval("extract_args" + args_exp)

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[: obj_exp.find("(")]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules_reg and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(
        KNOWN_MODULES[module_name] if module_name in KNOWN_MODULES else module_name
    )
    module_class = getattr(module, class_name)
    class_instance = module_class(*args, **kwargs)

    return class_instance


def partial_obj_factory(obj_exp, *args, **kwargs):
    """Creates objects from strings or partial objects with additional provided arguments.

    In case a sequence is provided, all objects in the sequence will be created recursively.
    Objects that are not strings or partials be returned as they are.

    Args:
        obj_exp (str or partial): The object string expresion or partial to be converted into an object. Can also be
            a sequence of object expressions
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        object or object list: Created object or list of recursively created objects
    """
    if isinstance(obj_exp, (list, tuple)):
        return [partial_obj_factory(o, *args, **kwargs) for o in obj_exp]
    if isinstance(obj_exp, partial):
        return partial(
            obj_exp.func, *(obj_exp.args + args), **{**obj_exp.keywords, **kwargs}
        )
    if not isinstance(obj_exp, str):
        return partial(obj_exp)

    # Handle arguments
    if "(" in obj_exp and ")" in obj_exp:
        args_exp = obj_exp[obj_exp.find("(") :]
        obj_args, obj_kwargs = eval("extract_args" + args_exp)

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[: obj_exp.find("(")]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules_reg and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(
        KNOWN_MODULES[module_name] if module_name in KNOWN_MODULES else module_name
    )
    module_class = getattr(module, class_name)

    return partial(module_class, *args, **kwargs)


def main(obj_exp):
    # obj = obj_factory(obj_exp)
    # print(obj)

    import inspect

    partial_obj = partial_obj_factory(obj_exp)
    print(f"is obj_exp a class = {inspect.isclass(partial_obj.func)}")
    print(partial_obj)


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser("utils test")
    parser.add_argument("obj_exp", help="object string")
    args = parser.parse_args()

    main(args.obj_exp)
