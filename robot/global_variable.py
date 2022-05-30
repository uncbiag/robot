import os
from robot.shape.point_cloud import PointCloud
from robot.shape.surface_mesh import SurfaceMesh, SurfaceMesh_Point
from robot.shape.poly_line import PolyLine
ROBOT_PATH =os.path.abspath("/playpen-raid2/zyshen/proj/robot/robot")
#ROBOT_PATH =os.path.abspath("/home/zyshen/proj/robot/robot")
shape_type = "pointcloud"
SHAPE_POOL = {
    "pointcloud": PointCloud,
    "surfacemesh": SurfaceMesh,
    "surfacemesh_pointmode": SurfaceMesh_Point,
    "polyline": PolyLine,
}
Shape = SHAPE_POOL[shape_type]


from robot.metrics.reg_losses import *

LOSS_POOL = {
    "current": CurrentDistance,
    "varifold": VarifoldDistance,
    "geomloss": GeomDistance,
    "l2": L2Distance,
    "localreg": LocalReg,
    "gmm": GMMLoss,
}


from robot.datasets.general_dataset import GeneralDataset
from robot.datasets.pair_dataset import RegistrationPairDataset

DATASET_POOL = {
    "general_dataset": GeneralDataset,
    "pair_dataset": RegistrationPairDataset,
    "custom_dataset": None,
}


from robot.models_reg.model_lddmm import LDDMMOPT
from robot.models_reg.model_discrete_flow import DiscreteFlowOPT
from robot.models_reg.model_gradient_flow import GradientFlowOPT
from robot.models_reg.model_probreg import ProRegOPT
from robot.models_reg.model_prealign import PrealignOPT
from robot.models_reg.model_deep_feature import DeepFeature
from robot.models_reg.model_deep_flow import DeepDiscreteFlow
from robot.models_reg.model_wasserstein_barycenter import WasserBaryCenterOPT
from robot.models_general.model_deep_pred import DeepPredictor

MODEL_POOL = {
    "lddmm_opt": LDDMMOPT,
    "discrete_flow_opt": DiscreteFlowOPT,
    "prealign_opt": PrealignOPT,
    "gradient_flow_opt": GradientFlowOPT,
    "feature_deep": DeepFeature,
    "flow_deep": DeepDiscreteFlow,
    "discrete_flow_deep": DeepDiscreteFlow,
    "barycenter_opt": WasserBaryCenterOPT,
    "probreg_opt": ProRegOPT,
    "deep_predictor": DeepPredictor,
}


from robot.shape.point_sampler import point_grid_sampler, point_uniform_sampler

SHAPE_SAMPLER_POOL = {
    "point_grid": point_grid_sampler,
    "point_uniform": point_uniform_sampler,
}
# INTERPOLATOR_POOL = {"point_kernel":nadwat_kernel_interpolator, "point_spline": spline_intepolator}
