"""
Coherent Point Drift module, here we use the package https://github.com/neka-nat/probreg

type 	    CPD 	            SVR, GMMReg 	        GMMTree 	    FilterReg 	            BCPD (experimental)
Rigid 	Scale + 6D pose 	    6D pose 	            6D pose 	    6D pose                 -
                                                                        (Point-to-point,
                                                                        Point-to-plane,
                                                                        FPFH-based) 	-

NonRigid Affine, MCT 	        TPS 	                  - 	        Deformable Kinematic    Combined model
                                                                        (experimental) 	        (Rigid + Scale + NonRigid-term)

"""

import numpy as np
import torch
from functools import partial
from robot.utils.obj_factory import partial_obj_factory

try:
    import open3d as o3d
except:
    print("open3d is not detected, related functions are disabled")
try:
    import cupy as cp

    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
except:
    cp = np
    to_cpu = lambda x: x

METHOD_POOL = [
    "cpd",
    "svr",
    "gmmtree",
    "filterreg_rigid",
    "filterreg_nonrigid",
    "BCPD_nonrigid",
]


class ProbReg(object):
    """
    https://github.com/neka-nat/probreg/tree/master
    an interface for probreg package
    """

    def __init__(self, opt):
        super(ProbReg, self).__init__()
        self.opt = opt
        self.method_name = opt[
            (
                "method_name",
                "cpd",
                "supported method name in probreg:" "{}".format(METHOD_POOL),
            )
        ]
        self.init_np_device()
        self.prealign = self.opt[
            ("prealign_mode", False, "run probreg for prealignment task")
        ]
        self.solver = getattr(self, "_init_{}".format(self.method_name))(
            opt[(self.method_name, {}, "settings for {}".format(self.method_name))]
        )

    def set_mode(self, mode):
        if mode == "prealign":
            self.prealign = True
        else:
            self.prealign = False

    def init_np_device(self):
        if self.method_name == "cpd":
            try:
                import cupy as cp

                self.cp = cp
                self.to_cpu = cp.asnumpy
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            except:
                self.cp = np
                self.to_cpu = lambda x: x
        else:
            self.cp = np
            self.to_cpu = lambda x: x

    def _init_cpd(self, opt):
        """CPD Registraion.

        Args:
            source (numpy.ndarray): Source point cloud data.
            target (numpy.ndarray): Target point cloud data.
            tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid')
            w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
            maxitr (int, optional): Maximum number of iterations to EM algorithm.
            tol (float, optional): Tolerance for termination.
            callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                `callback(probreg.Transformation)`
            beta (float, optional): Parameter of RBF kernel.
            lmd (float, optional): Parameter for regularization term.

        Keyword Args:
            update_scale (bool, optional): If this flag is true and tf_type is rigid transformation,
                then the scale is treated. The default is true.
            tf_init_params (dict, optional): Parameters to initialize transformation (for rigid or affine).
        """

        cpd_obj = opt[
            (
                "cpd_obj",
                "probreg_module.registration_cpd(tf_type_name='nonrigid', w=0.0, maxiter=50, tol=0.001,"
                " use_cuda=True,callbacks=[])",
                "cpd object",
            )
        ]
        cpd_solver = partial_obj_factory(cpd_obj)

        return cpd_solver

    def _init_svr(self, opt):
        """Support Vector Registration.

        Args:
            source (numpy.ndarray): Source point cloud data.
            target (numpy.ndarray): Target point cloud data.
            tf_type_name (str, optional): Transformation type('rigid', 'nonrigid')
            maxitr (int, optional): Maximum number of iterations for outer loop.
            tol (float, optional): Tolerance for termination of outer loop.
            opt_maxitr (int, optional): Maximum number of iterations for inner loop.
            opt_tol (float, optional): Tolerance for termination of inner loop.
            callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                `callback(probreg.Transformation)`
        """
        svr_obj = opt[
            (
                "svr_obj",
                "probreg.svr.registration_svr(tf_type_name='rigid', maxitr=1"
                "tol=1.0e-3, opt_maxitr=50,opt_tol=1.0e-3,callbacks=[])",
                "svr object",
            )
        ]
        svr_sovler = partial_obj_factory(svr_obj)
        return svr_sovler

    def _init_gmmtree(self, opt):
        """GMMTree registration

        Args:
            source (numpy.ndarray): Source point cloud data.
            target (numpy.ndarray): Target point cloud data.
            maxitr (int, optional): Maximum number of iterations to EM algorithm.
            tol (float, optional): Tolerance for termination.
            callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                `callback(probreg.Transformation)`

        Keyword Args:
            default: tree_level=2, lambda_c=0.01, lambda_s=0.001,
            tree_level (int, optional): Maximum depth level of GMM tree.
            lambda_c (float, optional): Parameter that determine the pruning of GMM tree.
            lambda_s (float, optional): Parameter that tolerance for building GMM tree.
            tf_init_params (dict, optional): Parameters to initialize transformation.
        """
        gmmtree_obj = opt[
            (
                "gmmtree_obj",
                "probreg.gmmtree.registration_svr(maxitr=20, tol=1.0e-4"
                "callbacks=[])",
                "gmmtree object",
            )
        ]
        gmmtree_sovler = partial_obj_factory(gmmtree_obj)
        return gmmtree_sovler

    def _init_filterreg(self, opt):
        """FilterReg registration

        Args:
            source (numpy.ndarray): Source point cloud data.
            target (numpy.ndarray): Target point cloud data.
            target_normals (numpy.ndarray, optional): Normal vectors of target point cloud.
            sigma2 (float, optional): Variance of GMM. If `sigma2` is `None`, `sigma2` is automatically updated.
            w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
            objective_type (str, optional): The type of objective function selected by 'pt2pt' or 'pt2pl'.
            maxitr (int, optional): Maximum number of iterations to EM algorithm.
            tol (float, optional): Tolerance for termination.
            min_sigma2 (float, optional): Minimum variance of GMM.
            feature_fn (function, optional): Feature function. If you use FPFH feature, set `feature_fn=probreg.feature.FPFH()`.
            callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                `callback(probreg.Transformation)`

        Keyword Args:
            tf_init_params (dict, optional): Parameters to initialize transformation (for rigid).
        """
        filterreg_obj = opt[
            (
                "filterreg_obj",
                "probreg.filterreg.registration_filterreg(sigma2=None, update_sigma2=True, w=0, "
                "objective_type='pt2pt', maxiter=50, tol=0.001, min_sigma2=1.0e-4, "
                "feature_fn=lambda x: x",
                "filterreg object",
            )
        ]
        filterreg_sovler = partial_obj_factory(filterreg_obj)
        feature_fn = probreg.features.FPFH()
        filterreg_sovler.keywords["feature_fn"] = feature_fn
        return filterreg_sovler

    def _init_bcpd(self, opt):
        """BCPD Registraion.

        Args:
            source (numpy.ndarray): Source point cloud data.
            target (numpy.ndarray): Target point cloud data.
            w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
            maxitr (int, optional): Maximum number of iterations to EM algorithm.
            tol (float, optional) : Tolerance for termination.
            callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                `callback(probreg.Transformation)`
        """

        bcpd_obj = opt[
            (
                "bcpd_obj",
                "probreg.bcpd.registration_bcpd(maxitr=20, w=0.0, maxiter=50, tol=0.001, callbacks=[])",
                "bcpd object",
            )
        ]
        bcpd_sovler = partial_obj_factory(bcpd_obj)
        return bcpd_sovler

    def _init_icp(self, opt):
        """
        registration_icp(source, target, max_correspondence_distance, init=(with default value), estimation_method=TransformationEstimationPointToPoint without scaling., criteria=ICPConvergenceCriteria class with relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, and max_iteration=30)

        Function for ICP registration

        Args:
            source (open3d.cuda.pybind.geometry.PointCloud): The source point cloud.
            target (open3d.cuda.pybind.geometry.PointCloud): The target point cloud.
            max_correspondence_distance (float): Maximum correspondence points-pair distance.
            init (numpy.ndarray[float64[4, 4]], optional): Initial transformation estimation Default value:

                array([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]])
            estimation_method (open3d.cuda.pybind.pipelines.registration.TransformationEstimation, optional, default=TransformationEstimationPointToPoint without scaling.): Estimation method. One of (``TransformationEstimationPointToPoint``, ``TransformationEstimationPointToPlane``, ``TransformationEstimationForColoredICP``)
            criteria (open3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria, optional, default=ICPConvergenceCriteria class with relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, and max_iteration=30): Convergence criteria

        Returns:
            open3d.cuda.pybind.pipelines.registration.RegistrationResult
        """
        max_correspondence_distance = opt[
            (
                "max_correspondence_distance",
                0.005,
                "Maximum correspondence points-pair distance",
            )
        ]
        max_iteration = opt[
            ("max_iteration", 1000, "Maximum correspondence points-pair distance")
        ]
        icp_sovler = partial(
            o3d.pipelines.registration.registration_icp,
            max_correspondence_distance=max_correspondence_distance,
            init=self.cp.identity(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )

        def _icp_sovler():
            def solver(source_points, target_points):
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(source_points)
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(target_points)
                return icp_sovler(source, target)

            return solver

        return _icp_sovler()

    def __call__(self, source, target, return_tranform_param=True):
        """
        :param source: Shape with points BxNxD
        :param target_batch: Shape with points BxMxD
        :return: Bx(D+1)xD transform matrix
        """
        source_batch, target_batch = source.points, target.points
        device = source_batch.device
        source_list, target_list = self._get_input(source_batch, target_batch)
        solution_list = []
        for source, target in zip(source_list, target_list):
            solution = self.solver(source, target)
            solution_list.append(solution)
        if return_tranform_param:
            return self._convert_transform_format(solution_list, device)
        else:
            return self._get_transformed_res(solution_list, source_list, device)

    def _get_input(self, source_batch, target_batch):
        device_warpper = lambda x: self.cp.asarray(x)
        source_list = [
            device_warpper(source) for source in source_batch.detach().cpu().numpy()
        ]
        target_list = [
            device_warpper(target) for target in target_batch.detach().cpu().numpy()
        ]
        return source_list, target_list

    def _get_transform_matrix(self, solution):
        try:
            return (self.to_cpu(solution[0].b).T).astype(np.float32)
        except:
            return (
                self.to_cpu(solution[0].rot).T * self.to_cpu(solution[0].scale)
            ).astype(np.float32)

    def _get_translation(self, solution):
        return (self.to_cpu(solution[0].t)[None]).astype(np.float32)

    def _get_deformation(self, solution):
        # todo to implement
        pass

    def _get_prealign_deformation(self, solution_list, device):
        transform_matrix_list = [
            self._get_transform_matrix(solution) for solution in solution_list
        ]
        translation_list = [
            self._get_translation(solution) for solution in solution_list
        ]
        batch_transform_matrix = torch.from_numpy(np.stack(transform_matrix_list))
        batch_translation = torch.from_numpy(np.stack(translation_list))
        return_param = torch.cat((batch_transform_matrix, batch_translation), dim=1)
        return return_param.to(device)

    def _convert_transform_format(self, solution_list, device):
        if self.prealign:
            convert_fn = self._get_prealign_deformation
            return convert_fn(solution_list, device)
        else:
            NotImplemented

    def _get_transformed_res(self, solution_list, source_list, device):
        if self.method_name == "icp":
            transformed_list = []
            for source_np, solution in zip(source_list, solution_list):
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(source_np)
                transformed = source.transform(solution.transformation)
                transformed_list.append(
                    np.asarray(transformed.points).astype(np.float32)
                )

        else:
            transformed_list = [
                self.to_cpu(solution.transformation.transform(source)).astype(
                    np.float32
                )
                for solution, source in zip(solution_list, source_list)
            ]
        batch_transformdc = torch.from_numpy(np.stack(transformed_list))
        return batch_transformdc.to(device)


###############   fix cupy  input ########################
def registration_cpd(
    source,
    target,
    tf_type_name="rigid",
    w=0.0,
    maxiter=50,
    tol=0.001,
    callbacks=[],
    **kargs
):
    """CPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid')
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`

    Keyword Args:
        update_scale (bool, optional): If this flag is true and tf_type is rigid transformation,
            then the scale is treated. The default is true.
        tf_init_params (dict, optional): Parameters to initialize transformation (for rigid or affine).
    """
    from probreg.cpd import RigidCPD, AffineCPD, NonRigidCPD

    if tf_type_name == "rigid":
        cpd = RigidCPD(source, **kargs)
    elif tf_type_name == "affine":
        cpd = AffineCPD(source, **kargs)
    elif tf_type_name == "nonrigid":
        cpd = NonRigidCPD(source, **kargs)
    else:
        raise ValueError("Unknown transformation type %s" % tf_type_name)
    cpd.set_callbacks(callbacks)
    return cpd.registration(target, w, maxiter, tol)
