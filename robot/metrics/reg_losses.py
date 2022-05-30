"""
the code is largely borrowed from deformetrica
here turns it into a batch version
"""
import torch
from pykeops.torch import LazyTensor
from robot.kernels.keops_kernels import LazyKeopsKernel
from robot.kernels.torch_kernels import TorchKernel
from robot.modules_reg.networks.pointconv_util import index_points_group
from robot.utils.obj_factory import obj_factory
from robot.global_variable import Shape
from robot.utils.utils import sigmoid_decay


class CurrentDistance(object):
    def __init__(self, opt):
        kernel_backend = opt[
            ("kernel_backend", "torch", "kernel backend can either be 'torch'/'keops'")
        ]
        sigma = opt[("sigma", 0.1, "the sigma in gaussian kernel")]
        self.kernel = (
            LazyKeopsKernel("gauss", sigma=sigma)
            if kernel_backend == "keops"
            else TorchKernel("gauss", sigma=sigma)
        )

    def __call__(self, flowed, target):
        assert flowed.type == "PolyLine"
        batch = flowed.batch
        c1, n1 = flowed.get_centers_and_currents()
        c2, n2 = target.get_centers_and_currents()

        def current_scalar_product(points_1, points_2, normals_1, normals_2):
            return (
                (normals_1.view(batch, -1))
                * (self.kernel(points_1, points_2, normals_2).view(batch, -1))
            ).sum(-1)

        distance = (
            current_scalar_product(c1, c1, n1, n1)
            + current_scalar_product(c2, c2, n2, n2)
            - 2 * current_scalar_product(c1, c2, n1, n2)
        )
        return distance


class VarifoldDistance(object):
    def __init__(self, opt):
        kernel_backend = opt[
            ("kernel_backend", "torch", "kernel backend can either be 'torch'/'keops'")
        ]
        sigma = opt[("sigma", 0.1, "the sigma in gaussian lin kernel")]
        self.kernel = (
            LazyKeopsKernel("gauss_lin", sigma=sigma)
            if kernel_backend == "keops"
            else TorchKernel("gauss_lin", sigma=sigma)
        )

    def __call__(self, flowed, target, epoch=None):
        assert flowed.type == "SurfaceMesh"
        batch = flowed.batch
        c1, n1 = flowed.get_centers_and_normals()
        c2, n2 = target.get_centers_and_normals()
        areaa = torch.norm(n1, 2, 2)[None]  # BxNx1
        areab = torch.norm(n2, 2, 2)[None]  # BxKx1
        nalpha = n1 / areaa
        nbeta = n2 / areab

        def varifold_scalar_product(x, y, areaa, areab, nalpha, nbeta):
            return (
                (
                    areaa.view(batch, -1)
                    * (self.kernel(x, y, nalpha, nbeta, areab).view(batch, -1))
                )
            ).sum(-1)

        return (
            varifold_scalar_product(c1, c1, areaa, areaa, nalpha, nalpha)
            + varifold_scalar_product(c2, c2, areab, areab, nbeta, nbeta)
            - 2 * varifold_scalar_product(c1, c2, areaa, areab, nalpha, nbeta)
        )


class L2Distance(object):
    def __init__(self, opt):
        self.attr = opt[
            (
                "attr",
                "pointfea",
                "compute distance on the specific class attribute: 'ponts','landmarks','pointfea",
            )
        ]

    def __call__(self, flowed, target, epoch=None):
        batch = flowed.nbatch
        attr1 = getattr(flowed, self.attr)
        attr2 = getattr(target, self.attr)
        return ((attr1.view(batch, -1) - attr2.view(batch, -1)) ** 2).mean(-1)  # B


class LocalReg(object):
    """
    compare local invariant shape feature (e.g. local eigenvale) before and after registration
    """

    def __init__(self, opt):
        local_feature_extractor_obj = opt[
            (
                "local_feature_extractor",
                "local_feature_extractor.pair_feature_extractor(fea_type_list=['eigenvalue_prod'],weight_list=[0.1], radius=0.05,include_pos=True)",
                "feature extractor",
            )
        ]

        self.pair_feature_extractor = obj_factory(local_feature_extractor_obj)

        local_kernel_obj = opt[
            (
                "local_kernel_obj",
                "keops_kernels.LazyKeopsKernel('gauss',sigma=0.1)",
                "kernel object",
            )
        ]
        self.local_kernel = obj_factory(local_kernel_obj)

    def __call__(self, source, flowed, epoch=None):
        source_tmp = Shape().set_data_with_refer_to(source.points)
        flowed_tmp = Shape().set_data_with_refer_to(flowed.points)
        source_tmp, flowed_tmp = self.pair_feature_extractor(source_tmp, flowed_tmp)
        loss = (source_tmp.pointfea - flowed_tmp.pointfea) ** 2
        loss = (loss.sum(2) * source.weight[..., 0]).sum(1)
        return loss


class GMMLoss:
    def __init__(self, opt):
        self.sigma = opt[("sigma", 0.1, "sigma of the gauss kernel")]
        self.w_noise = opt[("w_noise", 0.0, "ratio of the noise")]
        self.use_anneal = opt["use_anneal", False, "use anneal strategy for sigma"]
        anneal_strategy_obj = opt[
            "anneal_strategy_obj", "", "anneal strategy for sigma"
        ]
        self.anneal_strategy = (
            obj_factory(anneal_strategy_obj)
            if anneal_strategy_obj
            else self.anneal_strategy
        )
        self.attr = opt[
            (
                "attr",
                "points",
                "compute distance on the specific class attribute: 'ponts','landmarks','pointfea",
            )
        ]
        self.mode = opt[
            (
                "mode",
                "neglog_likelihood",
                "neglog_likelihood/sym_neglog_likelihood/log_sum_likelihood",
            )
        ]

    def update_sigma(self, epoch):

        if self.use_anneal and self.anneal_strategy is not None:
            return self.anneal_strategy(self.sigma, epoch)
        else:
            return self.sigma

    def anneal_strategy(self, sigma, iter):
        sigma = float(max(sigmoid_decay(iter, static=10, k=6) * 1, sigma))
        print(sigma)
        return sigma

    #
    # def log_sum_likelihood(self,  attr1, attr2,weight1, weight2):
    #     """Expectation step for CPD
    #     """
    #     N, M = attr1.shape[1], attr2.shape[1]
    #     D = float(attr1.shape[-1])
    #     sigma = self.sigma
    #     pi = 3.14159265359
    #     factor = (2*pi*(sigma**2))**(-D/2)
    #     attr1 = LazyTensor(attr1[:,:,None]/(sigma*(2**(1/D))))
    #     attr2 = LazyTensor(attr2[:,None]/(sigma*(2**(1/D))))
    #     dist = attr1.sqdist(attr2)
    #     logw_j = LazyTensor(weight2.log()[:,None])  # (B,1,M,1)
    #     scores_i = ((logw_j - dist).logsumexp(dim=2) +1e-7)  # (B,N, 1)
    #     scores = scores_i
    #     if self.w_noise>0:
    #         # todo , a normalization term is needed
    #         scores = scores_i*(1-self.w_noise) + 1/N*self.w_noise
    #     loss = -scores.sum(1)/N
    #     return loss

    def neglog_likelihood(self, attr1, attr2, weight1, weight2):
        D = float(attr1.shape[-1])
        sigma = self.sigma
        attr1 = LazyTensor(attr1[:, :, None] / (sigma * (2 ** (1 / 2))))
        attr2 = LazyTensor(attr2[:, None] / (sigma * (2 ** (1 / 2))))
        dist = attr1.sqdist(attr2)
        logw_j = LazyTensor(weight2.log()[:, None])  # (B,1,M,1)
        scores_i = ((logw_j - dist).logsumexp(dim=2)) * (sigma ** D)  # (B,N, 1)
        loss = -torch.sum(weight1 * scores_i, 1)
        return loss

    def sym_neglog_likelihood(self, attr1, attr2, weight1, weight2):
        return self.neglog_likelihood(
            attr1, attr2, weight1, weight2
        ) + self.neglog_likelihood(attr2, attr1, weight2, weight1)

    def __call__(self, flowed, target, epoch=None):
        self.sigma = self.update_sigma(epoch)
        attr1 = getattr(flowed, self.attr)
        attr2 = getattr(target, self.attr)
        weight1 = flowed.weights
        weight2 = target.weights
        fn = None
        if self.mode == "neglog_likelihood":
            fn = self.neglog_likelihood
        elif self.mode == "sym_neglog_likelihood":
            fn = self.sym_neglog_likelihood
        return fn(attr1, attr2, weight1, weight2)


class GeomDistance(object):
    def __init__(self, opt):
        self.attr = opt[
            (
                "attr",
                "points",
                "compute distance on the specific class attribute: 'ponts','landmarks','pointfea",
            )
        ]
        geom_obj = opt[
            (
                "geom_obj",
                "geomloss.SamplesLoss(loss='sinkhorn',blur=0.01, scaling=0.8, debias=False)",
                "blur argument in ot",
            )
        ]
        self.gemoloss = obj_factory(geom_obj)

    def __call__(self, flowed, target, epoch=None):
        attr1 = getattr(flowed, self.attr)
        attr2 = getattr(target, self.attr)
        weight1 = flowed.weights[:, :, 0]  # remove the last dim
        weight2 = target.weights[:, :, 0]  # remove the last dim
        grad_enable_record = torch.is_grad_enabled()
        loss = self.gemoloss(weight1, attr1, weight2, attr2)
        torch.set_grad_enabled(grad_enable_record)
        return loss


class CurvatureReg(object):
    def __init__(self, opt):
        opt.print_settings_off()
        weights = opt[
            ("weight", [3, 0.01], "weights for curvature diff and curvature smoothness")
        ]
        self.interp_sigma = opt[
            (
                "interp_sigma",
                0.01,
                "the kernel size used to interpolate between the flowed shape from the target shape",
            )
        ]
        knn_obj = opt[
            (
                "knn_obj",
                "knn_utils.KNN()",
                "the kernel size used to interpolate between the flowed shape from the target shape",
            )
        ]
        self.knn = obj_factory(knn_obj)
        self.w_diff = weights[0]
        self.w_smooth = weights[1]

    def curvature(self, pc):
        _, index = self.knn(pc, pc, 10)
        grouped_pc = index_points_group(pc, index)
        pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim=2) / 9.0
        return pc_curvature  # B N 3

    def curvatureWarp(self, pc, warped_pc):
        _, index = self.knn(pc, pc, 10)
        grouped_pc = index_points_group(warped_pc, index)
        pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim=2) / 9.0
        return pc_curvature  # B N 3

    def interpolateCurvature(self, pc1, pc2, pc2_curvature):
        sigma = self.interp_sigma
        K_dist, index = self.knn(pc1 / sigma, pc2 / sigma, 5)
        grouped_pc2_curvature = index_points_group(pc2_curvature, index)
        K_w = torch.nn.functional.softmax(-K_dist, dim=2)
        inter_pc2_curvature = torch.sum(K_w[..., None] * grouped_pc2_curvature, dim=2)
        return inter_pc2_curvature

    def computeSmooth(self, pc1, pred_flow):

        _, index = self.knn(pc1, pc1, 9)  # B N N
        grouped_flow = index_points_group(pred_flow, index)
        diff_flow = (
            torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim=3).sum(dim=2) / 8.0
        )
        return diff_flow

    def __call__(self, source, flowed, target, epoch=None):
        source_points, flowed_points, target_points = (
            source.points,
            flowed.points,
            target.points,
        )
        cur_pc2_curvature = self.curvature(target_points)  # curvature of target
        moved_pc1_curvature = self.curvatureWarp(
            source_points, flowed_points
        )  # define the flowed curvature  where topology is  define by the source
        smoothnessLoss = self.computeSmooth(
            source_points, flowed_points - source_points
        ).mean(
            dim=1
        )  # curvature of flow
        inter_pc2_curvature = self.interpolateCurvature(
            flowed_points, target_points, cur_pc2_curvature
        )  # define the flowed curvature via interpolating the neighboring target curvature
        curvatureLoss = torch.sum(
            (inter_pc2_curvature - moved_pc1_curvature) ** 2, dim=2
        ).mean(
            dim=1
        )  # difference between two definition of the flowed curvature
        return self.w_smooth * smoothnessLoss + self.w_diff * curvatureLoss


class Loss:
    """
    DynamicComposed loss class that compose of different loss and has dynamic loss weights during the training
    """

    def __init__(self, opt):
        from robot.global_variable import LOSS_POOL

        loss_name_list = opt[
            (
                "loss_list",
                ["l2"],
                "a list of loss name to compute: l2, geomloss, current, varifold",
            )
        ]
        loss_weight_strategy = opt[
            (
                "loss_weight_strategy",
                "",
                "for each loss in name_list, design weighting strategy: '{'loss_name':'strategy_param'}",
            )
        ]
        self.loss_weight_strategy = (
            obj_factory(loss_weight_strategy) if loss_weight_strategy else None
        )
        self.loss_activate_epoch_list = opt[
            (
                "loss_activate_epoch_list",
                [0],
                "for each loss in name_list, activate at # epoch'",
            )
        ]
        self.loss_fn_list = [
            LOSS_POOL[name](opt[(name, {}, "settings")]) for name in loss_name_list
        ]

    def update_weight(self, epoch):
        if not self.loss_weight_strategy:
            return [1.0] * len(self.loss_fn_list)
        else:
            return self.loss_weight_strategy(epoch)

    def __call__(self, flowed, target, epoch=-1):
        weights_list = self.update_weight(epoch)
        loss_list = [
            weight * loss_fn(flowed, target) if weight else 0.0
            for weight, loss_fn in zip(weights_list, self.loss_fn_list)
        ]
        total_loss = sum(loss_list)
        return total_loss


# class KernelNorm(object):
#     def __init__(self, opt):
#         kernel_backend = opt[("kernel_backend",'torch',"kernel backend can either be 'torch'/'keops'")]
#         kernel_norm_opt = opt[('kernel_norm',{},"settings for the kernel norm loss")]
#         sigma = kernel_norm_opt[('sigma',0.1,"the sigma in gaussian kernel")]
#         self.kernel = LazyKeopsKernel('gauss',sigma) if kernel_backend == 'keops' else TorchKernel('gauss',sigma)
#     def __call__(self,flowed, target):
#         batch = flowed.batch
#         c1, n1 = flowed.get_centers_and_normals()
#         c2, n2 = target.get_centers_and_normals()
#         def kernel_norm_scalar_product(points_1, points_2, normals_1, normals_2):
#             return ((normals_1.view(batch,-1))
#                     * (self.kernel(points_1, points_2, normals_2))
#                     ).sum(-1)
#
#         return kernel_norm_scalar_product(c1, c1, n1, n1)\
#                + kernel_norm_scalar_product(c2, c2, n2, n2)\
#                - 2 * kernel_norm_scalar_product(c1, c2, n1, n2)
