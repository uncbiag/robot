from robot.utils.obj_factory import obj_factory


def detect_folding(
    warped_grid_points, grid_size, spacing, saving_path=None, file_name=None
):
    from robot.utils.img_visual_utils import compute_jacobi_map
    from robot.utils.utils import point_to_grid

    warped_grid = point_to_grid(warped_grid_points, grid_size)
    compute_jacobi_map(warped_grid[None], spacing, saving_path, [file_name])


def get_omt_mapping(
    gemloss_setting, source, target, fea_to_map, p=2, mode="hard", confid=0.1
):
    """"""
    # here we assume batch_sz = 1
    from robot.metrics.reg_losses import GeomDistance
    from pykeops.torch import LazyTensor

    geom_obj = gemloss_setting["geom_obj"].replace(")", ",potentials=True)")
    blur_arg_filtered = filter(lambda x: "blur" in x, geom_obj.split(","))
    blur = eval(list(blur_arg_filtered)[0].replace("blur", "").replace("=", ""))
    geomloss = obj_factory(geom_obj)
    attr = gemloss_setting[("attr", "points", "attribute used to compute the loss")]
    attr1 = getattr(source, attr)
    attr2 = getattr(target, attr)
    weight1 = source.weights[:, :, 0]  # remove the last dim
    weight2 = target.weights[:, :, 0]  # remove the last dim
    F_i, G_j = geomloss(
        weight1, attr1, weight2, attr2
    )  # todo batch sz of input and output in geomloss is not consistent

    N, M, D = source.points.shape[1], target.points.shape[1], source.points.shape[2]
    a_i, x_i = LazyTensor(source.weights.view(N, 1, 1)), LazyTensor(
        attr1.view(N, 1, -1)
    )
    b_j, y_j = LazyTensor(target.weights.view(1, M, 1)), LazyTensor(
        attr2.view(1, M, -1)
    )
    F_i, G_j = LazyTensor(F_i.view(N, 1, 1)), LazyTensor(G_j.view(1, M, 1))
    C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (N,M,1) cost matrix
    eps = blur ** p  # temperature epsilon
    P_j = ((F_i + G_j - C_ij) / eps).exp() * (a_i)  # (N,M,1) transport plan
    if mode == "soft":
        fea_to_map = LazyTensor(fea_to_map.view(N, 1, -1))  # Nx1xfea_dim
        mapped_fea = (P_j * fea_to_map).sum_reduction(0)  # (N,M,fea_dim)-> (M,fea_dim)
    elif mode == "hard":
        P_j_max, P_j_index = P_j.max_argmax(0)
        mapped_fea = fea_to_map[P_j_index][:, 0]
        below_thre_index = (P_j_max < confid)[:, 0]
        mapped_fea[below_thre_index] = 0
    elif mode == "prob":
        # P_j_max, P_j_index = P_j.max_argmax(0)
        # mapped_fea = P_j_max
        mapped_fea = P_j.sum_reduction(0)
    else:
        raise ValueError("mode {} not defined, support: soft/ hard/ prob".format(mode))
    return mapped_fea
