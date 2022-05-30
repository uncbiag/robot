import torch.optim as optim


def optimizer_builder(opt):
    lr = opt[("lr", 0.001, "learning rate")]
    optimizer_type = opt[("type", "adam", "optimizer type")]
    assert optimizer_type in ["lbgfs", "sgd", "adam"]

    def init_lbgfs(opt):
        rel_ftol = opt["lbgfs"][
            ("rel_ftol", 1e-4, "relative termination tolerance for optimizer")
        ]
        max_iter = opt["lbgfs"][("max_iter", 1, "maximum number of iterations")]
        max_eval = opt["lbgfs"][("max_eval", 5, "maximum number of evaluation")]
        history_size = opt["lbgfs"][
            ("history_size", 5, "Size of the optimizer history")
        ]
        # line_search_fn = opt['lbgfs'][('line_search_fn', 'backtracking', 'Type of line search function')]
        def create_instance(params):
            opt_instance = optim.LBFGS(
                params,
                lr=lr,
                max_iter=max_iter,
                max_eval=max_eval,
                tolerance_grad=rel_ftol * 10,
                tolerance_change=rel_ftol,
                history_size=history_size,
                line_search_fn=None,
            )
            return opt_instance

        return create_instance

    def init_sgd(opt):
        momentum = opt["sgd"][("momentum", 0.9, "sgd momentum")]
        dampening = opt["sgd"][("dampening", 0.0, "sgd dampening")]
        weight_decay = opt["sgd"][("weight_decay", 0.0, "sgd weight decay")]
        nesterov = opt["sgd"][("nesterov", True, "use Nesterove scheme")]

        def create_instance(params):
            opt_instance = optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
            return opt_instance

        return create_instance

    def init_adam(opt):
        adam_betas = opt["adam"][("betas", [0.9, 0.999], "adam betas")]

        def create_instance(params):
            opt_instance = optim.Adam(params, lr=lr, betas=adam_betas)
            return opt_instance

        return create_instance

    init_opt_dict = {"lbgfs": init_lbgfs, "sgd": init_sgd, "adam": init_adam}
    init_opt = init_opt_dict[optimizer_type](opt)
    return init_opt
