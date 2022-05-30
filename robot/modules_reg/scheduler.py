import torch.optim.lr_scheduler as lr_scheduler


def scheduler_builder(opt):
    def init_step_lr(opt):
        step_size = opt["step_lr"][
            ("step_size", 50, "update the learning rate every # epoch")
        ]
        gamma = opt["step_lr"][
            ("gamma", 0.5, "the factor for updateing the learning rate")
        ]

        def create_instance(optimizer):
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return scheduler

        return create_instance

    def init_plateau(opt):
        patience = opt["plateau"][
            (
                "patience",
                3,
                "ReduceLROnPlateau param, Number of epochs with no improvement after "
                "which learning rate will be reduced",
            )
        ]
        mode = opt["plateau"][("mode", "min", "ReduceLROnPlateau param, min or max")]
        factor = opt["plateau"][
            ("factor", 0.5, "ReduceLROnPlateau param, decay factor")
        ]
        threshold = opt["plateau"][
            ("threshold", 1e-3, " ReduceLROnPlateau param threshold")
        ]
        threshold_mode = opt["plateau"][
            (
                "threshold_mode",
                "rel",
                "ReduceLROnPlateau param threshold mode, 'rel' or 'abs'",
            )
        ]
        cooldown = opt["plateau"][
            (
                "cooldown",
                0,
                "ReduceLROnPlateau param cooldown, Number of epochs to wait before resuming"
                "normal operation after lr has been reduced",
            )
        ]
        min_lr = opt["plateau"][("min_lr", 1e-8, "ReduceLROnPlateau param min_lr")]

        def create_instance(optimizer):
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                patience=patience,
                factor=factor,
                verbose=True,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
            )
            return scheduler

        return create_instance

    scheduler_type = opt[("type", "step_lr", "scheduler type")]
    assert scheduler_type in ["step_lr", "plateau"]
    init_scheduler_dict = {"step_lr": init_step_lr, "plateau": init_plateau}
    init_scheduler = init_scheduler_dict[scheduler_type](opt)
    return init_scheduler


def set_warmming_up(optimizer, scheduler, opt, warmming_up=True):
    """
    warmming up the training
    for optimization tasks, this function is disabled
    :param optimizer:
    :param scheduler:
    :param warmming_up:
    :return:
    """
    lr = opt[("lr", 0.001, "learning rate")]
    if not warmming_up:
        print(" no warming up the learning rate is {}".format(lr))
    else:
        lr = opt["lr"] / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        scheduler.base_lrs = [lr]
    print(" warming up on the learning rate is {}".format(lr))
