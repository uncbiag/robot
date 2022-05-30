from time import time
from robot.utils.net_utils import resume_train, save_checkpoint, update_res
from robot.utils.utils import set_seed


def train_model(opt, model, dataloaders, writer, device):
    since = time()
    print_step = opt[("print_step", [10, 4, 4], "num of steps to print")]
    num_epochs = opt[("epoch", 100, "num of training epoch")]
    continue_train = opt[("continue_train", False, "continue to train")]
    model_path = opt["path"]["model_load_path"]
    reset_train_epoch = opt[
        ("reset_train_epoch", False, "allow the training epoch to be reset of not")
    ]
    load_model_but_train_from_epoch = opt[
        (
            "load_model_but_train_from_epoch",
            0,
            "if reset_train_epoch is true, the epoch will be set as the given number",
        )
    ]
    check_point_path = opt["path"]["check_point_path"]
    max_batch_num_per_epoch_list = opt[
        (
            "max_batch_num_per_epoch",
            (-1, -1, -1),
            "max batch number per epoch for train|val|debug",
        )
    ]
    best_score = -1
    start_epoch = 0
    best_epoch = -1
    phases = ["train", "val", "debug"]
    global_step = {x: 0 for x in phases}
    period_loss = {x: 0.0 for x in phases}
    period_detailed_scores = {x: {} for x in phases}
    max_batch_num_per_epoch = {
        x: max_batch_num_per_epoch_list[i] for i, x in enumerate(phases)
    }
    period = {x: print_step[i] for i, x in enumerate(phases)}
    check_best_model_period = opt[
        ("check_best_model_period", 5, "save best performed model every # epoch")
    ]
    save_fig_on = opt[
        ("save_fig_on", False, "save the visualizatio results during the evaluation")
    ]
    tensorboard_print_period = {
        phase: min(max_batch_num_per_epoch[phase], period[phase]) for phase in phases
    }
    val_period = opt[("val_period", 10, "do validation every num epoch")]
    warmming_up_epoch = opt[
        ("warmming_up_epoch", 2, "warming up the model in the first # epoch")
    ]
    continue_train_lr = opt[
        ("continue_train_lr", -1, "learning rate for continuing to train")
    ]
    opt["optim"]["lr"] = opt["optim"]["lr"] if not continue_train else continue_train_lr

    if continue_train:
        start_epoch, best_prec1, global_step = resume_train(
            model_path, model.get_model(), model.optimizer
        )
        if continue_train_lr > 0:
            model.reset_lr_optimizer(continue_train_lr)
            print(
                "the learning rate has been changed into {} when resuming the training".format(
                    continue_train_lr
                )
            )
            model.rebuild_lr_scheduler(base_epoch=start_epoch)
            model.iter_count = global_step["train"]
        if reset_train_epoch:
            start_epoch = load_model_but_train_from_epoch
            global_step = {
                x: load_model_but_train_from_epoch * max_batch_num_per_epoch[x]
                for x in phases
            }
            print(
                "the model has been initialized from extern, but will train from the epoch {}".format(
                    start_epoch
                )
            )
            model.rebuild_lr_scheduler(base_epoch=start_epoch)
            model.iter_count = 0

    for epoch in range(start_epoch, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        model.set_cur_epoch(epoch)
        if epoch == warmming_up_epoch and not reset_train_epoch:
            model.update_learning_rate()

        for phase in phases:
            # if is not training phase, and not the #*val_period , then break
            if phase != "train" and epoch % val_period != 0:
                break
            # if # = 0 or None then skip the val or debug phase
            if not max_batch_num_per_epoch[phase]:
                continue
            if phase == "train":
                set_seed(seed=None)
                model.update_scheduler(epoch)
                model.set_train()
            elif phase == "val":
                set_seed(0)
                model.set_val()
            else:
                set_seed(0)
                model.set_debug()

            running_val_score = {}
            running_debug_score = {}

            for data in dataloaders[phase]:

                global_step[phase] += 1
                end_of_epoch = (
                    global_step[phase]
                    % min(max_batch_num_per_epoch[phase], len(dataloaders[phase]))
                    == 0
                )
                is_train = True if phase == "train" else False
                input = model.set_input(data, device=device, phase=phase)
                loss = 0.0
                detailed_scores = 0.0

                if phase == "train":
                    model.optimize_parameters(input)
                    loss = model.get_current_errors()
                    input = None

                elif phase == "val":
                    val_res = model.get_evaluation(input)

                    score, detailed_scores = model.analyze_res(val_res, cache_res=True)
                    print(
                        "val score of batch {} is {}:".format(
                            model.get_batch_names(), score
                        )
                    )
                    print("val detailed scores are {}:".format(detailed_scores))
                    model.save_visual_res(save_fig_on, input, val_res, phase)
                    model.update_loss(epoch, end_of_epoch)
                    update_res(detailed_scores, running_val_score)
                    update_res({"val_score": [score]}, running_val_score)
                    loss = score
                    val_res, input = None, None

                elif phase == "debug":
                    print("debugging loss:")
                    debug_res = model.get_evaluation(input)
                    score, detailed_scores = model.analyze_res(
                        debug_res, cache_res=True
                    )
                    print(
                        "debug score of batch {} is {}:".format(
                            model.get_batch_names(), score
                        )
                    )
                    print("debug detailed scores are {}:".format(detailed_scores))
                    model.save_visual_res(save_fig_on, input, debug_res, phase)
                    update_res(detailed_scores, running_debug_score)
                    update_res({"debug_score": [score]}, running_debug_score)
                    loss = score
                    debug_res, input = None, None

                model.do_some_clean()

                # save for tensorboard, both train and val will be saved
                period_loss[phase] += loss
                if not is_train:
                    update_res(detailed_scores, period_detailed_scores[phase])
                if (
                    global_step[phase] > 0
                    and global_step[phase] % tensorboard_print_period[phase] == 0
                ):
                    if not is_train:
                        for metric in period_detailed_scores[phase]:
                            period_avg_detailed_scores = (
                                sum(period_detailed_scores[phase][metric])
                                / tensorboard_print_period[phase]
                            )
                            writer.add_scalar(
                                "{}_" + phase.format(metric),
                                period_avg_detailed_scores,
                                global_step["train"],
                            )
                            period_detailed_scores[phase][metric] = []

                    period_avg_loss = (
                        period_loss[phase] / tensorboard_print_period[phase]
                    )
                    writer.add_scalar(
                        "score/" + phase, period_avg_loss, global_step["train"]
                    )
                    print(
                        "global_step:{}, {} score is{}".format(
                            global_step["train"], phase, period_avg_loss
                        )
                    )
                    period_loss[phase] = 0.0

                if end_of_epoch:
                    break

            if phase == "val":
                model.save_res(phase)
                for metric in running_val_score:
                    epoch_val_score_metric = sum(running_val_score[metric]) / len(
                        running_val_score[metric]
                    )
                    print(
                        "{} epoch_val_{}: {:.4f}".format(
                            epoch, metric, epoch_val_score_metric
                        )
                    )
                epoch_val_score = sum(running_val_score["val_score"]) / len(
                    running_val_score["val_score"]
                )

                if epoch == 0:
                    best_score = epoch_val_score

                if epoch_val_score > best_score:
                    best_score = epoch_val_score
                    best_epoch = epoch
                    save_model(
                        model,
                        check_point_path,
                        epoch,
                        global_step,
                        "epoch_" + str(epoch),
                        True,
                        best_score,
                    )

            if phase == "train":
                if epoch % check_best_model_period == 0:
                    save_model(
                        model,
                        check_point_path,
                        epoch,
                        global_step,
                        "epoch_" + str(epoch),
                        False,
                        best_score,
                    )
                    print("saving the model into thes checkpoint directory")

            if phase == "debug":
                model.save_res(phase, saving=False)
                for metric in running_debug_score:
                    epoch_debug_score_metric = sum(running_debug_score[metric]) / len(
                        running_debug_score[metric]
                    )
                    print(
                        "{} epoch_debug_{}: {:.4f}".format(
                            epoch, metric, epoch_debug_score_metric
                        )
                    )
                epoch_debug_score = sum(running_debug_score["debug_score"]) / len(
                    running_debug_score["debug_score"]
                )
                print("{} epoch_debug_score: {:.4f}".format(epoch, epoch_debug_score))

    time_elapsed = time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val score : {:4f} is at epoch {}".format(best_score, best_epoch))
    writer.close()
    # return the model at the last epoch, not the best epoch
    return model


def save_model(
    model, check_point_path, epoch, global_step, name, is_best=False, best_score=-1
):
    optimizer_state = model.optimizer.state_dict()
    save_checkpoint(
        {
            "epoch": epoch,
            "state_dict": model.get_model().state_dict(),
            "optimizer": optimizer_state,
            "best_score": best_score,
            "global_step": global_step,
        },
        is_best,
        check_point_path,
        name,
        "",
    )
