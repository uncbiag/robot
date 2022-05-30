import os
import torch
from copy import deepcopy


def resume_train(model_path, model, optimizer=None):
    """
    resume the training from checkpoint
    :param model_path: the checkpoint path
    :param model: the model to be set
    :param optimizer: the optimizer to be set
    :return:
    """
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(
            model_path, map_location="cpu"
        )  # {'cuda:'+str(old_gpu):'cuda:'+str(cur_gpu)})
        start_epoch = 0
        best_prec1 = 0.0
        load_only_one = False
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print("the started epoch now is {}".format(start_epoch))
        else:
            start_epoch = 0
        if "best_loss" in checkpoint:
            best_prec1 = checkpoint["best_loss"]
        else:
            best_prec1 = 0.0
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        else:
            phases = ["train", "val", "debug"]
            global_step = {x: 0 for x in phases}
        try:
            model.load_state_dict(checkpoint["state_dict"])
            print("=> succeed load model '{}'".format(model_path))
        except:
            print(
                "Warning !!! Meet error is reading the whole model, now try to read the part"
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(" The incomplelet model is succeed load from '{}'".format(model_path))
        if "optimizer" in checkpoint:
            if not isinstance(optimizer, tuple):
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    # for state in optimizer.state.values():
                    #     for k, v in state.items():
                    #         if isinstance(v, torch.Tensor):
                    #             state[k] = v.cuda()
                    print("=> succeed load optimizer '{}'".format(model_path))
                    optimizer.zero_grad()
                except:
                    print(
                        "Warning !!! Meet error during loading the optimize, not externaly initialized"
                    )

        return start_epoch, best_prec1, global_step
    else:
        print("=> no checkpoint found at '{}'".format(model_path))


get_test_model = resume_train


def save_checkpoint(state, is_best, path, prefix, filename="checkpoint.pth.tar"):
    """
    save checkpoint during training
    'epoch': epoch,'
    :param state_dict': {'epoch': epoch,'state_dict':  model.network.state_dict(),'optimizer': optimizer_state,
                  'best_score': best_score, 'global_step':global_step}
    :param is_best: if is the best model
    :param path: path to save the checkpoint
    :param prefix: prefix to add before the fname
    :param filename: filename
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    prefix_save = os.path.join(path, prefix)
    name = "_".join([prefix_save, filename])
    torch.save(state, name)
    if is_best:
        torch.save(state, path + "/model_best.pth.tar")


def print_model(net):
    """ print out the structure and #parameter of the model"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


def update_res(metrics_res, recorder):
    for metric in metrics_res:
        if metric in recorder:
            recorder[metric] += deepcopy(metrics_res[metric])
        else:
            recorder.update({metric: deepcopy(metrics_res[metric])})


class VisualWarpper(torch.nn.Module):
    def __init__(self, network):
        super(VisualWarpper, self).__init__()
        self.network = network

    def forward(self, input):
        output = self.network((input, None))
        return -output[
            3
        ]  # here we inverse the fea because the class 0 is what we interested
