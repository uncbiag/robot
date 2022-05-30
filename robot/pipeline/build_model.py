from robot.models_reg.model_opt import OptModel as reg_opt_model
from robot.models_reg.model_deep import DeepModel as reg_deep_model
from robot.models_general.model_deep import DeepModel as general_deep_model


def build_model(opt, devie, gpus):
    """
    create model object
    :param opt: ParameterDict, task setting
    :return: model object
    """
    model_name = opt["model"]
    task_type = opt[
        (
            "task_type",
            "reg",
            "pair-based approach or single input based approach, reg/general",
        )
    ]
    if task_type == "reg" and model_name == "optimization":
        model = reg_opt_model()
    elif task_type == "reg" and model_name == "deep_learning":
        model = reg_deep_model()
    elif task_type == "general":
        model = general_deep_model()
    else:
        raise ValueError(
            "For {} task, Model {} is not recognized.".format(task_type, model_name)
        )
    model.initialize(opt, devie, gpus)
    print("Task {} , model {} is created".format(task_type, model.name()))
    return model
