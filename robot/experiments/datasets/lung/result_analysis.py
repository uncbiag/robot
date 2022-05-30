import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "Liberation Serif"
ID_COPD = {
    "copd6": "copd6",
    "copd7": "copd7",
    "copd8": "copd8",
    "copd9": "copd9",
    "copd10": "copd10",
    "copd1": "copd1",
    "copd2": "copd2",
    "copd3": "copd3",
    "copd4": "copd4",
    "copd5": "copd5",
}

pTVreg = {
    "copd1": 0.71,
    "copd2": 1.91,
    "copd3": 0.77,
    "copd4": 0.67,
    "copd5": 0.71,
    "copd6": 0.66,
    "copd7": 0.75,
    "copd8": 0.78,
    "copd9": 0.64,
    "copd10": 0.85,
}
DGCNN = {
    "copd1": 3.4,
    "copd2": 8.9,
    "copd3": 2.4,
    "copd4": 3.2,
    "copd5": 4.6,
    "copd6": 4.3,
    "copd7": 2.5,
    "copd8": 3.9,
    "copd9": 2.6,
    "copd10": 7.4,
}


def get_experiment_data_from_record(order, path):
    data = np.load(path).squeeze()
    return order, data


def get_experiment_data_from_record_detail(order, path):
    data_detail = np.load(path)
    data = np.mean(data_detail[:, 1:], 1)
    return order, data


def plot_box(data_list, name_list, x_label="Method", y_label="Score", saving_path=None):
    fig, ax = plt.subplots(figsize=(22, 10))
    bplot = ax.boxplot(data_list, vert=True, patch_artist=True)
    ax.yaxis.grid(True)
    ax.set_xticks(
        [y + 1 for y in range(len(data_list))],
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.setp(ax, xticks=[y + 1 for y in range(len(data_list))], xticklabels=name_list)
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(15)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    if saving_path:
        fig1.savefig(saving_path, dpi=300)
    # plt.clf()


def plot_trendency(
    data_list, name_list, x_label="Method", y_label="Score", saving_path=None
):
    data_mean = [np.mean(data) for data in data_list]
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(data_mean)
    plt.xticks(np.arange(len(data_mean)), name_list, rotation=45)
    plt.title("vSVF self-iter")
    # plt.xlabel('vSVF self-iter')
    plt.set_xlabel(x_label)
    plt.set_ylabel(y_label)
    plt.show()
    plt.draw()
    if saving_path:
        fig1.savefig(saving_path, dpi=300)
    # plt.clf()


def compute_std(data_list, name_list):
    for i, name in enumerate(name_list):
        print(
            "the mean and  std  and median of the {}: is {} , {}".format(
                name, np.mean(data_list[i]), np.std(data_list[i])
            ),
            np.median(data_list[i]),
        )


def get_list_from_dic(data_dic, use_log=False, use_perc=False):
    data_list = [None for _ in range(len(data_dic))]
    name_list = [None for _ in range(len(data_dic))]
    for key, item in data_dic.items():
        order = data_dic[key][0]
        data = data_dic[key][1]
        if use_log:
            data = np.log10(data)
            data = data[data != -np.inf]
        if use_perc:
            data = data * 100
        data_list[order] = data
        name_list[order] = key
    return data_list, name_list


def get_df_from_list(name_list, data_list1, name=""):
    data_combined1 = np.array([])
    group_list = np.array([])
    for i in range(len(name_list)):
        data1 = data_list1[i]
        tmp_data1 = np.empty(len(data1))
        tmp_data1[:] = data1[:]
        data_combined1 = np.append(data_combined1, tmp_data1)
        group_list = np.append(group_list, np.array([name_list[i]] * len(data1)))
    group_list = list(group_list)
    df = pd.DataFrame({"Group": group_list, name: data_combined1})
    return df


def get_df_from_double_list(name_list, data_list1, data_list2):
    data_combined1 = np.array([])
    data_combined2 = np.array([])
    group_list = np.array([])
    for i in range(len(name_list)):
        data1 = data_list1[i]
        data2 = data_list2[i]
        if len(data1) != len(data2):
            print(
                "Warning the data1, data2 not consistant, the expr name is {}, len of data1 is {}, len of data2 is {}".format(
                    name_list[i], len(data1), len(data2)
                )
            )
        max_len = max(len(data1), len(data2))
        tmp_data1 = np.empty(max_len)
        tmp_data2 = np.empty(max_len)
        tmp_data1[:] = np.nan
        tmp_data2[:] = np.nan
        tmp_data1[: len(data1)] = data1
        tmp_data2[: len(data2)] = data2
        data_combined1 = np.append(data_combined1, tmp_data1)
        data_combined2 = np.append(data_combined2, tmp_data2)
        group_list = np.append(group_list, np.array([name_list[i]] * max_len))
    group_list = list(group_list)

    df = pd.DataFrame(
        {
            "Group": group_list,
            "Longitudinal": data_combined1,
            "Cross-subject": data_combined2,
        }
    )
    return df


def get_res_dic():
    data_dic = {}
    data_dic["ICP(affine)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/opt_icp/eval/res/records/lmk_diff_mean_records_detail.npy",
    )
    data_dic["RobOT(affine)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/opt_prealign/eval/res/records/lmk_diff_mean_records_detail.npy",
    )
    data_dic["CPD(non-param)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/opt_cpd/eval/res/records/lmk_diff_mean_records_detail.npy",
    )
    data_dic["RobOT(non-param)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/opt_gf/eval/res/records/lmk_diff_mean_records_detail.npy",
    )
    data_dic["S-RobOT(spline)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/opt_discrete_flow_deep/eval/res/records/lmk_diff_mean_records_detail.npy",
        #"/playpen-raid1/zyshen/data/lung_expri/timing/opt_discrete_flow_deep/records/lmk_diff_mean_records_detail.npy",
    )
    data_dic["S-RobOT(LDDMM)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/opt_discrete_flow_deep_lddmm/eval/res/records/lmk_diff_mean_records_detail.npy",
        #"/playpen-raid1/zyshen/data/lung_expri/discrete_flow_on_dirlab_deep_lddmm/records/lmk_diff_mean_records_detail.npy",
    )
    data_dic["DGCNN-CPD"] = inc(), np.array([item[1] for item in DGCNN.items()])
    data_dic["D-RobOT(disp)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/deepflow_pwc_disp/eval/res/records/lmk_diff_mean_and_gf_records_detail.npy",
    )
    data_dic["D-RobOT(spline)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/deepflow_pwc_spline/eval/res/records/lmk_diff_mean_and_gf_records_detail.npy",
    )
    data_dic["D-RobOT(LDDMM)"] = get_experiment_data_from_record(
        inc(),
        "/playpen-raid1/zyshen/proj/robot/robot/demos/output/reproduce/test_deepflow_on_one_case/deepflow_pwc_lddmm/eval/res/records/lmk_diff_mean_gf_records_detail.npy",
    )
    # data_dic['DRobOT(disp_nn)'] = get_experiment_data_from_record(inc(),
    #                                                               '/playpen-raid1/zyshen/data/lung_expri/model_eval/nn/deep_flow_prealign_pwc2_2_continue_60000/records/lmk_diff_mean_gf_records_detail.npy')
    # data_dic['DRobOT(spline_nn)'] = get_experiment_data_from_record(inc(),
    #                                                                 '/playpen-raid1/zyshen/data/lung_expri/model_eval/nn/deep_flow_prealign_pwc_spline_4096_new_60000_8192_aniso_rerun/records/lmk_diff_mean_gf_records_detail.npy')
    # data_dic['DRobOT(LDDMM_nn)'] = get_experiment_data_from_record(inc(),
    #                                                                '/playpen-raid1/zyshen/data/lung_expri/model_eval/nn/deep_flow_prealign_pwc_lddmm_4096_new_60000_8192_aniso_rerun/records/lmk_diff_mean_gf_records_detail.npy')
    # data_dic['DRobOT(disp_nn_ns)'] = get_experiment_data_from_record(inc(),
    #                                                                  '/playpen-raid1/zyshen/data/lung_expri/model_eval/nn_nonsmooth/deep_flow_prealign_pwc2_2_continue_60000/records/lmk_diff_mean_gf_records_detail.npy')
    # data_dic['DRobOT(spline_nn_ns)'] = get_experiment_data_from_record(inc(),
    #                                                                    '/playpen-raid1/zyshen/data/lung_expri/model_eval/nn_nonsmooth/deep_flow_prealign_pwc_spline_4096_new_60000_8192_aniso_rerun/records/lmk_diff_mean_gf_records_detail.npy')
    # data_dic['DRobOT(LDDMM_nn_ns)'] = get_experiment_data_from_record(inc(),
    #                                                                   '/playpen-raid1/zyshen/data/lung_expri/model_eval/nn_nonsmooth/deep_flow_prealign_pwc_lddmm_4096_new_60000_8192_aniso_rerun/records/lmk_diff_mean_gf_records_detail.npy')
    return data_dic


###########################################


def draw_histogram(
    name_list, data_list1, data_list2, label="Jacobi Distribution", fpth=None
):
    n_bins = 10

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
    (
        ax0,
        ax1,
    ) = axes.flatten()

    ax0.hist(data_list1, n_bins, histtype="bar", label=name_list, range=[0, 4])
    ax0.set_title("Longitudinal logJacobi-Iteration Distribution (176 samples)")
    ax0.legend(prop={"size": 10}, loc=2)
    ax1.hist(data_list2, n_bins, histtype="bar", label=name_list, range=[0, 4])
    ax1.set_title("Cross subject logJacobi-Iteration Distribution (300 samples)")
    ax1.legend(prop={"size": 10}, loc=2)

    fig.tight_layout()
    if fpth is not None:
        plt.savefig(fpth, dpi=500, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()
        plt.clf()


def draw_single_boxplot(
    name_list,
    data_list,
    label="Landmark Mean Squared Error",
    titile=None,
    fpth=None,
    data_name=None,
    title=None,
):
    df = get_df_from_list(name_list, data_list, name=data_name)
    df = df[["Group", data_name]]
    dd = pd.melt(df, id_vars=["Group"], value_vars=[data_name], var_name="task")
    fig, ax = plt.subplots(figsize=(15, 10))
    # fig, ax = plt.subplots(figsize=(12, 12))
    sn = sns.boxplot(x="Group", y="value", palette="Set2", data=dd, ax=ax)
    sn.set_title(title, fontsize=50)
    sn.set_xlabel("")
    sn.set_ylabel(label, fontsize=30)
    # plt.xticks(rotation=45)
    ax.yaxis.grid(True)
    for item in (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize(30)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    if fpth is not None:
        plt.savefig(fpth, dpi=500, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()
        plt.clf()


def draw_group_boxplot(
    name_list, data_list1, data_list2, label="Dice Score", titile=None, fpth=None
):
    df = get_df_from_double_list(name_list, data_list1, data_list2)
    df = df[["Group", "Longitudinal", "Cross-subject"]]
    dd = pd.melt(
        df,
        id_vars=["Group"],
        value_vars=["Longitudinal", "Cross-subject"],
        var_name="task",
    )
    fig, ax = plt.subplots(figsize=(15, 8))
    sn = sns.boxplot(x="Group", y="value", data=dd, hue="task", palette="Set2", ax=ax)
    # sns.palplot(sns.color_palette("Set2"))
    sn.set_xlabel("")
    sn.set_ylabel(label)
    # plt.xticks(rotation=45)
    ax.yaxis.grid(True)
    leg = plt.legend(prop={"size": 18}, loc=4)
    leg.get_frame().set_alpha(0.2)
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    if fpth is not None:
        plt.savefig(fpth, dpi=500, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()
        plt.clf()


def plot_group_trendency(
    trend_name,
    trend1,
    trend2,
    label="Average Dice",
    title=None,
    rotation_on=True,
    fpth=None,
):
    trend1_mean = [np.mean(data) for data in trend1]
    trend2_mean = [np.mean(data) for data in trend2]
    max_len = max(len(trend1), len(trend2))
    t = list(range(max_len))
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    # ax1.set_xlabel('step')
    ax1.set_ylabel(label, color=color)
    ln1 = ax1.plot(t, trend1_mean, color=color, linewidth=3.0, label="Longitudinal")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ln2 = ax2.plot(t, trend2_mean, color=color, linewidth=3.0, label="Cross-subject")
    ax2.tick_params(axis="y", labelcolor=color)
    plt.xticks(t, trend_name, rotation=45)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs, loc=0, prop={"size": 20})

    # leg = plt.legend(loc='best')
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(4)
    # get label texts inside legend and set font size
    for text in leg.get_texts():
        text.set_fontsize("x-large")

    for item in (
        [ax1.title, ax1.xaxis.label, ax1.yaxis.label, ax2.yaxis.label]
        + ax1.get_xticklabels()
        + ax1.get_yticklabels()
        + ax2.get_yticklabels()
    ):
        item.set_fontsize(18)
    for tick in ax1.get_xticklabels():
        rotation = 0
        if rotation_on:
            rotation = 30
            tick.set_rotation(rotation)
    plt.title(title, fontsize=20)
    if fpth is not None:
        plt.savefig(fpth, dpi=500, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()
        plt.clf()
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped


order = -1


def inc():
    global order
    order += 1
    return order


#
draw_trendency = False
draw_boxplot = False
title = None
label = "Landmark Mean Square Error"
##################################Get Data ##############################################################


order = -1
data_list, name_list = get_list_from_dic(get_res_dic(), use_perc=False)
order = -1

fpth = "/playpen-raid1/zyshen/debug/lung_plots"
os.makedirs(fpth, exist_ok=True)
def save_json(path, data):
    import json
    with open(path, "w") as f:
        json.dump(data, f)
res = {}
res["data_list"] = [_data.tolist() for _data in data_list]
res["name_list"] = name_list
save_json("./res.json",res)
def load_json(file_path):
    import json
    with open(file_path) as f:
        data_dict = json.load(f)
    return data_dict
res = load_json("./res.json")
fpth = os.path.join(fpth, "res_boxplot.png")
name_list = res["name_list"]
data_list = res["data_list"]
data_list = [np.array(_data) for _data in data_list]
draw_single_boxplot(name_list,data_list,label=label,fpth=fpth,data_name='synth',title='Performance on DirLab')

compute_std(data_list, name_list)
data_list, name_list = get_list_from_dic(get_res_dic(), use_perc=False)
compute_std(data_list, name_list)

order = -1

######################################################compute mean and std ##################################3

# data_list1, name_list = get_list_from_dic(get_res_dic(draw_intra=True, draw_trendency=False),use_perc=True)
# order = -1
# data_list2, _ = get_list_from_dic(get_res_dic(draw_intra=False, draw_trendency=False),use_perc=True)
# order = -1
# compute_std(data_list1, name_list)
# print( "now compute the cross subject ")
# compute_std(data_list2, name_list)

#
# # #################################################### plot boxplot
# if draw_boxplot:
#     draw_group_boxplot(name_list,data_list1,data_list2,label=label)
# #
# ####################################################3 plot trend
# if draw_trendency:
#     plot_group_trendency(name_list, data_list1, data_list2,label, title)
