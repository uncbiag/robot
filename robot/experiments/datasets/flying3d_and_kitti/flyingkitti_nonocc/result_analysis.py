import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "Liberation Serif"


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



def get_performance():
    data_dic = {}
    global order
    order = -1
    task_folder = (
        "/playpen-raid1/zyshen/data/flying3d_nonocc_test_on_kitti/model_eval/timing"
    )
    task_name = "deepflow_official_8192_with_aug_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(), os.path.join(task_folder, task_name, "records/EPE3D_records_detail.npy")
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_8192_with_aug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/EPE3D_and_gf_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_30000_withaug_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(), os.path.join(task_folder, task_name, "records/EPE3D_records_detail.npy")
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_30000_withaug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/EPE3D_and_gf_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_spline_8192_withaug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/EPE3D_gf_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_spline_30000_withaug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/EPE3D_and_gf_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_spline_30000_withaug_kitti_prealigned_2step"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/EPE3D_and_gf_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "opt_gf_8192_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(), os.path.join(task_folder, task_name, "records/EPE3D_records_detail.npy")
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "opt_gf_30000_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(), os.path.join(task_folder, task_name, "records/EPE3D_records_detail.npy")
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_8192_flot_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(), os.path.join(task_folder, task_name, "records/EPE3D_records_detail.npy")
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = 'deepflow_official_8192_flot_kitti_prealigned'
    data_dic[task_name] = get_experiment_data_from_record(inc(), os.path.join(task_folder, task_name,'records/EPE3D_and_gf_records_detail.npy'))
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    return data_dic


def get_speed():
    data_dic = {}
    global order

    order = -1
    task_folder = (
        "/playpen-raid1/zyshen/data/flying3d_nonocc_test_on_kitti/model_eval/timing"
    )
    batch_task_folder = (
        "/playpen-raid1/zyshen/data/flying3d_nonocc_larger_kitti/model_eval/batch"
    )
    task_name = "deepflow_official_8192_with_aug_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = "deepflow_official_8192_with_aug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = "deepflow_official_30000_withaug_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_30000_withaug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_spline_8192_withaug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = "deepflow_spline_30000_withaug_kitti_prealigned"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = "deepflow_spline_30000_withaug_kitti_prealigned_2step"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = "opt_gf_8192_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = "opt_gf_30000_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = "deepflow_official_8192_flot_kitti"
    data_dic[task_name] = get_experiment_data_from_record(
        inc(),
        os.path.join(task_folder, task_name, "records/forward_t_records_detail.npy"),
    )
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(
        inc(),
        os.path.join(
            batch_task_folder, task_name, "records/forward_t_records_detail.npy"
        ),
    )
    task_name = 'deepflow_official_8192_flot_kitti_prealigned'
    data_dic[task_name] = get_experiment_data_from_record(inc(), os.path.join(task_folder, task_name,'records/forward_t_records_detail.npy'))
    data_dic[task_name + "_batch"] = get_experiment_data_from_record(inc(), os.path.join(batch_task_folder, task_name, 'records/forward_t_records_detail.npy'))

    return data_dic


###########################################


def get_memory():
    data_dic = {}
    global order

    order = -1
    task_name = "deepflow_official_8192_with_aug_kitti"
    data_dic[task_name] = inc(), 1016
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_8192_with_aug_kitti_prealigned"
    data_dic[task_name] = inc(), 1.01 * 1024
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_30000_withaug_kitti"
    data_dic[task_name] = inc(), 10.44 * 1024
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_30000_withaug_kitti_prealigned"
    data_dic[task_name] = inc(), 10.44 * 1024
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_spline_8192_withaug_kitti_prealigned"
    data_dic[task_name] = inc(), 396
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_spline_30000_withaug_kitti_prealigned"
    data_dic[task_name] = inc(), 610
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_spline_30000_withaug_kitti_prealigned_2step"
    data_dic[task_name] = inc(), 780
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "opt_gf_8192_kitti"
    data_dic[task_name] = inc(), 4.0
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "opt_gf_30000_kitti"
    data_dic[task_name] = inc(), 89
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = "deepflow_official_8192_flot_kitti"
    data_dic[task_name] = inc(), 2.76 * 1024
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    task_name = 'deepflow_official_8192_flot_kitti_prealigned'
    data_dic[task_name] = inc(), 2.76 * 1024
    data_dic[task_name + "_batch"] = inc(), data_dic[task_name][1]
    return data_dic


###########################################


def get_kitti_res(return_detail=False):
    res = {}
    perf, perf_name = get_list_from_dic(get_performance())
    time, time_name = get_list_from_dic(get_speed())
    memory, memory_name = get_list_from_dic(get_memory(), use_log=True)  #
    assert perf_name == time_name
    assert time_name == memory_name
    if return_detail:
        res["performance"] = perf
        res["time"] = time
        res["memory"] = memory
    else:
        res["performance"] = [np.mean(item) for item in perf]
        res["time"] = [np.mean(item) for item in time]
        res["memory"] = [np.mean(item) for item in memory]

    return res, perf_name


def scatter_plot(name_list, x, y, xy_value, title, fpth=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.suptitle(title, fontsize=50)
    sc = ax.scatter(x, y, s=200, c=xy_value, cmap="tab10")  #
    cbar = plt.colorbar(sc)
    cbar.ax.get_xaxis().labelpad = 10
    # cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_xlabel("        Memory", fontsize=40)  # , rotation=90
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_yticklabels(
        ["10 MB", "30 MB", "0.1 GB", "0.3 GB", "1 GB", "3 GB", "10 GB"]
    )
    for i, name in enumerate(name_list):
        if name == "D-RobOT(PWC)":
            ax.annotate(name, (x[i] - 0.012, y[i] +40), fontsize=25)
        elif name == "RobOT":
            ax.annotate(name, (x[i] - 0.010, y[i]), fontsize=25)
        elif name == "PWC":
            ax.annotate(name, (x[i] + 0.002, y[i]-10), fontsize=25)
        else:
            ax.annotate(name, (x[i] + 0.001, y[i]), fontsize=25)

    ax.yaxis.grid(True)
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    plt.xlabel("EPE3D (m)", fontsize=40)
    plt.ylabel("Time (ms)", fontsize=40)
    plt.tight_layout()
    if fpth is not None:
        plt.savefig(fpth, dpi=500, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()
        plt.clf()


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
    name_list, data_list1, data_list2, label="Dice Score", title=None, fpth=None
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
#
res, name_list = get_kitti_res()
filtered = ["_batch" not in name and "_2step" not in name for name in name_list]
name_list = [
    "PWC",
    "PWC_batch",
    "D-RobOT(PWC)",
    "D-RobOT(PWC)_batch",
    "PWC(dense)",
    "PWC(dense)_batch",
    "D-RobOT(PWC&dense)",
    "D-RobOT(PWC&dense)_batch",
    "D-RobOT(spline)",
    "D-RobOT(spline)_batch",
    "D-RobOT(spline&dense)",
    "D-RobOT(spline&dense)_batch",
    "spline_im_dense_2step",
    "spline_im_dense_2step_batch",
    "RobOT",
    "RobOT_batch",
    "RobOT(dense)",
    "RobOT(dense)_batch",
    "FLOT",
    "FLOT_batch",
    "D-RobOT(FLOT)",
    "D-RobOT(FLOT)_batch",
]
name_list = [name.replace("_batch", "") for name in name_list]
name_list = [name_list[i] for i, ind in enumerate(filtered) if ind]
res["performance"] = [res["performance"][i] for i, ind in enumerate(filtered) if ind]
res["time"] = [res["time"][i] for i, ind in enumerate(filtered) if ind]
res["memory"] = [res["memory"][i] for i, ind in enumerate(filtered) if ind]

def save_json(path, data):
    import json
    with open(path, "w") as f:
        json.dump(data, f)

res["name_list"] = name_list
save_json("./res.json",res)
def load_json(file_path):
    import json
    with open(file_path) as f:
        data_dict = json.load(f)
    return data_dict
res = load_json("./res.json")
fpth = "/playpen-raid1/zyshen/debug/kitti_plots"
os.makedirs(fpth, exist_ok=True)
fpth = os.path.join(fpth, "scatter_plot.png")
scatter_plot(
    res["name_list"],
    res["performance"],
    res["time"],
    res["memory"],
    "Performance on KITTI",
    fpth=None,
)
print({res["name_list"][i]: res["performance"][i] for i in range(len(res["name_list"]))})