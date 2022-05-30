from copy import deepcopy
import random
import torch
from robot.shape.point_sampler import batch_uniform_sampler
from robot.utils.obj_factory import obj_factory
from robot.utils.utils import index_points


class HybirdData(object):
    def __init__(
        self,
        synthsizer_obj,
        data_aug_obj=None,
        raw_source_target_has_corr=False,
        corr_sampled_source_target=False,
        test_time_randomize=False,
        npoints=-1,
        synth_ratio=1.0,
        ratio_decay_rate=8,
        min_synth_ratio=0.3,
        sampled_by_weight=True,
    ):
        super(HybirdData, self).__init__()
        self.synthsizer = obj_factory(synthsizer_obj) if synthsizer_obj else None
        self.data_aug = obj_factory(data_aug_obj) if data_aug_obj else None
        self.synth_ratio = synth_ratio
        self.ratio_decay_rate = ratio_decay_rate
        self.min_synth_ratio = min_synth_ratio
        self.sampler = None
        self.raw_source_target_has_corr = raw_source_target_has_corr
        self.corr_sampled_source_target = corr_sampled_source_target
        self.test_time_randomize = test_time_randomize
        self.sampled_by_weight = sampled_by_weight
        self.npoints = npoints

    def sampling(self, input_data, use_synth):
        sp, sw = input_data["source"]["points"], input_data["source"]["weights"]
        tp, tw = input_data["target"]["points"], input_data["target"]["weights"]

        (
            input_data["source"]["points"],
            input_data["source"]["weights"],
            ssind,
        ) = self.sampler(sp, sw)
        ssind = ssind.long()
        if "pointfea" in input_data["source"]:
            input_data["source"]["pointfea"] = index_points(
                input_data["source"]["pointfea"], ssind
            )
        input_data["source"]["extra_info"] = {
            key: index_points(item, ssind)
            for key, item in input_data["source"].get("extra_info", {}).items()
        }
        if use_synth or self.raw_source_target_has_corr:
            gf = tp - sp
            input_data["source"]["extra_info"].update(
                {"gt_flow": index_points(gf, ssind)}
            )  # here we save gt_flow to both source and target, another option is save it to shapepair
        if self.corr_sampled_source_target:
            tsind = ssind
            input_data["target"]["points"], input_data["target"]["weights"] = (
                index_points(input_data["target"]["points"], tsind),
                index_points(input_data["target"]["weights"], tsind),
            )
        else:
            (
                input_data["target"]["points"],
                input_data["target"]["weights"],
                tsind,
            ) = self.sampler(tp, tw)

        if "pointfea" in input_data["target"]:
            input_data["target"]["pointfea"] = index_points(
                input_data["target"]["pointfea"], tsind
            )
        input_data["extra_info"] = {
            key: index_points(item, tsind)
            for key, item in input_data["target"].get("extra_info", {}).items()
        }
        if "gt_flow" in input_data["source"]["extra_info"]:
            input_data["extra_info"]["gt_flow"] = input_data["source"]["extra_info"][
                "gt_flow"
            ]  # here we save gt_flow to both source and target, another option is save it to shapepair
            input_data["extra_info"]["gt_flowed"] = (
                input_data["extra_info"]["gt_flow"] + input_data["source"]["points"]
            )  # here we save gt_flow to both source and target, another option is save it to shapepair
        return input_data

    def non_sampling(self, input_data, use_synth):
        input_data["extra_info"] = input_data.get("extra_info", {})
        if self.raw_source_target_has_corr or use_synth:
            input_data["extra_info"]["gt_flow"] = (
                input_data["target"]["points"] - input_data["source"]["points"]
            )
            input_data["extra_info"]["gt_flowed"] = input_data["target"][
                "points"
            ].clone()
        return input_data

    def prepare_synth_input(self, input_data, batch_info):
        synth_on_source = random.random() > 0.5
        source_dict = input_data["source"] if synth_on_source else input_data["target"]
        input_data["target"], synth_info = self.synthsizer(deepcopy(source_dict))
        if self.data_aug is not None:
            input_data["source"], _ = self.data_aug(source_dict)
        input_data["source"] = source_dict
        batch_info["source_info"] = (
            batch_info["source_info"] if synth_on_source else batch_info["target_info"]
        )
        batch_info["target_info"] = batch_info["source_info"]
        batch_info["pair_name"] = [
            name + "_and_synth" for name in batch_info["source_info"]["name"]
        ]
        batch_info["synth_info"] = synth_info
        batch_info["is_synth"] = True
        batch_info["corr_source_target"] = True
        return input_data, batch_info

    def prepare_raw_pair(self, input_data, batch_info):
        if self.data_aug is not None and batch_info["phase"] == "train":
            input_data["source"], _ = self.data_aug(input_data["source"])
            input_data["target"], _ = self.data_aug(input_data["target"])
        batch_info["is_synth"] = False
        batch_info["corr_source_target"] = self.raw_source_target_has_corr

        return input_data, batch_info

    def update_synth_ratio(self, epoch):
        from robot.utils.utils import sigmoid_decay

        cur_synth_ratio = max(
            sigmoid_decay(epoch, static=20, k=self.ratio_decay_rate) * self.synth_ratio,
            self.min_synth_ratio,
        )
        return cur_synth_ratio

    def planner(self, phase, current_epoch):
        """
        1. the  input data already has source-target one-to-one correspondence: raw_source_target_has_corr = True
            during the train, the data can be either real or synth depending on synth_ratio, the corr_sampled_source_target depends on setting
            during the val, the data is real, the corr_sampled_source_target depends on setting
            during the debug, the data is real,  corr_sampled_source_target will be true

            in any case, we can infer the ground truth flow and set has_gt=True

        2. the input data don't have one-to-one correspondence: raw_source_target_has_corr = False
            during the train, the data can be either real or synth depending on synth_ratio, the corr_sampled_source_target depends on setting
            during the val, the data is real, the corr_sampled_source_target depends on setting, has_gt=self.raw_source_target_has_corr
            during the debug, the data is synth,  corr_sampled_source_target will be true, has_gt=True

            if the data is synth, we can infer the gt flow, set has_gt=True otherwise has_gt=False
        :param phase:
        :param current_epoch:
        :return:
        """

        if phase == "debug":
            self.sampler = batch_uniform_sampler(
                self.npoints,
                fixed_random_seed=not self.test_time_randomize,
                sampled_by_weight=self.sampled_by_weight,
            )
            return not self.raw_source_target_has_corr, True, True
        elif phase == "val":
            self.sampler = batch_uniform_sampler(
                self.npoints,
                fixed_random_seed=not self.test_time_randomize,
                sampled_by_weight=self.sampled_by_weight,
            )
            return (
                False,
                self.corr_sampled_source_target,
                self.raw_source_target_has_corr,
            )
        elif phase == "test":
            self.sampler = batch_uniform_sampler(
                self.npoints,
                fixed_random_seed=not self.test_time_randomize,
                sampled_by_weight=self.sampled_by_weight,
            )
            return (
                False,
                self.corr_sampled_source_target,
                self.raw_source_target_has_corr,
            )
        elif phase == "train":
            self.sampler = batch_uniform_sampler(
                self.npoints, fixed_random_seed=False, sampled_by_weight=self.sampled_by_weight
            )
            synth_ratio = (
                self.synth_ratio
                if self.ratio_decay_rate == -1
                else self.update_synth_ratio(current_epoch)
            )
            use_synth = random.random() < synth_ratio
            return (
                use_synth,
                self.corr_sampled_source_target,
                use_synth or self.raw_source_target_has_corr,
            )

    def __call__(self, input_data, batch_info):
        """
        here list several common scenario
        raw_source_target_has_corr: the input source and target has sorted one-to-one mapping
        corr_source_target:  the output source and target has sorted one-to-one mapping
        gt_flow : the output source and target has ground truth flow, but the source and target are not necessary to be one-to-one sorted mapped

        1. the  input data already has source-target one-to-one correspondence: raw_source_target_has_corr = True
            1.1 if not use the synth data
                if set corr_sampled_source_target=True, then corr_source_target=True, otherwise  corr_source_target=False
            1.2 if use the synth data
                if set corr_sampled_source_target=True, then corr_source_target=True, otherwise  corr_source_target=False
            the gt_flow will be provided in any case

        2. the input data don't have one-to-one correspondence: raw_source_target_has_corr = False
            2.1 if not use the synth data
                not matter how to set corr_sampled_source_target, corr_source_target=False
                the gt_flow is not provided
            2.2. if use the synth data
                if set corr_sampled_source_target=True, then corr_source_target=True, otherwise  corr_source_target=False
                the gt_flow is provided


        :param input_data:
        :param batch_info:
        :return:
        """
        use_synth, corr_sampled_source_target, has_gt = self.planner(
            batch_info["phase"], batch_info["epoch"]
        )
        batch_info["has_gt"] = has_gt
        if use_synth:
            input_data, batch_info = self.prepare_synth_input(input_data, batch_info)
        else:
            input_data, batch_info = self.prepare_raw_pair(input_data, batch_info)
        if self.npoints > 0:
            batch_info["corr_source_target"] = (
                batch_info["corr_source_target"] and corr_sampled_source_target
            )
            return self.sampling(input_data, use_synth), batch_info
        else:
            return self.non_sampling(input_data, use_synth), batch_info
