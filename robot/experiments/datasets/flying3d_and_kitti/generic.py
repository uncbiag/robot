import torch
import numpy as np
from torch.utils.data import Dataset


class Batch:
    def __init__(self, batch):
        """
        Concatenate list of dataset.generic.SceneFlowDataset's item in batch
        dimension.

        Parameters
        ----------
        batch : list
            list of dataset.generic.SceneFlowDataset's item.

        """

        self.data = {}
        batch_size = len(batch)
        for key in ["sequence", "ground_truth"]:
            self.data[key] = []
            for ind_seq in range(2):
                tmp = []
                for ind_batch in range(batch_size):
                    tmp.append(batch[ind_batch][key][ind_seq])
                self.data[key].append(torch.cat(tmp, 0))

    def __getitem__(self, item):
        """
        Get 'sequence' or 'ground_thruth' from the batch.

        Parameters
        ----------
        item : str
            Accept two keys 'sequence' or 'ground_truth'.

        Returns
        -------
        list(torch.Tensor, torch.Tensor)
            item='sequence': returns a list [pc1, pc2] of point clouds between
            which to estimate scene flow. pc1 has size B x n x 3 and pc2 has
            size B x m x 3.

            item='ground_truth': returns a list [mask, flow]. mask has size
            B x n x 1 and flow has size B x n x 3. flow is the ground truth
            scene flow between pc1 and pc2. flow is the ground truth scene
            flow. mask is binary with zeros indicating where the flow is not
            valid or occluded.

        """
        return self.data[item]

    def to(self, *args, **kwargs):

        for key in self.data.keys():
            self.data[key] = [d.to(*args, **kwargs) for d in self.data[key]]

        return self

    def pin_memory(self):

        for key in self.data.keys():
            self.data[key] = [d.pin_memory() for d in self.data[key]]

        return self


class SceneFlowDataset(Dataset):
    def __init__(self, nb_points=-1):
        """
        Abstract constructor for scene flow datasets.

        Each item of the dataset is returned in a dictionary with two keys:
            (key = 'sequence', value=list(torch.Tensor, torch.Tensor)):
            list [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.

            (key = 'ground_truth', value = list(torch.Tensor, torch.Tensor)):
            list [mask, flow]. mask has size 1 x n x 1 and pc1 has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

        Parameters
        ----------
        nb_points : int
            Maximum number of points in point clouds: m, n <= self.nb_points.

        """

        super(SceneFlowDataset, self).__init__()
        self.nb_points = nb_points

    def __getitem__(self, idx):

        sequence, ground_truth = self.to_torch(
            *self.subsample_points(*self.load_sequence(idx))
        )
        data = {"sequence": sequence, "ground_truth": ground_truth}

        return data

    def to_torch(self, sequence, ground_truth):
        """
        Convert numpy array and torch.Tensor.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size n x 3 and pc2 has size m x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is
            binary with zeros indicating where the flow is not valid/occluded.

        Returns
        -------
        sequence : list(torch.Tensor, torch.Tensor)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.

        ground_truth : list(torch.Tensor, torch.Tensor)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

        """

        sequence = [torch.unsqueeze(torch.from_numpy(s), 0).float() for s in sequence]
        ground_truth = [
            torch.unsqueeze(torch.from_numpy(gt), 0).float() for gt in ground_truth
        ]

        return sequence, ground_truth

    def subsample_points(self, sequence, ground_truth):
        """
        Subsample point clouds randomly.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x N x 3 and pc2 has size 1 x M x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x N x 1 and pc1 has size
            1 x N x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3. The n
            points are chosen randomly among the N available ones. The m points
            are chosen randomly among the M available ones. If N, M >=
            self.nb_point then n, m = self.nb_points. If N, M <
            self.nb_point then n, m = N, M.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not
            valid/occluded.

        """

        # Choose points in first scan
        ind1 = np.random.permutation(sequence[0].shape[0])[: self.nb_points]
        sequence[0] = sequence[0][ind1]
        ground_truth = [g[ind1] for g in ground_truth]

        # Choose point in second scan
        ind2 = np.random.permutation(sequence[1].shape[0])[: self.nb_points]
        sequence[1] = sequence[1][ind2]

        return sequence, ground_truth

    def load_sequence(self, idx):
        """
        Abstract function to be implemented to load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Must return:
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size N x 3 and pc2 has size M x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size N x 1 and pc1 has size N x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is
            binary with zeros indicating where the flow is not valid/occluded.

        """

        raise NotImplementedError
