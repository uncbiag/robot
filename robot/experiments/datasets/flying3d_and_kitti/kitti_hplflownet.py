import os
import numpy as np
from .generic import SceneFlowDataset


class Kitti(SceneFlowDataset):
    def __init__(self, root_dir, nb_points):
        """
        Construct the KITTI scene flow datatset as in:
        Gu, X., Wang, Y., Wu, C., Lee, Y.J., Wang, P., HPLFlowNet: Hierarchical
        Permutohedral Lattice FlowNet for scene ﬂow estimation on large-scale
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition
        (CVPR). pp. 3254–3263 (2019)

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.

        """

        super(Kitti, self).__init__(nb_points)
        self.root_dir = root_dir
        self.paths = self.make_dataset()

    def __len__(self):

        return len(self.paths)

    def make_dataset(self):
        """
        Find and filter out paths to all examples in the dataset.

        """

        #
        root = os.path.realpath(os.path.expanduser(self.root_dir))
        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        assert len(useful_paths) == 200, "Problem with size of kitti dataset"

        # Mapping / Filtering of scans as in HPLFlowNet code
        mapping_path = os.path.join(os.path.dirname(__file__), "KITTI_mapping.txt")
        with open(mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]
        useful_paths = [
            path for path in useful_paths if lines[int(os.path.split(path)[-1])] != ""
        ]

        return useful_paths

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size n x 3 and pc2 has size m x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is
            binary with zeros indicating where the flow is not valid/occluded.

        """

        # Load data
        sequence = [np.load(os.path.join(self.paths[idx], "pc1.npy"))]
        sequence.append(np.load(os.path.join(self.paths[idx], "pc2.npy")))

        # Remove ground points
        is_ground = np.logical_and(sequence[0][:, 1] < -1.4, sequence[1][:, 1] < -1.4)
        not_ground = np.logical_not(is_ground)
        sequence = [sequence[i][not_ground] for i in range(2)]

        # Remove points further than 35 meter away as in HPLFlowNet code
        is_close = np.logical_and(sequence[0][:, 2] < 35, sequence[1][:, 2] < 35)
        sequence = [sequence[i][is_close] for i in range(2)]

        # Scene flow
        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            sequence[1] - sequence[0],
        ]  # [Occlusion mask, scene flow]

        return sequence, ground_truth
