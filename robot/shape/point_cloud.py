import logging

logger = logging.getLogger(__name__)
from robot.shape.shape_base import ShapeBase


class PointCloud(ShapeBase):
    """
    This class is designed for batch based processing.
    For each batch, we assume the point clouds are sampled into the same size

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self):
        """

        :param points: BxNxD
        """
        super(PointCloud, self).__init__()
        self.type = "pointcloud"
        self.attr_list = ["points", "label", "landmarks", "pointfea", "weights", "seg"]
        self.points_mode_on = True

    def set_data_with_refer_to(self, points, pointcloud, detach=False):
        if not detach:
            fn = lambda x: x
        else:
            fn = lambda x: x.detach().clone() if x is not None else None
        self.points = fn(points)
        self.label = fn(pointcloud.label)
        self.name_list = pointcloud.name_list
        self.landmarks = fn(pointcloud.landmarks)
        self.pointfea = fn(pointcloud.pointfea)
        self.weights = fn(pointcloud.weights)
        self.seg = fn(pointcloud.seg)
        self.mask = fn(pointcloud.mask)
        self.extra_info = pointcloud.extra_info
        self.scale = pointcloud.scale
        self.update_info()
        return self

    def get_centers(self):
        """

        :return: centers:BxNxD
        """
        centers = self.points
        return centers
