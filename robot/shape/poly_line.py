import torch

from robot.shape.shape_base import ShapeBase


class PolyLine(ShapeBase):
    """
    This class is designed for batch based processing.
    2d polyline with edges
    For each batch, we assume nodes are sampled into the same size

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self):
        """

        :param points: BxNxD
        :param edges: BxNx2
        """
        super(PolyLine, self).__init__()
        self.type = "polyline"
        self.edges = None
        self.index = None
        self.attr_list = [
            "points",
            "edges",
            "index",
            "label",
            "landmarks",
            "name_list",
            "pointfea",
            "weights",
            "seg",
            "mask",
        ]
        self.points_mode_on = False
        """the mesh sampling is not implemented, if the topology changed, only points related operators are allowed"""

    def set_data(self, **args):
        """

        :param points: BxNxD
        :param edges: BxNx2
        :param index: [index_a_list, index_b_list], each is an overbatch index list with B*N length
        :param reindex: generate index over batch for two ends
        :return:
        """
        ShapeBase.set_data(self, **args)
        edges = args["edges"]
        assert edges is not None
        self.edges = edges
        index = args["index"] if "index" in args else None
        reindex = args["reindex"] if "reindex" in args else False
        if index is not None:
            self.index = index
        if self.index is None or reindex:
            index_a_list = []
            index_b_list = []
            for b in range(self.nbatch):
                index_a_list += edges[b, 0] + b * self.npoints
                index_b_list += edges[b, 1] + b * self.npoints
            self.index = [index_a_list, index_b_list]
        self.update_info()

    def set_data_with_refer_to(self, points, polyline, detach=False):
        if not detach:
            fn = lambda x: x
        else:
            fn = lambda x: x.detach().clone() if x is not None else None
        self.points = fn(points)
        self.edges = fn(polyline.edges)
        self.index = fn(polyline.index)
        self.label = fn(polyline.label)
        self.name_list = polyline.name_list
        self.landmarks = fn(polyline.landmarks)
        self.pointfea = fn(polyline.pointfea)
        self.weights = fn(polyline.weights)
        self.seg = fn(polyline.seg)
        self.scale = polyline.scale
        self.points_mode_on = self.scale != -1
        self.update_info()
        return self

    def get_edges(self):
        return self.edges

    def get_centers_and_currents(self):
        """

        :return: centers:BxNxD, currents: BxNxD
        """
        if self.points_mode_on:
            raise NotImplemented(
                "the topology of the shape has changed, only point related operators are allowed"
            )

        a = self.points.view(-1)[self.index[0]]
        b = self.points.view(-1)[self.index[1]]
        centers = (a + b) / 2.0
        currents = b - a
        zero_normal_index = torch.nonzero(torch.norm(currents, 2, 2) == 0)
        if zero_normal_index.shape[0] > 0:
            currents.data[zero_normal_index] = 1e-7
            print(
                " {} zero normal is detected, set the zero value to 1e-7".format(
                    len(zero_normal_index)
                )
            )
        return centers.view([self.nbatch, -1, self.dimension]), currents.view(
            [self.nbatch, -1, self.dimension]
        )
