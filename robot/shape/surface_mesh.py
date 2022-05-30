import torch

from robot.shape.shape_base import ShapeBase


class SurfaceMesh(ShapeBase):
    """
    This class is designed for batch based processing.
    3D Triangular mesh.
    For each batch, we assume nodes are subsampled into the same size

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self):

        super(SurfaceMesh, self).__init__()
        self.type = "surfacemesh"
        self.faces = None
        self.index = None
        self.attr_list = [
            "points",
            "faces",
            "index",
            "label",
            "landmarks",
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
        :param faces: BxNx2
        :param index: [index_a_list, index_b_list], each is an overbatch index list with B*N length
        :param reindex: generate index over batch for two ends
        :return:
        """
        ShapeBase.set_data(self, **args)
        faces = (
            args["faces"] if "faces" in args else None
        )  # faces are not always consistent, when run in batch
        # assert faces is not None
        self.faces = faces
        index = args["index"] if "index" in args else None
        reindex = args["reindex"] if "reindex" in args else False
        if index is not None:
            self.index = index
        if not self.points_mode_on and (self.index is None or reindex):
            index_a_list = []
            index_b_list = []
            index_c_list = []
            for b in range(self.nbatch):
                index_a_list += faces[b, 0] + b * self.npoints
                index_b_list += faces[b, 1] + b * self.npoints
                index_c_list += faces[b, 2] + b * self.npoints
            self.index = [index_a_list, index_b_list, index_c_list]
        self.update_info()
        return self

    def set_data_with_refer_to(self, points, mesh, detach=False):
        if not detach:
            fn = lambda x: x
        else:
            fn = lambda x: x.detach().clone() if x is not None else None
        self.points = fn(points)
        self.faces = fn(mesh.faces)
        self.index = fn(mesh.index)
        self.label = fn(mesh.label)
        self.name_list = mesh.name_list
        self.landmarks = fn(mesh.landmarks)
        self.pointfea = fn(mesh.pointfea)
        self.weights = fn(mesh.weights)
        self.seg = fn(mesh.seg)
        self.mask = fn(mesh.mask)
        self.extra_info = mesh.extra_info
        self.scale = mesh.scale
        self.points_mode_on = self.scale != -1
        self.update_info()
        return self

    def get_faces(self):
        return self.faces

    def get_centers_and_normals(self):
        """

        :return: centers:BxNxD, normals: BxNxD
        """
        if self.points_mode_on:
            raise NotImplemented(
                "the topology of the shape has changed, only point related operators are allowed"
            )
        a = self.points.view(-1)[self.index[0]]
        b = self.points.view(-1)[self.index[1]]
        c = self.points.view(-1)[self.index[2]]
        centers = (a + b + c) / 3.0
        normals = torch.cross(b - a, c - a) / 2  # BxNxdim
        zero_normal_index = torch.nonzero(torch.norm(normals, 2, 2) == 0)
        if zero_normal_index.shape[0] > 0:
            normals.data[zero_normal_index] = 1e-7
            print(
                " {} zero normal is detected, set the zero value to 1e-7".format(
                    len(zero_normal_index)
                )
            )
        return centers.view([self.nbatch, -1, self.dimension]), normals.view(
            [self.nbatch, -1, self.dimension]
        )


class SurfaceMesh_Point(SurfaceMesh):
    def __init__(self):
        super(SurfaceMesh_Point, self).__init__()
        self.points_mode_on = True
