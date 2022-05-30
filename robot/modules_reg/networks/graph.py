import torch


class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models_reg.graph.Graph
            Graph build on input point cloud containing the list of nearest
            neighbors (NN) for each point and all edge features (relative
            coordinates with NN).

        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        effective_nb_neighbors = neighbors.shape[-1]
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph
