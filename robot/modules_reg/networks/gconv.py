import torch


class SetConv(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        """
        Module that performs PointNet++-like convolution on point clouds.

        Parameters
        ----------
        nb_feat_in : int
            Number of input channels.
        nb_feat_out : int
            Number of ouput channels.

        Returns
        -------
        None.

        """

        super(SetConv, self).__init__()

        self.fc1 = torch.nn.Conv2d(nb_feat_in + 3, nb_feat_out, 1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc2 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn2 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn3 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, signal, graph):
        """
        Performs PointNet++-like convolution

        Parameters
        ----------
        signal : torch.Tensor
            Input features of size B x N x nb_feat_in.
        graph : flot.models_reg.graph.Graph
            Graph build on the input point cloud on with the input features
            live. The graph contains the list of nearest neighbors (NN) for
            each point and all edge features (relative point coordinates with
            NN).

        Returns
        -------
        torch.Tensor
            Ouput features of size B x N x nb_feat_out.

        """

        # Input features dimension
        b, n, c = signal.shape
        n_out = graph.size[0] // b

        # Concatenate input features with edge features
        signal = signal.reshape(b * n, c)
        signal = torch.cat((signal[graph.edges], graph.edge_feats), -1)
        signal = signal.view(b, n_out, graph.k_neighbors, c + 3)
        signal = signal.transpose(1, -1)

        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.bn1,
            self.lrelu,
            self.fc2,
            self.bn2,
            self.lrelu,
            self.fc3,
            self.bn3,
            self.lrelu,
            self.pool,
        ]:
            signal = func(signal)

        return signal.transpose(1, -1)
