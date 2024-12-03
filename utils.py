import shutil
import random
import os
import numpy as np
import torch
from typing import List
from torch_geometric.data import Data

import torch.nn.functional as F
import torch.nn as nn

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data
import torch_geometric
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_undirected

class FocalLoss(nn.Module):

    def __init__(self, weight: torch.Tensor = None, gamma: float = 2., reduction: str = 'none'):
        """
         Initialize the module. This is the entry point for the module. You can override this if you want to do something other than setting the weights and / or gamma.

         Args:
         	 weight: The weight to apply to the layer. If None the layer weights are set to 1.
         	 gamma: The gamma parameter for the layer. Defaults to 2.
         	 reduction: The reduction method to apply. Possible values are'mean'or'std '
        """
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
         Computes NLL loss for each element of input_tensor. This is equivalent to : math : ` L_ { t } ` where L is the log - softmax of the input tensor

         Args:
         	 input_tensor: Tensor of shape ( batch_size num_input_features )
         	 target_tensor: Tensor of shape ( batch_size num_target_features )

         Returns:
         	 A tensor of shape ( batch_size num_output_features ) - > loss ( float )
        """
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
     Computes F1 loss for classification. It is used to compute the F1 loss for each class and its predicted values

     Args:
     	 y_true: ( torch. Tensor ) Ground truth labels
     	 y_pred: ( torch. Tensor ) Predicted labels

     Returns:
     	 ( torch. Tensor ) Corresponding F1 loss ( tp tn fn fn p r r )
    """
    tp = torch.sum((y_true * y_pred).float(), dim=0)
    tn = torch.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)

def create_folder(log_dir: str) -> None:
    """
     Creates folder for log files. This will overwrite existing folder if it exists. It will be created as a temporary folder in order to avoid file duplication.

     Args:
     	 log_dir: path to log directory. The directory will be created if it does not exist

     Returns:
     	 True if success False
    """
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        # print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

def set_seed(seed: int) -> None:
    """
     Set the seed for random np. random and torch. cuda. This is useful for reproducibility and to avoid seeding the torch module in the middle of a test.

     Args:
     	 seed: The seed to use. If None is passed the seed is set to the default
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

class DGM_c(nn.Module):
    input_dim = 4
    debug=False

    def __init__(self, embed_f, k=None, distance: str = 'euclidean'):
        """
         Initialize DGM_c. Args : embed_f : Embedding function to be used for training.

         Args:
         	 embed_f
         	 k
         	 distance
        """
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.distance = distance

        self.scale = nn.Parameter(torch.tensor(-1).float(),requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1,1,DGM_c.input_dim)).float(),requires_grad=False)


    def forward(self, x, A, not_used=None, fixedges=None):
        """
        Forward pass of DGM. In this case we are going to estimate the norm of the embeddings and use it to normalize the input

        Args:
            x: torch. Tensor of shape [ nb_samples dim ]
            A: torch. Tensor of shape [ nb_samples dim ]
            not_used: Not used in this implementation. Ignored.
            fixedges: List of fixedges. Ignored.

        Returns:
            torch. Tensor of shape [ nb_samples dim ] A : torch. Tensor of shape [ nb_samples
        """

        x = self.embed_f(x,A)

        # estimate normalization parameters
        if self.scale <0:
            self.centroid.data = x.mean(-2,keepdim=True).detach()
            self.scale.data = (0.9/(x-self.centroid).abs().max()).detach()

        if self.distance=='hyperbolic':
            D, _x = pairwise_poincare_distances((x-self.centroid)*self.scale)
        else:
            D, _x = pairwise_euclidean_distances((x-self.centroid)*self.scale)

        A = torch.sigmoid(self.temperature*(self.threshold.abs()-D))

        if DGM_c.debug:
            self.A = A.data.cpu()
            self._x = _x.data.cpu()

#         self.A=A
#         A = A/A.sum(-1,keepdim=True)
        return x, A, None

def pairwise_euclidean_distances(x: torch.Tensor, dim: int = -1) -> list:
    """
     Computes Euclidean distances between a tensor and its elements. This is a wrapper around torch. cdist for use in TensorFlow

     Args:
     	 x: Tensor to compute distances from
     	 dim: Dimension of x ( default : - 1 )

     Returns:
     	 List of distance and input tensor ( s ) in form [ dist x ] where dist is the distance between
    """
    dist = torch.cdist(x,x)**2
    return [dist, x]

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x: torch.Tensor, dim: int = -1) -> list:
    """
     Computes poincare distances between x and each element of x. It is used to compute pairwise distances between vectors of length n ( where n is the number of elements in x )

     Args:
     	 x: torch. Tensor of shape ( n )
     	 dim: int dimension along which to compute distances. Default : - 1

     Returns:
     	 tuple of ( torch. Tensor of shape ( n ) x ) where dist is the distance between x and each
    """
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)

    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return (dist, x)


def rotmat2graph(k: int, num_nodes: int) -> torch_geometric.data.Data:
    """
     Creates a graph from rotmat. This is a function to be used in conjunction with graph2k

     Args:
     	 k: number of nodes in the graph
     	 num_nodes: number of nodes in the graph ( not used in this function )

     Returns:
     	 torch_geometric.data.Data object that contains the graph
    """

    x_angles = torch.load('/media/johannes/WD Elements/NSCLC_Stefan/augmented_slices/x_rotations.pt')[:num_nodes]
    y_angles = torch.load('/media/johannes/WD Elements/NSCLC_Stefan/augmented_slices/y_rotations.pt')[:num_nodes]
    z_angles = torch.load('/media/johannes/WD Elements/NSCLC_Stefan/augmented_slices/z_rotations.pt')[:num_nodes]

    rot_matrices =[Rotation.from_euler('xyz', (x_angles[i], y_angles[i], z_angles[i]), degrees=True).as_matrix() for i in range(len(x_angles))]

    result_matrix = np.zeros((len(x_angles), len(x_angles)))

    for i in range(len(x_angles)):
        for j in range(len(x_angles)):
            result_matrix[i, j] = np.linalg.norm(rot_matrices[i] - rot_matrices[j], ord='fro')

    result_matrix = torch.tensor(result_matrix)

    # get edge connections according to rotation similarities
    edge_connections = {}
    for i in range(len(x_angles)):
        row = result_matrix[i, :]
        indices = torch.topk(row, k=k+1, largest=False)
        edge_connections[i] = indices[1][1:].tolist()

    # create adjacency matrix from edge connections
    g = edge_connections
    keys=sorted(g.keys())
    size=len(keys)

    M = [ [0]*size for i in range(size) ]

    for a,b in [(keys.index(a), keys.index(b)) for a, row in g.items() for b in row]:
        M[a][b] = 2 if (a==b) else 1

    M = np.array(M)

    # create edge index matrix
    edge_indices = np.argwhere(np.array(M) != 0)
    edge_indices = np.transpose(edge_indices).astype(int)

    data = Data(x=None, edge_index=torch.tensor(edge_indices), num_nodes=num_nodes)

    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)


    return data
