import shutil
import random
import os
import numpy as np
import torch

import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )

def f1_loss(y_true, y_pred):
    tp = torch.sum((y_true * y_pred).float(), dim=0)
    tn = torch.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)

def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        # print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

def set_seed(seed: int) -> None:
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

    def __init__(self, embed_f, k=None, distance='euclidean'):
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

def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)

    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x
