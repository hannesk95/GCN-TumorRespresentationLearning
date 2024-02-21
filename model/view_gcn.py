import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from .Model import Model
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.data import Data, Batch
# from torch_geometric.transforms import KNNGraph
import torch_cluster
# from .resnet import create_resnet


mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

        if self.use_resnet:            
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
            elif self.cnn_name ==  'resnet18_3D':

                if self.pretraining:
                    print("Load ResNet18 with KINECT400 weights!")
                    self.net = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
                
                else:
                    print("Load ResNet18 without weights!")
                    self.net = models.video.r3d_18()
                    self.net.apply(self.init_weights)

                self.net.stem = nn.Sequential(
                    nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True))
                self.net.stem.apply(self.init_weights)
                
                self.net.fc = nn.Linear(512, self.nclasses)        
                self.net.fc.apply(self.init_weights)   
            
            # elif self.cnn_name == "resnet50_3D":
            #     self.net = create_resnet(input_channel=1, model_depth=50, norm=nn.BatchNorm3d, model_num_class=self.nclasses)
            #     self.net.fc = nn.Linear(512, self.nclasses)

            # elif self.cnn_name == "resnet101_3D":
            #     self.net = create_resnet(input_channel=1, model_depth=101, norm=nn.BatchNorm3d, model_num_class=self.nclasses)
            #     self.net.fc = nn.Linear(512, self.nclasses)
            

            if not self.pretraining:
                self.net.apply(self.init_weights)


        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x.to(torch.float32))
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))
    
    def init_weights(self, m):                

        if isinstance(m, nn.Linear):
            print("Initialize Weights: Linear")
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.Conv3d):
            print("Initialize Weights: Conv3d")
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class view_GCN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='resnet18', num_views=20):
        super(view_GCN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
        
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            vertices = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                        [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                        [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                        [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
        
        self.vertices = torch.tensor(vertices).cuda()

        self.LocalGCN1 = LocalGCN(k=4,n_views=self.num_views)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views)

        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views//2)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views//2)
        
        self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views//4)
        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views//2)
        self.View_selector2 = View_selector(n_views=self.num_views//2, sampled_view=self.num_views//4)

        self.cls = nn.Sequential(
            nn.Linear(512*3,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, self.nclasses)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):

        views = self.num_views
        y = self.net_1(x)
        y = y.view((int(x.shape[0] / views), views, -1))
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1).to(torch.float32)

        y = self.LocalGCN1(y, vertices)
        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]

        z, F_score, vertices2 = self.View_selector1(y2, vertices, k=4)
        z = self.LocalGCN2(z,vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]

        w, F_score2, vertices3 = self.View_selector2(z2, vertices2, k=4)
        w = self.LocalGCN3(w,vertices3)
        pooled_view3 = torch.max(w, 1)[0]

        pooled_view = torch.cat((pooled_view1, pooled_view2, pooled_view3),1)
        pooled_view = self.cls(pooled_view)
        
        return pooled_view, F_score, F_score2
    

class GNN(Model):
    def __init__(self, name, cnn, freeze_cnn, n_augmentations):
        super(GNN, self).__init__(name)
        # torch.manual_seed(12345)

        self.n_augmentations = n_augmentations

        self.cnn_encoder = nn.Sequential(*list(cnn.net.children())[:-1])

        if freeze_cnn:
            for param in self.cnn_encoder.parameters():
                param.requires_grad = False        

        # GNN layers
        self.conv1 = SAGEConv(512, 512)
        self.conv2 = SAGEConv(512, 512)
        self.conv3 = SAGEConv(512, 512)
        self.cls = nn.Linear(512, 2)
        
    def forward(self, x):

        # 1. Get 3DResNet18 encodings
        encodings = []
        for i in range(self.n_augmentations):
            temp = x[:, i, :, :, :].unsqueeze(1)
            encodings.append(self.cnn_encoder(temp).squeeze().unsqueeze(1))        
        encodings = torch.concatenate(encodings, dim=1)

        # 2. Graph Module
        knn_graphs = []
        for i in range(encodings.shape[0]):
            temp = encodings[i]
            data = Data(pos=temp)
            edge_index = knn_graph(data.pos, k=3, loop=True, cosine=True)
            data.edge_index = edge_index
            knn_graphs.append(data)
        
        graphs = Batch().from_data_list(knn_graphs)
        x = graphs.pos
        edge_index = graphs.edge_index
        batch = graphs.batch        

        # encodings, adjacency_matrix, _ = DGM_c(encodings)

        # 3. GNN
        x = self.conv1(x, edge_index)
        x = nn.LeakyReLU(0.2, inplace=True)(x)      
        x = self.conv2(x, edge_index)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        # x = F.dropout(x, p=0.3, training=self.training)
        x = self.cls(x)

        return x


class DGM_c(nn.Module):
    input_dim = 4
    debug=False
    
    def __init__(self, embed_f, k=None, distance="euclidean"):
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
        
        if self.distance=="hyperbolic":
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
    
#Euclidean distance
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