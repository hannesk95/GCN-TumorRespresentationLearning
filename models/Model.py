import torch
import torch.nn as nn
import torchvision.models as models
# from .ResNet import create_resnet
from torch_geometric.nn.conv import SAGEConv, GATConv
from torch_geometric.nn import MLP, Sequential
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, knn_graph, global_max_pool
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.transforms import NormalizeScale


class CNN(nn.Module):
    def __init__(self, nclasses=2, pretraining=True, cnn_name='resnet18'):
        super(CNN, self).__init__()
        
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
                   
        if self.cnn_name == 'ResNet18':
            self.net = models.resnet18(pretrained=self.pretraining)
            self.net.fc = nn.Linear(512, self.nclasses)
        elif self.cnn_name == 'ResNet34':
            self.net = models.resnet34(pretrained=self.pretraining)
            self.net.fc = nn.Linear(512, self.nclasses)
        elif self.cnn_name == 'ResNet50':
            self.net = models.resnet50(pretrained=self.pretraining)
            self.net.fc = nn.Linear(2048, self.nclasses)
        elif self.cnn_name == 'ResNet18-3D':
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
            
        elif self.cnn_name == "ResNet50-3D":
            # self.net = create_resnet(input_channel=1, model_depth=50, norm=nn.BatchNorm3d, model_num_class=self.nclasses)
            self.net.fc = nn.Linear(512, self.nclasses)

        elif self.cnn_name == "ResNet101-3D":
            # self.net = create_resnet(input_channel=1, model_depth=101, norm=nn.BatchNorm3d, model_num_class=self.nclasses)
            self.net.fc = nn.Linear(512, self.nclasses)            

        if not self.pretraining:
            self.net.apply(self.init_weights)

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

    def forward(self, x):
        return self.net(x.to(torch.float32))
        

class GNN(nn.Module):
    def __init__(self, mode, cnn, freeze_cnn, n_augmentations, n_neighbors):
        super(GNN, self).__init__()
        
        self.mode = mode

        self.n_neighbors = n_neighbors
        self.n_augmentations = n_augmentations

        self.cnn_encoder = nn.Sequential(*list(cnn.net.children())[:-1])

        if freeze_cnn:
            for param in self.cnn_encoder.parameters():
                param.requires_grad = False        

        # GNN layers
        # self.conv1 = SAGEConv(512, 128)
        # self.conv2 = SAGEConv(512, 512)
        # self.conv3 = SAGEConv(512, 512)
                
        in_channels = 512
        hidden_channels = 512
        out_channels = 512
        num_layers = 3

        self.gnn = Sequential('x, edge_index', [
            (GATConv(in_channels, hidden_channels), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(hidden_channels, hidden_channels), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(hidden_channels, out_channels), 'x, edge_index -> x')
        ])
        
        # MLP layers
        self.mlp = MLP(in_channels=512, hidden_channels=512, out_channels=512, num_layers=num_layers)

        # Classification head
        self.cls = nn.Linear(out_channels, 2)


        
    def forward(self, x):

        # (old) 1. Get 3DResNet18 encodings
        # encodings = []
        # for i in range(self.n_augmentations):
        #     temp = x[:, i, :, :, :].unsqueeze(1)
        #     encodings.append(self.cnn_encoder(temp).squeeze().unsqueeze(1))        
        # encodings = torch.concatenate(encodings, dim=1)

        encodings = x

        # 2. Graph Module
        knn_graphs = []
        for i in range(encodings.shape[0]):
            temp = encodings[i]
            data = Data(pos=temp)
            edge_index = knn_graph(data.pos, k=self.n_neighbors, cosine=False)
            data.edge_index = edge_index
            g = torch_geometric.utils.to_networkx(data, to_undirected=True)
            fig = plt.figure()
            nx.draw(g)
            fig.savefig(f"./graph.png")
            plt.close(fig)
            
            data = NormalizeScale()(data=data)

            knn_graphs.append(data)
        
        graphs = Batch().from_data_list(knn_graphs)
        x = graphs.pos
        edge_index = graphs.edge_index
        batch = graphs.batch        

        # encodings, adjacency_matrix, _ = DGM_c(encodings)

        # 3. GNN
        # x = self.conv1(x, edge_index)
        # x = nn.ReLU()(x)
        # x = nn.LeakyReLU(0.2, inplace=True)(x)      
        # x = self.conv2(x, edge_index)
        # x = nn.LeakyReLU(0.2, inplace=True)(x)
        # x = self.conv3(x, edge_index)

        if self.mode == "MLP":
            x = self.mlp(x, edge_index)
        
        elif self.mode == "GNN":
            x = self.gnn(x, edge_index)

        x = global_max_pool(x, batch)
        # x = F.dropout(x, p=0.3, training=self.training)
        x = self.cls(x)

        return x
    