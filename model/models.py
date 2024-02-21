import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.data import Data, Batch
from .resnet import create_resnet
import glob

class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name

    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(),
                   os.path.join(complete_path,
                                "model-{}.pth".format(str(epoch).zfill(5))))

    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path + "/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf))


class CNN(Model):
    def __init__(self, name, nclasses=2, pretraining=True, cnn_name='resnet18'):
        super(CNN, self).__init__(name)

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
            self.net = create_resnet(input_channel=1, model_depth=50, norm=nn.BatchNorm3d, model_num_class=self.nclasses)
            self.net.fc = nn.Linear(512, self.nclasses)

        elif self.cnn_name == "ResNet101-3D":
            self.net = create_resnet(input_channel=1, model_depth=101, norm=nn.BatchNorm3d, model_num_class=self.nclasses)
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
        

class GNN(Model):
    def __init__(self, name, cnn, freeze_cnn, n_augmentations):
        super(GNN, self).__init__(name)

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
    