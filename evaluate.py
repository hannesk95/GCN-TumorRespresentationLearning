import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
from torch.autograd import Variable
import argparse
from tools.Trainer import ModelNetTrainer
from tools.Dataset import MultiviewImgDataset, SingleImgDataset, SingleImgDataset3D, MultiviewImgDataset3D
from model.models import view_GCN, SVCNN, GNN
import mlflow
from tools.utils import set_seed, create_folder
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle


# parser = argparse.ArgumentParser()
# parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="view-gcn")
# parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=20)# it will be *12 images in each batch for mvcnn
# parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
# parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
# parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
# parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
# parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18_3D")
# parser.add_argument("-num_views", type=int, help="number of views", default=20)
# parser.add_argument("-train_path", type=str, default="data/modelnet40v2png_ori4/*/train")
# parser.add_argument("-val_path", type=str, default="data/modelnet40v2png_ori4/*/test")
# parser.set_defaults(train=False)
# args = parser.parse_args()


def evaluate(model, loader):    
    
    val_test_true = []
    val_test_pred = []
    val_test_score = []

    model.eval()

    for data in tqdm(loader, desc=f"[EVALUATION]"):
        in_data = Variable(data[1]).cuda()            
        target = Variable(data[0]).cuda().long()

        out_data = model(in_data)
        pred = torch.max(out_data, dim=1).indices
        score = torch.softmax(out_data, dim=1)[:, 1]       
              
        val_test_true.extend(target.detach().cpu().numpy())
        val_test_pred.extend(pred.detach().cpu().numpy())
        val_test_score.extend(score.detach().cpu().numpy())          
       
    f1 = f1_score(val_test_true, val_test_pred, average="macro")
    mcc = matthews_corrcoef(val_test_true, val_test_pred)
    auc = roc_auc_score(val_test_true, val_test_score)
    bacc = balanced_accuracy_score(val_test_true, val_test_pred)

    # cf_matrix = confusion_matrix(val_test_true, val_test_pred)
    # fig = plt.figure()
    # sns.heatmap(cf_matrix, annot=True, cmap="Blues", cbar=False, fmt='g', 
    #             xticklabels=["HPV-negative", "HPV-positive"],
    #             yticklabels=["HPV-negative", "HPV-positive"])
    # plt.xlabel('pred class')
    # plt.ylabel('true class')
    # fig.savefig(f"eval_results/confusion_matrix.png")
    # plt.close(fig)

    return f1, mcc, auc, bacc, val_test_true, val_test_pred, val_test_score


if __name__ == '__main__': 

    set_seed(seed=28)
    model_path = '/home/johannes/Desktop/view-GCN/experiments/view-gcn_resnet18_3D_2024-02-20 22:29:38/model-200.pth'
    
    test_time_augmentation = False 
    test_time_aggregation = "mean"
    test_time_augmentations = 5
    dataset = "NSCLC"

    cnn = SVCNN("blabla", nclasses=2, cnn_name="resnet18_3D").cuda()
    cnn.load_state_dict(torch.load(model_path))
    pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"Number of Parameters: {pytorch_total_params}")
   
    train_dataset = SingleImgDataset3D(dataset=dataset, mode="train", cnn_name="resnet18_3D")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    val_dataset = SingleImgDataset3D(dataset=dataset, mode="val", cnn_name="resnet18_3D")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    test_dataset = SingleImgDataset3D(dataset=dataset, mode="test", cnn_name="resnet18_3D")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    print('num_test_files: '+str(len(test_dataset.filepaths)))

    if test_time_augmentation:

        for i in range(test_time_augmentations):
            with torch.no_grad():
                f1, mcc, auc, bacc, true, pred, score = evaluate(model=cnn, loader=test_loader)
    
    else:

        f1_list = []
        mcc_list = []
        auc_list = []
        bacc_list = []

        with torch.no_grad():
                for i in range(25):                   
                    
                    test_dataset = SingleImgDataset3D(dataset=dataset, mode="test", cnn_name="resnet18_3D")
                    indices = np.arange(start=0, stop=len(test_dataset), step=1)
                    
                    # bootstrap
                    indices, _ = shuffle(indices, indices, random_state=i)
                    indices = indices[:72]

                    # cross-validation
                    # split_length = int(np.floor(len(indices)/5))
                    # indices = indices[i*split_length:(i+1)*split_length]

                    test_subset = torch.utils.data.Subset(test_dataset, indices)
                    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)

                    f1, mcc, auc, bacc, true, pred, score = evaluate(model=cnn, loader=test_loader)

                    f1_list.append(f1)
                    mcc_list.append(mcc)
                    auc_list.append(auc)
                    bacc_list.append(bacc)

    print(f"F1-Score: {np.mean(f1_list)}")
    print(f"std:      {np.std(f1_list)}")
    print(f"MCC:      {np.mean(mcc_list)}")
    print(f"std:      {np.std(mcc_list)}")
    print(f"AUC:      {np.mean(auc_list)}")
    print(f"std:      {np.std(auc_list)}")
    print(f"bACC:     {np.mean(bacc_list)}")
    print(f"std:      {np.std(bacc_list)}")

    torch.save(true, "eval_results/true.pt")
    torch.save(pred, "eval_results/pred.pt")
    torch.save(score, "eval_results/score.pt")
    