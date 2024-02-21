import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset, SingleImgDataset3D, MultiviewImgDataset3D
from model.view_gcn import view_GCN, SVCNN, GNN
import mlflow
from tools.utils import create_folder, set_seed, FocalLoss
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="view-gcn")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=16)
# parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=0.001)
# parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
# parser.add_argument("-pretraining", dest='pretraining', action='store_false')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18_3D")
# parser.add_argument("-num_views", type=int, help="number of views", default=20)
# parser.add_argument("-train_path", type=str, default="data/modelnet40v2png_ori4/*/train")
# parser.add_argument("-val_path", type=str, default="data/modelnet40v2png_ori4/*/test")
# parser.set_defaults(train=False)
parser.set_defaults(pretrained=True)
args = parser.parse_args()


if __name__ == '__main__':    
    
    # seed_torch()    
    set_seed(seed=28)
    # pretraining = args.no_pretraining
    # log_dir = args.name
    # create_folder(args.name)
    # config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    # json.dump(vars(args), config_f)
    # config_f.close()
    
    ###############################################################
    # Train 3DResNet ##############################################
    ###############################################################

    dataset = "NSCLC"

    mlflow.set_experiment(f"CNN-{dataset}")
    date = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # experiment = mlflow.get_experiment_by_name("3DResNet18")
    # with mlflow.start_run(experiment_id=experiment, run_name=date):
    with mlflow.start_run(run_name=date):

        mlflow.log_params(args.__dict__)
        log_dir = 'experiments/'+args.name+f'_{args.cnn_name}_{date}'
        create_folder(log_dir)
        confusion_matrix_dir = log_dir+"/ConfusionMatrices"
        create_folder(confusion_matrix_dir)        

        cnet = SVCNN(args.name, nclasses=2, pretraining=args.pretrained, cnn_name=args.cnn_name)
        pytorch_total_params = sum(p.numel() for p in cnet.parameters() if p.requires_grad)
        mlflow.log_param("n_parameters", pytorch_total_params)
        print(f"Number of Parameters: {pytorch_total_params}")

        # optimizer = optim.SGD(cnet.parameters(), lr=1e-2, weight_decay=args.weight_decay, momentum=0.9)
        # mlflow.log_param("optimizer", "SGD")
        optimizer = optim.AdamW(cnet.parameters(), lr=args.lr)
        mlflow.log_param("optimizer", "AdamW")
        # n_models_train = args.num_models*args.num_views        
    
        # train_dataset = SingleImgDataset3D(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
        train_dataset = SingleImgDataset3D(dataset=dataset, mode="train", cnn_name=args.cnn_name)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4)
        
        if dataset == "RADCURE":
            # val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
            val_dataset0 = SingleImgDataset3D(dataset=dataset, mode="val", cnn_name=args.cnn_name, split=0)
            val_loader0 = torch.utils.data.DataLoader(val_dataset0, batch_size=args.batchSize, shuffle=False, num_workers=4)

            val_dataset1 = SingleImgDataset3D(dataset=dataset, mode="val", cnn_name=args.cnn_name, split=1)
            val_loader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=args.batchSize, shuffle=False, num_workers=4)

            val_dataset2 = SingleImgDataset3D(dataset=dataset, mode="val", cnn_name=args.cnn_name, split=2)
            val_loader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=args.batchSize, shuffle=False, num_workers=4)
        
        else:
            val_dataset = SingleImgDataset3D(dataset=dataset, mode="val", cnn_name=args.cnn_name, split=0)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
        
        test_dataset = SingleImgDataset3D(dataset=dataset, mode="test", cnn_name=args.cnn_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
        
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        if dataset == "RADCURE":
            print('num_val_files: '+str(len(val_dataset0.filepaths)))
            print('num_val_files: '+str(len(val_dataset1.filepaths)))
            print('num_val_files: '+str(len(val_dataset2.filepaths)))
        else:
            print('num_val_files: '+str(len(val_dataset.filepaths)))

        print('num_test_files: '+str(len(test_dataset.filepaths)))

        # lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
        # warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=5 * int(len(train_dataset)/args.batchSize))

        lr_scheduler = {"learning_rate": args.lr,
                        "warmup_start_value": args.lr / 100, 
                        "warmup_end_value": args.lr,
                        "warmup_period": 50, 
                        "discount_factor": 0.999}
        
        mlflow.log_params(lr_scheduler)

        # loss_fn = nn.CrossEntropyLoss(weight=train_dataset.class_weights)
        loss_fn = FocalLoss(weight=train_dataset.class_weights, reduction="mean")

        if dataset == "RADCURE":
            trainer = ModelNetTrainer(model=cnet, train_loader=train_loader, val_loader=[val_loader0, val_loader1, val_loader2], 
                                        test_loader=test_loader, optimizer=optimizer, loss_fn=loss_fn, model_name='svcnn', 
                                        log_dir=log_dir, num_views=1, lr_scheduler=lr_scheduler)    
        else:
            trainer = ModelNetTrainer(model=cnet, train_loader=train_loader, val_loader=[val_loader], 
                                        test_loader=test_loader, optimizer=optimizer, loss_fn=loss_fn, model_name='svcnn', 
                                        log_dir=log_dir, num_views=1, lr_scheduler=lr_scheduler) 
        
        trainer.train(200)

    ##############################################################
    # Train Graph Neural Network #################################
    ##############################################################
        
    mlflow.set_experiment("GNN")
    date = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #experiment = mlflow.get_experiment_by_name("3DResNet18")
    with mlflow.start_run(run_name=date):

        cnn_3DResNet18 = SVCNN(args.name, nclasses=2, pretraining=args.pretrained, cnn_name=args.cnn_name)
        cnn_3DResNet18.load_state_dict(torch.load("/home/johannes/Desktop/view-GCN/view-gcn_resnet18_3D_2024-02-19 17:25:46/model-085.pth"))

        n_augmentations = 16
        batch_size = 16

        log_dir = args.name+'_GNN'
        create_folder(log_dir)
        # confusion_matrix_dir = log_dir+"/ConfusionMatrices"
        # create_folder(confusion_matrix_dir)
        # # cnet_2 = view_GCN(args.name, cnn, nclasses=2, cnn_name=args.cnn_name, num_views=args.num_views)
        gnn = GNN(name="gnn", cnn=cnn_3DResNet18, freeze_cnn=True, n_augmentations=n_augmentations)
        
        optimizer = optim.AdamW(gnn.parameters(), lr=0.001)
        # # optimizer = optim.SGD(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
        # # train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, 
        # #                                     num_models=n_models_train, num_views=args.num_views, test_mode=True)

        scales = np.random.uniform(0.95, 1.05, n_augmentations)
        degrees = np.random.uniform(-10, 10, n_augmentations)
        translations = np.random.uniform(-20, 20, n_augmentations)

        train_dataset = MultiviewImgDataset3D(mode="train", n_augmentations=n_augmentations, scales=scales, degrees=degrees, translations=translations)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # # val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, 
        # #                                   num_views=args.num_views, test_mode=True)
        val_dataset = MultiviewImgDataset3D(mode="val", n_augmentations=n_augmentations, scales=scales, degrees=degrees, translations=translations)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

        test_dataset = MultiviewImgDataset3D(mode="test", n_augmentations=n_augmentations, scales=scales, degrees=degrees, translations=translations)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: '+str(len(val_dataset.filepaths)))
        print('num_val_files: '+str(len(test_dataset.filepaths)))

        # loss_fn = nn.CrossEntropyLoss(weight=train_dataset.class_weights)
        loss_fn = FocalLoss(weight=train_dataset.class_weights, reduction="mean")

        trainer = ModelNetTrainer(gnn, train_loader, val_loader, test_loader, optimizer, loss_fn, 'gnn', log_dir, num_views=20)
        
        # mlflow.set_experiment("GNN")
        # experiment = mlflow.get_experiment_by_name("GNN")
        # with mlflow.start_run(experiment_id=experiment.experiment_id):
        trainer.train(100)

        # # #use trained_view_gcn
        # # #cnet_2.load_state_dict(torch.load('trained_view_gcn.pth'))
        # # #trainer.update_validation_accuracy(1)
