import numpy as np
import torch
import torch.optim as optim
import argparse
from Trainer import ModelNetTrainer
from Dataset import SingleImgDataset3D, MultiviewImgDataset3D
from models.Model import CNN, GNN
import mlflow
from utils import create_folder, set_seed, FocalLoss
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset", type=str, help="Name of the dataset", default="NSCLC")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=16)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="ResNet18-3D")
parser.add_argument("-lr", type=float, help="learning rate", default=0.001)
parser.set_defaults(pretrained=True)
args = parser.parse_args()


if __name__ == '__main__':         
    
    ###############################################################
    # Train 3DResNet ##############################################
    ###############################################################

    set_seed(seed=28)
    mlflow.set_experiment(f"CNN-{args.dataset}")
    date = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    with mlflow.start_run(run_name=date):

        mlflow.log_params(args.__dict__)
        log_dir = 'experiments/'+f'{args.cnn_name}_{date}'
        create_folder(log_dir)
        confusion_matrix_dir = log_dir+"/ConfusionMatrices"
        create_folder(confusion_matrix_dir)        

        cnn = CNN(nclasses=2, pretraining=args.pretrained, cnn_name=args.cnn_name).cuda()
        pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
        mlflow.log_param("n_parameters", pytorch_total_params)
        print(f"Number of Parameters: {pytorch_total_params}")

        optimizer = optim.AdamW(cnn.parameters(), lr=args.lr)
        mlflow.log_param("optimizer", "AdamW")      
    
        train_dataset = SingleImgDataset3D(dataset=args.dataset, mode="train", cnn_name=args.cnn_name)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4)
        
        if args.dataset == "RADCURE":
            val_dataset0 = SingleImgDataset3D(dataset=args.dataset, mode="val", cnn_name=args.cnn_name, split=0)
            val_loader0 = torch.utils.data.DataLoader(val_dataset0, batch_size=args.batchSize, shuffle=False, num_workers=4)

            val_dataset1 = SingleImgDataset3D(dataset=args.dataset, mode="val", cnn_name=args.cnn_name, split=1)
            val_loader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=args.batchSize, shuffle=False, num_workers=4)

            val_dataset2 = SingleImgDataset3D(dataset=args.dataset, mode="val", cnn_name=args.cnn_name, split=2)
            val_loader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=args.batchSize, shuffle=False, num_workers=4)
        
        else:
            val_dataset = SingleImgDataset3D(dataset=args.dataset, mode="val", cnn_name=args.cnn_name, split=0)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
        
        test_dataset = SingleImgDataset3D(dataset=args.dataset, mode="test", cnn_name=args.cnn_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
        
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        if args.dataset == "RADCURE":
            print('num_val_files: '+str(len(val_dataset0.filepaths)))
            print('num_val_files: '+str(len(val_dataset1.filepaths)))
            print('num_val_files: '+str(len(val_dataset2.filepaths)))
        else:
            print('num_val_files: '+str(len(val_dataset.filepaths)))
        print('num_test_files: '+str(len(test_dataset.filepaths)))

        lr_scheduler = {"learning_rate": args.lr,
                        "warmup_start_value": args.lr / 100, 
                        "warmup_end_value": args.lr,
                        "warmup_period": 50, 
                        "discount_factor": 0.999,
                        "discount_mode": "exponential"}
        
        mlflow.log_params(lr_scheduler)

        # loss_fn = nn.CrossEntropyLoss(weight=train_dataset.class_weights)
        loss_fn = FocalLoss(weight=train_dataset.class_weights, reduction="mean")

        if args.dataset == "RADCURE":
            trainer = ModelNetTrainer(model=cnn, train_loader=train_loader, val_loader=[val_loader0, val_loader1, val_loader2], 
                                      test_loader=test_loader, optimizer=optimizer, loss_fn=loss_fn, model_name='cnn', 
                                      log_dir=log_dir, lr_scheduler=lr_scheduler)  
              
        elif args.dataset == "NSCLC":
            trainer = ModelNetTrainer(model=cnn, train_loader=train_loader, val_loader=[val_loader], 
                                      test_loader=[test_loader], optimizer=optimizer, loss_fn=loss_fn, model_name='cnn', 
                                      log_dir=log_dir, lr_scheduler=lr_scheduler) 
        
        trainer.train(100)

    ##############################################################
    # Load Best CNN Model ########################################
    ##############################################################    

    model_path = ""
    cnn = CNN(nclasses=2, pretraining=args.pretrained, cnn_name=args.cnn_name)
    cnn.load_state_dict(torch.load(model_path))

    ##############################################################
    # Train Graph Neural Network #################################
    ##############################################################       
        
    set_seed(seed=28)
    mlflow.set_experiment(f"GNN-{args.dataset}")
    date = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    with mlflow.start_run(run_name=date):        

        n_augmentations = 12
        batch_size = 8
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("n_augmentations", n_augmentations)

        log_dir = 'experiments/'+f'GNN_{date}'
        create_folder(log_dir)
        confusion_matrix_dir = log_dir+"/ConfusionMatrices"
        create_folder(confusion_matrix_dir)

        gnn = GNN(name="gnn", cnn=cnn, freeze_cnn=True, n_augmentations=n_augmentations).cuda()
        pytorch_total_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
        mlflow.log_param("n_parameters", pytorch_total_params)
        print(f"Number of Parameters: {pytorch_total_params}")

        optimizer = optim.AdamW(cnn.parameters(), lr=args.lr)
        mlflow.log_param("optimizer", "AdamW") 

        scales = np.random.uniform(0.95, 1.05, n_augmentations)
        degrees = np.random.uniform(-10, 10, n_augmentations)
        translations = np.random.uniform(-20, 20, n_augmentations)

        train_dataset = MultiviewImgDataset3D(mode="train", n_augmentations=n_augmentations, scales=scales, degrees=degrees, translations=translations)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        val_dataset = MultiviewImgDataset3D(mode="val", n_augmentations=n_augmentations, scales=scales, degrees=degrees, translations=translations)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

        test_dataset = MultiviewImgDataset3D(mode="test", n_augmentations=n_augmentations, scales=scales, degrees=degrees, translations=translations)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: '+str(len(val_dataset.filepaths)))
        print('num_val_files: '+str(len(test_dataset.filepaths)))

        # loss_fn = nn.CrossEntropyLoss(weight=train_dataset.class_weights)
        loss_fn = FocalLoss(weight=train_dataset.class_weights, reduction="mean")

        lr_scheduler = {"learning_rate": args.lr,
                        "warmup_start_value": args.lr / 100, 
                        "warmup_end_value": args.lr,
                        "warmup_period": 50, 
                        "discount_factor": 0.999,
                        "discount_mode": "exponential"}
        
        mlflow.log_params(lr_scheduler)

        trainer = ModelNetTrainer(gnn, train_loader, val_loader, test_loader, optimizer, loss_fn, 'gnn', log_dir, lr_scheduler=lr_scheduler)            
        trainer.train(100)        
