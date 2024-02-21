import torch
import mlflow
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from tools.utils import f1_loss
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score


class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, loss_fn, model_name, log_dir, lr_scheduler):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.lr_scheduler = lr_scheduler
    
    def train(self, n_epochs):
        
        for epoch in range(1, n_epochs+1):
            self.model.train()

            train_loss = []
            train_true = []
            train_pred = []
            train_score = []  
            
            if epoch == 1:
                self.optimizer.param_groups[0]['lr'] = self.lr_scheduler["warmup_start_value"]

            elif epoch <= self.lr_scheduler["warmup_period"]:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] + ((self.lr_scheduler["warmup_end_value"]-self.lr_scheduler["warmup_start_value"])/self.lr_scheduler["warmup_period"])

            else:
                if self.lr_scheduler["discount_mode"] == "exponential":    
                    self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * (self.lr_scheduler["discount_factor"]**(epoch-self.lr_scheduler["warmup_period"]))
                elif self.lr_scheduler["discount_mode"] == "linear":
                    self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] - ((self.lr_scheduler["warmup_end_value"]-self.lr_scheduler["warmup_start_value"]*0.1)/(n_epochs - self.lr_scheduler["warmup_period"]))
           
            lr = self.optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", lr, step=epoch)
            
            # train one epoch
            for data in tqdm(self.train_loader, desc=f"Training: Epoch {epoch} | {n_epochs}"):
                
                in_data = data[1].cuda().to(torch.float32)
                target = data[0].cuda().to(torch.long)              
                self.optimizer.zero_grad()
      
                out_data = self.model(in_data)
                pred = torch.max(out_data, 1)[1]
                score = torch.softmax(out_data, dim=1)[:, 1]

                f1_loss_value = f1_loss(target, score)
                loss = self.loss_fn(out_data, target) + f1_loss_value
                loss.backward()
                self.optimizer.step()
                
                train_loss.append(loss.detach().cpu().numpy().item())
                train_true.extend(target.detach().cpu().numpy())
                train_pred.extend(pred.detach().cpu().numpy())
                train_score.extend(score.detach().cpu().numpy())

            train_loss = np.mean(train_loss)
            train_f1 = f1_score(train_true, train_pred, average="macro")
            train_auc = roc_auc_score(train_true, train_score)
            train_mcc = matthews_corrcoef(train_true, train_pred)
            train_bacc = balanced_accuracy_score(train_true, train_pred)           

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("train_bacc", train_bacc, step=epoch)
            mlflow.log_metric("train_mcc", train_mcc, step=epoch)
            mlflow.log_metric("train_auc", train_auc, step=epoch)

            cf_matrix = confusion_matrix(train_true, train_pred)
            fig = plt.figure()
            sns.heatmap(cf_matrix, annot=True, cmap="Blues", cbar=False, fmt='g', 
                        xticklabels=["HPV-negative", "HPV-positive"],
                        yticklabels=["HPV-negative", "HPV-positive"])
            plt.xlabel('pred class')
            plt.ylabel('true class')
            fig.savefig(f"{self.log_dir}/ConfusionMatrices/cm_epoch{epoch}_train.png")
            plt.close(fig)     

            ####################################
            # Validation #######################
            ####################################       
           
            with torch.no_grad():                  
                self.evaluate(epoch, mode="val")   
            
            path = f"{self.log_dir}/model-{str(epoch).zfill(3)}.pth"
            torch.save(self.model.state_dict(), path)            

        #####################################  
        # Test ##############################
        #####################################
            
        with torch.no_grad():
            self.evaluate(epoch, mode="test")
            

    def evaluate(self, epoch, mode):

        self.model.eval()
        
        loss_list = []
        true_list = []
        pred_list = []
        score_list = []        

        if mode == "val":
            loader = self.val_loader
        elif mode == "test":
            loader = self.test_loader

        for loader_instance in loader:

            for data in tqdm(loader_instance, desc=f"[EVALUATION] {mode}"):              

                in_data = data[1].cuda().to(torch.float32)            
                target = data[0].cuda().to(torch.long)              

                out_data = self.model(in_data)

                pred = torch.max(out_data, 1)[1]
                score = torch.softmax(out_data, dim=1)[:, 1]         
                
                f1_loss_value = f1_loss(target, score).cpu().numpy().item()
                loss = self.loss_fn(out_data, target).cpu().numpy().item() + f1_loss_value

                loss_list.append(loss)               
                true_list.extend(target.detach().cpu().numpy())
                pred_list.extend(pred.detach().cpu().numpy())
                score_list.extend(score.detach().cpu().numpy())                        
            
            loss = np.mean(loss_list)       
            f1 = f1_score(true_list, pred_list, average="macro")
            mcc = matthews_corrcoef(true_list, pred_list)
            auc = roc_auc_score(true_list, score_list)
            bacc = balanced_accuracy_score(true_list, pred_list)

            cf_matrix = confusion_matrix(true_list, pred_list)
            fig = plt.figure()
            sns.heatmap(cf_matrix, annot=True, cmap="Blues", cbar=False, fmt='g', 
                        xticklabels=["HPV-negative", "HPV-positive"],
                        yticklabels=["HPV-negative", "HPV-positive"])
            plt.xlabel('pred class')
            plt.ylabel('true class')
            fig.savefig(f"{self.log_dir}/ConfusionMatrices/cm_epoch{epoch}_{mode}.png")
            plt.close(fig)

            mlflow.log_artifact(f"{self.log_dir}/ConfusionMatrices")

            if mode == "val":
                split = str(loader_instance.dataset.split)
                mlflow.log_metric(f"val_loss_{split}", loss, step=epoch)
                mlflow.log_metric(f"val_bacc_{split}", bacc, step=epoch)
                mlflow.log_metric(f"val_f1_{split}", f1, step=epoch)
                mlflow.log_metric(f"val_mcc_{split}", mcc, step=epoch)
                mlflow.log_metric(f"val_auc_{split}", auc, step=epoch)
            
            elif mode == "test":
                mlflow.log_metric("test_loss", loss, step=epoch)
                mlflow.log_metric("test_bacc", bacc, step=epoch)
                mlflow.log_metric("test_f1", f1, step=epoch)
                mlflow.log_metric("test_mcc", mcc, step=epoch)
                mlflow.log_metric("test_auc", auc, step=epoch)
                