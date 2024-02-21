import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math
import os
import mlflow
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


# class F1_Loss:
#     def __init__(self) -> None:
#         print("F1_Loss")

#     def forward(self, y_true, y_pred):
#         tp = torch.sum((y_true * y_pred).float(), dim=0)
#         tn = torch.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
#         fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
#         fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)

#         p = tp / (tp + fp + 1e-7)
#         r = tp / (tp + fn + 1e-7)

#         f1 = 2 * p * r / (p + r + 1e-7)
#         f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
#         return 1 - torch.mean(f1)  

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


class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12, lr_scheduler=None, warmup_scheduler=None):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        # self.loss_f1 = F1_Loss
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.model.cuda()
        self.lr_scheduler = lr_scheduler
        # self.warmup_scheduler = warmup_scheduler
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)
    
    def train(self, n_epochs):
        # best_acc = 0
        # i_acc = 0        

        self.model.train()
        for epoch in range(1, n_epochs+1):

            train_loss = []
            # train_acc = []     
            # train_f1 = []  
            # train_bacc = [] 
            # train_mcc = []
            train_true = []
            train_pred = []
            train_score = []    
            

            # if self.model_name == 'view-gcn':
            #     if epoch == 1:
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] = lr
            #     if epoch > 1:
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] = param_group['lr'] * 0.5 * ( 1 + math.cos(epoch * math.pi / 15))
            
            # UNCOMMENT THE FOLLOWING FOR INITIAL LEARNING RATE SCHEDULER
            # else:
            #     if epoch > 0 and (epoch + 1) % 10 == 0:
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] = param_group['lr'] * 0.5

            if self.model_name != "gnn":
                if epoch == 1:
                    self.optimizer.param_groups[0]['lr'] = self.lr_scheduler["warmup_start_value"]     

                
                elif epoch <= self.lr_scheduler["warmup_period"]:
                    self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] + ((self.lr_scheduler["warmup_end_value"]-self.lr_scheduler["warmup_start_value"])/self.lr_scheduler["warmup_period"])

                else:
                    # Exponential
                    self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * (self.lr_scheduler["discount_factor"]**(epoch-self.lr_scheduler["warmup_period"]))
                    # Linear
                    # self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] - ((self.lr_scheduler["warmup_end_value"]-self.lr_scheduler["warmup_start_value"]*0.1)/(n_epochs - self.lr_scheduler["warmup_period"]))

            # permute data for mvcnn
            # rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            # filepaths_new = []
            # for i in range(len(rand_idx)):
            #     filepaths_new.extend(self.train_loader.dataset.filepaths[
            #                          rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            # self.train_loader.dataset.filepaths = filepaths_new
            
            # plot learning rate
            # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            lr = self.optimizer.param_groups[0]['lr']
            # self.writer.add_scalar('params/lr', lr, epoch)
            mlflow.log_metric("learning_rate", lr, step=epoch)
            
            # train one epoch
            out_data = None
            in_data = None
            # iteration = 0
            for data in tqdm(self.train_loader, desc=f"Training: Epoch {epoch} | {n_epochs}"):
                # if self.model_name == 'view-gcn' and epoch == 0:
                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] = lr * ((i + 1) / (len(rand_idx) // 20))
                # if self.model_name == 'view-gcn':
                #     N, V, C, H, W = data[1].size()
                #     in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                # else:
                in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()
                # target_ = target.unsqueeze(1).repeat(1, 4*(10+5)).view(-1)
                
                self.optimizer.zero_grad()
                # if self.model_name == 'view-gcn':
                #     out_data, F_score,F_score2= self.model(in_data)
                #     out_data_ = torch.cat((F_score, F_score2), 1).view(-1, 40)
                #     loss = self.loss_fn(out_data, target)+ self.loss_fn(out_data_, target_)
                # else:
                out_data = self.model(in_data)
                pred = torch.max(out_data, 1)[1]
                score = torch.softmax(out_data, dim=1)[:, 1]# .cpu().detach()

                f1_loss_value = f1_loss(target, score)
                loss = self.loss_fn(out_data, target) + f1_loss_value
                # self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)
                # temp = loss.detach().cpu().numpy()
                train_loss.append(loss.detach().cpu().numpy())

                
                
                # results = pred == target
                # correct_points = torch.sum(results.long())

                train_true.extend(target.detach().cpu().numpy())
                train_pred.extend(pred.detach().cpu().numpy())
                train_score.extend(score.detach().cpu().numpy())

                # train_f1.append(f1_score(target.cpu(), pred.cpu(), average='macro'))
                # train_bacc.append(balanced_accuracy_score(target.cpu(), pred.cpu()))
                # train_mcc.append(matthews_corrcoef(target.cpu(), pred.cpu()))

                # acc = correct_points.float() / results.size()[0]
                # temp = float(acc.detach().cpu().numpy())
                # train_acc.append(temp)                
                # self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                #print('lr = ', str(param_group['lr']))
                loss.backward()
                self.optimizer.step()

                # with self.warmup_scheduler.dampening():
                #     self.lr_scheduler.step()
                
                # iteration += 1
                
                

                # with self.warmup_scheduler.dampening():
                #     if self.warmup_scheduler.last_step + 1 >= self.warmup_scheduler.warmup_params[0]["warmup_period"]:
                #         pass
                        # self.lr_scheduler.step()

                # log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch, i + 1, loss, acc)
                # if (i + 1) % 1 == 0:
                    # print(log_str)

            # i_acc += i
            # with self.warmup_scheduler.dampening():
            #     self.lr_scheduler.step()
                
            # if epoch <= self.lr_scheduler["warmup_period"]:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] + ((self.lr_scheduler["warmup_end_value"]-self.lr_scheduler["warmup_start_value"])/self.lr_scheduler["warmup_period"])

            # else:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] * self.lr_scheduler["discount_factor"]**(epoch-self.lr_scheduler["warmup_period"])


            train_loss = np.mean(train_loss)
            # train_acc = np.mean(train_acc)
            # train_f1 = np.mean(train_f1)
            # train_bacc = np.mean(train_bacc)
            # train_mcc = np.mean(train_mcc)
            train_f1 = f1_score(train_true, train_pred, average="macro")
            train_auc = roc_auc_score(train_true, train_score)
            train_mcc = matthews_corrcoef(train_true, train_pred)
            train_bacc = balanced_accuracy_score(train_true, train_pred)           

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            # mlflow.log_metric("train_acc", train_acc, step=epoch)
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
            
            # evaluation
            # if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                    # loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch, mode="val")
                self.update_validation_accuracy(epoch, mode="val")
                # self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                # self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                # self.writer.add_scalar('val/val_loss', loss, epoch + 1)
            
            path = f"{self.log_dir}/model-{str(epoch).zfill(3)}.pth"
            torch.save(self.model.state_dict(), path)
            
            # self.model.save(self.log_dir, epoch)
                
                # save best model
                # if val_overall_acc > best_acc:
                #     best_acc = val_overall_acc
                # print('best_acc', best_acc)
        # export scalar data to JSON for external processing
        # self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        # self.writer.close()
                
        # Test
        with torch.no_grad():
            self.update_test_accuracy(epoch, mode="test")
            

    def update_validation_accuracy(self, epoch, mode):
        # all_correct_points = 0
        # all_points = 0
        # count = 0
        # wrong_class = np.zeros(40)
        # samples_class = np.zeros(40)
        # all_loss = 0
        
        val_test_loss = []
        # val_test_bacc = []
        # val_test_f1 = []
        val_test_true = []
        val_test_pred = []
        val_test_score = []

        self.model.eval()

        if mode == "val":
            loader = self.val_loader
        elif mode == "test":
            loader = self.test_loader

        for loader in self.val_loader:

            for data in tqdm(loader, desc=f"[EVALUATION] {mode}"):

                # if self.model_name == 'view-gcn':
                #     N, V, C, H, W = data[1].size()
                #     in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                # else:  # 'svcnn'

                in_data = Variable(data[1]).cuda()            
                target = Variable(data[0]).cuda().long()

                # if self.model_name == 'view-gcn':
                #     out_data,F1,F2=self.model(in_data)
                # else:

                out_data = self.model(in_data)
                pred = torch.max(out_data, 1)[1]
                score = torch.softmax(out_data, dim=1)[:, 1]#.cpu().detach()            
                
                f1_loss_value = f1_loss(target, score).cpu().numpy()
                loss = self.loss_fn(out_data, target).cpu().numpy() + f1_loss_value

                # loss = self.loss_fn(out_data, target).cpu().numpy() + self.loss_f1(out_data, target).cpu().numpy()
                val_test_loss.append(loss)
                # results = pred == target            
                
                val_test_true.extend(target.detach().cpu().numpy())
                val_test_pred.extend(pred.detach().cpu().numpy())
                val_test_score.extend(score.detach().cpu().numpy())

                
                # val_test_bacc.append(balanced_accuracy_score(target.cpu(), pred.cpu()))
                # val_test_f1.append(f1_score(target.cpu(), pred.cpu(), average='macro'))
                

            #     for i in range(results.size()[0]):
            #         if not bool(results[i].cpu().data.numpy()):
            #             wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
            #         samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            #     correct_points = torch.sum(results.long())

            #     all_correct_points += correct_points
            #     all_points += results.size()[0]

            # print('Total # of test models: ', all_points)
            # class_acc = (samples_class - wrong_class) / samples_class
            # val_mean_class_acc = np.mean(class_acc)
            # acc = all_correct_points.float() / all_points
            # val_overall_acc = acc.cpu().data.numpy()
            # loss = all_loss / len(loader)
            
            loss = np.mean(val_test_loss)       
            f1 = f1_score(val_test_true, val_test_pred, average="macro")
            mcc = matthews_corrcoef(val_test_true, val_test_pred)
            auc = roc_auc_score(val_test_true, val_test_score)
            bacc = balanced_accuracy_score(val_test_true, val_test_pred)

            cf_matrix = confusion_matrix(val_test_true, val_test_pred)
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
                split = str(loader.dataset.split)
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


            # print('val mean class acc. : ', val_mean_class_acc)
            # print('val overall acc. : ', val_overall_acc)
            # print('val loss : ', loss)
            # print(class_acc)
            self.model.train()

            # return loss, val_overall_acc, val_mean_class_acc
    
    def update_test_accuracy(self, epoch, mode):
        # all_correct_points = 0
        # all_points = 0
        # count = 0
        # wrong_class = np.zeros(40)
        # samples_class = np.zeros(40)
        # all_loss = 0
        
        val_test_loss = []
        # val_test_bacc = []
        # val_test_f1 = []
        val_test_true = []
        val_test_pred = []
        val_test_score = []

        self.model.eval()

        if mode == "val":
            loader = self.val_loader
        elif mode == "test":
            loader = self.test_loader

        for data in tqdm(loader, desc=f"[EVALUATION] {mode}"):

            # if self.model_name == 'view-gcn':
            #     N, V, C, H, W = data[1].size()
            #     in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            # else:  # 'svcnn'

            in_data = Variable(data[1]).cuda()            
            target = Variable(data[0]).cuda().long()

            # if self.model_name == 'view-gcn':
            #     out_data,F1,F2=self.model(in_data)
            # else:

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            score = torch.softmax(out_data, dim=1)[:, 1]#.cpu().detach()            
            
            f1_loss_value = f1_loss(target, score).cpu().numpy()
            loss = self.loss_fn(out_data, target).cpu().numpy() + f1_loss_value

            # loss = self.loss_fn(out_data, target).cpu().numpy() + self.loss_f1(out_data, target).cpu().numpy()
            val_test_loss.append(loss)
            # results = pred == target            
            
            val_test_true.extend(target.detach().cpu().numpy())
            val_test_pred.extend(pred.detach().cpu().numpy())
            val_test_score.extend(score.detach().cpu().numpy())

            
            # val_test_bacc.append(balanced_accuracy_score(target.cpu(), pred.cpu()))
            # val_test_f1.append(f1_score(target.cpu(), pred.cpu(), average='macro'))
            

        #     for i in range(results.size()[0]):
        #         if not bool(results[i].cpu().data.numpy()):
        #             wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
        #         samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
        #     correct_points = torch.sum(results.long())

        #     all_correct_points += correct_points
        #     all_points += results.size()[0]

        # print('Total # of test models: ', all_points)
        # class_acc = (samples_class - wrong_class) / samples_class
        # val_mean_class_acc = np.mean(class_acc)
        # acc = all_correct_points.float() / all_points
        # val_overall_acc = acc.cpu().data.numpy()
        # loss = all_loss / len(loader)
        
        loss = np.mean(val_test_loss)       
        f1 = f1_score(val_test_true, val_test_pred, average="macro")
        mcc = matthews_corrcoef(val_test_true, val_test_pred)
        auc = roc_auc_score(val_test_true, val_test_score)
        bacc = balanced_accuracy_score(val_test_true, val_test_pred)

        cf_matrix = confusion_matrix(val_test_true, val_test_pred)
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
            split = str(loader.dataset.split)
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


        # print('val mean class acc. : ', val_mean_class_acc)
        # print('val overall acc. : ', val_overall_acc)
        # print('val loss : ', loss)
        # print(class_acc)
        self.model.train()

        # return loss, val_overall_acc, val_mean_class_acc
