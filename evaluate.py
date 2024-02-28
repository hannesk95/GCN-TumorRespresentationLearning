import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from Dataset import SingleImgDataset3D
from models.Model import CNN
from utils import set_seed
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from typing import Any, List, Iterable, Callable


def evaluate(model: Callable[[CNN]] , loader: Iterable[torch.utils.data.Dataloader]) -> List[Any]:
    """_summary_

    Args:
        model (Callable[[CNN]]): _description_
        loader (Iterable[torch.utils.data.Dataloader]): _description_

    Returns:
        List[Any]: _description_
    """


    encoder = nn.Sequential(*list(model.net.children())[:-1])

    model.eval()
    encoder.eval()
    val_test_true = []
    val_test_pred = []
    val_test_score = []
    probabilities = []
    features = []

    for data in tqdm(loader, desc=f'[EVALUATION]'):
        in_data = data[1].cuda().to(torch.float32)
        target = data[0].cuda().to(torch.long)

        out_data = model(in_data)
        feature = encoder(in_data).squeeze()
        pred = torch.max(out_data, dim=1)[1]
        score = torch.softmax(out_data, dim=1)[:, 1]

        val_test_true.extend(target.detach().cpu().numpy())
        val_test_pred.extend(pred.detach().cpu().numpy())
        val_test_score.extend(score.detach().cpu().numpy())
        probabilities.extend(torch.softmax(out_data, dim=1).detach().cpu().numpy())
        features.append(feature.detach().cpu())

    f1 = f1_score(val_test_true, val_test_pred, average='macro')
    mcc = matthews_corrcoef(val_test_true, val_test_pred)
    auc = roc_auc_score(val_test_true, val_test_score)
    bacc = balanced_accuracy_score(val_test_true, val_test_pred)

    cf_matrix = confusion_matrix(val_test_true, val_test_pred)
    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', cbar=False, fmt='g',
                xticklabels=['HPV-negative', 'HPV-positive'],
                yticklabels=['HPV-negative', 'HPV-positive'])
    plt.xlabel('pred class')
    plt.ylabel('true class')
    fig.savefig(f'eval_results/confusion_matrix.png')
    plt.close(fig)

    return f1, mcc, auc, bacc, val_test_true, val_test_pred, val_test_score, np.array(probabilities), torch.concat(features, dim=0)


if __name__ == '__main__':

    set_seed(seed=28)
    # model_path = '/home/johannes/Code/TumorRepresentationLearningGCN/experiments/NSCLC_ResNet18-3D_2024-02-22_09:33:50/model-250.pth'
    model_path = '/home/johannes/Code/TumorRepresentationLearningGCN/experiments/RADCURE_ResNet18-3D_2024-02-19_12:07:28/model-100.pth'

    test_time_augmentation = False
    test_time_augmentations = 1
    dataset = 'RADCURE'

    cnn = CNN(nclasses=2, cnn_name='ResNet18-3D').cuda()
    cnn.load_state_dict(torch.load(model_path))
    pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f'Number of Parameters: {pytorch_total_params}')

    train_dataset = SingleImgDataset3D(dataset=dataset, mode='train', cnn_name='ResNet18-3D', tta=test_time_augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset = SingleImgDataset3D(dataset=dataset, mode='val', cnn_name='ResNet18-3D', tta=test_time_augmentation)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    test_dataset = SingleImgDataset3D(dataset=dataset, mode='test', cnn_name='ResNet18-3D', tta=test_time_augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    print('num_test_files: '+str(len(test_dataset.filepaths)))

    f1_list = []
    mcc_list = []
    auc_list = []
    bacc_list = []
    scores = np.zeros((len(test_dataset.filepaths), 2)) ####

    with torch.no_grad():
        for i in range(test_time_augmentations):

            # test_dataset = SingleImgDataset3D(dataset=dataset, mode="test", cnn_name="ResNet18-3D")
            # indices = np.arange(start=0, stop=len(test_dataset), step=1)

            # bootstrap
            # indices, _ = shuffle(indices, indices, random_state=i)
            # indices = indices[:72]

            # cross-validation
            # split_length = int(np.floor(len(indices)/5))
            # indices = indices[i*split_length:(i+1)*split_length]

            # test_subset = torch.utils.data.Subset(test_dataset, indices)
            # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

            f1, mcc, auc, bacc, true, pred, score, probs, features = evaluate(model=cnn, loader=val_loader) ####

            f1_list.append(f1)
            mcc_list.append(mcc)
            auc_list.append(auc)
            bacc_list.append(bacc)
            scores += probs

            path = Path(model_path)
            parent_path = path.parent.absolute()
            # torch.save(features, f"{parent_path}/train_features{str(i).zfill(2)}.pt") ####
            # torch.save(torch.tensor(train_dataset.labels_train), f"{parent_path}/train_labels.pt") ####

    pred = torch.max(torch.tensor(scores), dim=1)[1]
    score = torch.softmax(torch.tensor(scores), dim=1)[:, 1]

    print(f"F1:   {f1_score(true, pred, average='macro')}")
    print(f'BACC: {balanced_accuracy_score(true, pred)}')
    print(f'MCC:  {matthews_corrcoef(true, pred)}')
    print(f'AUC:  {roc_auc_score(true, score)}')


    print(f'F1-Score: {np.mean(f1_list)}')
    print(f'std:      {np.std(f1_list)}')
    print(f'MCC:      {np.mean(mcc_list)}')
    print(f'std:      {np.std(mcc_list)}')
    print(f'AUC:      {np.mean(auc_list)}')
    print(f'std:      {np.std(auc_list)}')
    print(f'bACC:     {np.mean(bacc_list)}')
    print(f'std:      {np.std(bacc_list)}')

    torch.save(true, 'eval_results/true.pt')
    torch.save(pred, 'eval_results/pred.pt')
    torch.save(score, 'eval_results/score.pt')
