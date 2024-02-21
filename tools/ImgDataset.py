import numpy as np
from glob import glob
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms
import torchio as tio
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20, shuffle=True):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            stride = int(20/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)

class SingleImgDataset3D(torch.utils.data.Dataset):

    def __init__(self, dataset, mode, cnn_name, split=-1, ):      

        assert mode in ["train", "val", "test"]  
    
        self.mode = mode
        self.split = split
        self.dataset = dataset

        if self.dataset == "NSCLC":
            all_filepaths = glob("/media/johannes/WD Elements/NSCLC_Stefan/median_resampled+max_cropped/*image.pt")
            all_labels = glob("/media/johannes/WD Elements/NSCLC_Stefan/labels/*")
            all_labels_ids = [label.split("/")[-1].split("_")[0] for label in all_labels]

            files = sorted([file for file in all_filepaths if file.split("/")[-1].split("_")[0] in all_labels_ids])
            # files_new = [file for file in files if "LUNG1" in file.split("/")[-1]]
            # files = files_new
            files_ids = [file.split("/")[-1].split("_")[0] for file in files]

            labels = sorted([label for label in all_labels if label.split("/")[-1].split("_")[0] in files_ids])
            labels_binary = [0 if "adeno" in label else 1 for label in labels]

            self.train_imgs, self.val_imgs, self.labels_train, self.labels_val = train_test_split(files, labels_binary, train_size=0.6, 
                                                                                                  random_state=28, stratify=labels_binary)
            
            self.val_imgs, self.test_imgs, self.labels_val, self.labels_test = train_test_split(self.val_imgs, self.labels_val, test_size=0.5,
                                                                                                random_state=28, stratify=self.labels_val)

            test_val_images = self.test_imgs + self.val_imgs
            test_val_labels = self.labels_test + self.labels_val      
   
            self.test_imgs, self.labels_test = shuffle(test_val_images, test_val_labels, random_state=42)

        elif self.dataset == "RADCURE":
            # self.tta = tta
            # self.filepaths = sorted(glob("/home/johannes/Desktop/MICCAI24/data/RADCURE/volume+seg/*crimgnew.pt"))
            self.filepaths = sorted(glob("/home/johannes/Desktop/MICCAI24/data/RADCURE/volume+seg/*crimgnew.pt"))
            self.labelpaths = sorted(glob("/home/johannes/Desktop/MICCAI24/data/RADCURE/volume+seg/*label*.pt"))

            train_ids = torch.load("/home/johannes/Desktop/MICCAI24/data/RADCURE/X_train_RADCURE_split.pt")
            val_ids = torch.load("/home/johannes/Desktop/MICCAI24/data/RADCURE/X_val_RADCURE_split.pt")
            test_ids = torch.load("/home/johannes/Desktop/MICCAI24/data/RADCURE/X_test_RADCURE_split.pt")

            val_test = val_ids + test_ids
            np.random.shuffle(val_test,)
            half_size = int(len(val_test)/2)
            val_ids = val_test[:half_size]
            test_ids = val_test[half_size:]

            self.train_imgs = [path for path in self.filepaths if path.split("/")[-1].split("_")[0] in train_ids]
            self.val_imgs = [path for path in self.filepaths if path.split("/")[-1].split("_")[0] in val_ids]
            self.test_imgs = [path for path in self.filepaths if path.split("/")[-1].split("_")[0] in test_ids]

            if self.mode == "val":
                if self.split == 0:
                    self.val_imgs = self.val_imgs[:60]
                elif self.split == 1:
                    self.val_imgs = self.val_imgs[60:120]
                else:
                    self.val_imgs = self.val_imgs[120:]                

            self.train_labels = [path for path in self.labelpaths if path.split("/")[-1].split("_")[0] in train_ids]
            self.val_labels = [path for path in self.labelpaths if path.split("/")[-1].split("_")[0] in val_ids]
            self.test_labels = [path for path in self.labelpaths if path.split("/")[-1].split("_")[0] in test_ids]

            self.labels_train = [0 if "negative" in path else 1 for path in self.train_labels]
            self.labels_val = [0 if "negative" in path else 1 for path in self.val_labels]
            self.labels_test = [0 if "negative" in path else 1 for path in self.test_labels]

        if self.mode == "train":
            self.filepaths = self.train_imgs
        elif self.mode == "val":
            self.filepaths = self.val_imgs
        else:
            self.filepaths = self.test_imgs

        self.class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(self.labels_train), y=self.labels_train)).to(torch.float32).cuda()

        if cnn_name in ['resnet50_3D', 'resnet101_3D']:
            self.transform = tio.transforms.Compose([
            tio.transforms.Clamp(-1000, 1000),
            tio.transforms.ZNormalization(),
            tio.transforms.Resize(224),
            tio.transforms.RandomAffine(scales=0.05, degrees=10, translation=20)
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
            
        # elif not "new" in self.filepaths[0]:
        #     self.transform = tio.transforms.Compose([
        #         tio.transforms.Clamp(-1000, 1000),
        #         tio.transforms.Resize(96),
        #         tio.transforms.ZNormalization(),
        #         tio.transforms.RandomAffine(scales=0.05, degrees=10, translation=20)
        #         # transforms.RandomHorizontalFlip(),
        #         # transforms.ToTensor(),
        #         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #         #                      std=[0.229, 0.224, 0.225])
        # ])

        
        else: 

            self.train_transform = tio.transforms.Compose([
                # tio.transforms.Clamp(-1000, 1000),
                # tio.transforms.ZNormalization(),
                tio.transforms.RandomAffine(scales=0.0, degrees=0.0, translation=20),
                tio.transforms.RandomFlip(axes=(0, 1, 2))
                # tio.transforms.ZNormalization(),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
        ])
            
        #     self.val_test_transform = tio.transforms.Compose([
        #         # tio.transforms.Clamp(-1000, 1000),
        #         tio.transforms.ZNormalization(),
        #         # tio.transforms.RandomAffine(scales=0.05, degrees=10, translation=20)
        #         # transforms.RandomHorizontalFlip(),
        #         # transforms.ToTensor(),
        #         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #         #                      std=[0.229, 0.224, 0.225])
        # ])

    def __len__(self):

        if self.mode == "train":
            return len(self.train_imgs)
        elif self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.test_imgs)

    def __getitem__(self, idx):

        if self.mode == "train":
            images = self.train_imgs
            labels = self.labels_train
        elif self.mode == "val":
            images = self.val_imgs
            labels = self.labels_val
        else:
            images = self.test_imgs
            labels = self.labels_test
        
        path = images[idx]
        img = torch.load(path).unsqueeze(dim=0).to(torch.float32)
        label = torch.tensor(labels[idx]).to(torch.long)

        if self.mode == "train":
            # img = self.clip(img)
            img = torch.clamp(img, -1000, 1000)
            img = self.normalize(img)
            if self.train_transform:
                img = self.train_transform(img)
        
        else:
            img = torch.clamp(img, -1000, 1000)
            img = self.normalize(img)
        #     if self.val_test_transform:
        #         img = self.val_test_transform(img)
        
        return (label, img, path)
    
    def normalize(self, volume):
        return (volume - torch.mean(volume)) / torch.std(volume)
    
    def clip(self, volume):
        min_value = torch.quantile(volume, 0.05)
        max_value = torch.quantile(volume, 0.95)
        return torch.clamp(volume, min=min_value, max=max_value)    
    
class MultiviewImgDataset3D(torch.utils.data.Dataset):
    def __init__(self, mode, n_augmentations, scales, degrees, translations):

        assert mode in ["train", "val", "test"]  

        self.mode = mode
        self.n_augmentations = n_augmentations
        self.filepaths = sorted(glob("/home/johannes/Desktop/MICCAI24/data/RADCURE/volume+seg/*crimgnew.pt"))
        self.labelpaths = sorted(glob("/home/johannes/Desktop/MICCAI24/data/RADCURE/volume+seg/*label*.pt"))

        train_ids = torch.load("/home/johannes/Desktop/MICCAI24/data/RADCURE/X_train_RADCURE_split.pt")
        val_ids = torch.load("/home/johannes/Desktop/MICCAI24/data/RADCURE/X_val_RADCURE_split.pt")
        test_ids = torch.load("/home/johannes/Desktop/MICCAI24/data/RADCURE/X_test_RADCURE_split.pt")

        val_test = val_ids + test_ids
        np.random.shuffle(val_test,)
        half_size = int(len(val_test)/2)
        val_ids = val_test[:half_size]
        test_ids = val_test[half_size:]

        self.train_imgs = [path for path in self.filepaths if path.split("/")[-1].split("_")[0] in train_ids]
        self.val_imgs = [path for path in self.filepaths if path.split("/")[-1].split("_")[0] in val_ids]
        self.test_imgs = [path for path in self.filepaths if path.split("/")[-1].split("_")[0] in test_ids]

        self.train_labels = [path for path in self.labelpaths if path.split("/")[-1].split("_")[0] in train_ids]
        self.val_labels = [path for path in self.labelpaths if path.split("/")[-1].split("_")[0] in val_ids]
        self.test_labels = [path for path in self.labelpaths if path.split("/")[-1].split("_")[0] in test_ids]

        self.labels_train = [0 if "negative" in path else 1 for path in self.train_labels]
        self.labels_val = [0 if "negative" in path else 1 for path in self.val_labels]
        self.labels_test = [0 if "negative" in path else 1 for path in self.test_labels]

        if self.mode == "train":
            self.filepaths = self.train_imgs
        elif self.mode == "val":
            self.filepaths = self.val_imgs
        else:
            self.filepaths = self.test_imgs

        self.class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(self.labels_train), y=self.labels_train)).to(torch.float32).cuda()

        self.preprocess = tio.transforms.Compose([
            tio.transforms.Clamp(-1000, 1000),
            tio.transforms.ZNormalization()
        ])

        # self.augment = tio.transforms.RandomAffine(scales=0.0, translation=0.0, degrees=180.0)

        self.augmentations = []
        # scales = np.random.uniform(0.95, 1.05, n_augmentations)
        # degrees = np.random.uniform(-10, 10, n_augmentations)
        # translations = np.random.uniform(-20, 20, n_augmentations)
        for i in range(n_augmentations):
            self.augmentations.append(tio.transforms.RandomAffine(scales=(scales[i], scales[i]),
                                                                  degrees=(degrees[i], degrees[i]),
                                                                  translation=(translations[i], translations[i])))

        # self.augment = tio.transforms.RandomAffine(scales=0.05, degrees=10, translation=20)
        

    def __len__(self):
        if self.mode == "train":
            return len(self.train_imgs)
        elif self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.test_imgs)

    def __getitem__(self, idx):

        if self.mode == "train":
            images = self.train_imgs
            labels = self.labels_train
        elif self.mode == "val":
            images = self.val_imgs
            labels = self.labels_val
        else:
            images = self.test_imgs
            labels = self.labels_test

        path = images[idx]
        img = torch.load(path).unsqueeze(dim=0)
        label = torch.tensor(labels[idx]).to(torch.long)

        img_augmentations = []
        for i in range(self.n_augmentations):
            img_augmentations.append(self.augmentations[i](img))
        
        img_augmentations = torch.concatenate(img_augmentations)

        return label, img_augmentations
