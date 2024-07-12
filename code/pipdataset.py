import os
import cv2 as cv
import torchvision.models
from sklearn.model_selection import train_test_split
import torch
import torchvision as tv
import numpy as np
from torch.utils.data import Dataset,DataLoader

# caculate the values of mean and std of datasets of trian-images or test-images
def calmeanstd(image_paths,height,width):
    B_means=0.0
    G_means=0.0
    R_means=0.0
    B_stds=0.0
    G_stds=0.0
    R_stds=0.0
    for image_path in image_paths:
        image=cv.imread(image_path)
        image = cv.resize(image, (height, width))
        image = image.astype(np.float32) / 255.0
        B_means += np.sum(image[:, :, 0])
        G_means += np.sum(image[:, :, 1])
        R_means += np.sum(image[:, :, 2])
    # print("means : ", B_means, G_means, R_means)
    B_mean = B_means / len(image_paths) / height / width
    G_mean = G_means / len(image_paths) / height / width
    R_mean = R_means / len(image_paths) / height / width
    # print(B_mean,G_mean,R_mean)
    for image_path in image_paths:
        image=cv.imread(image_path)
        image = cv.resize(image, (height, width))
        image = image.astype(np.float32) / 255.0
        B_stds += np.sum((image[:, :, 0] - B_mean) ** 2)
        G_stds += np.sum((image[:, :, 1] - G_mean) ** 2)
        R_stds += np.sum((image[:, :, 2] - R_mean) ** 2)
    # print("stds : ", B_stds, G_stds, R_stds)
    B_std = np.sqrt(B_stds / len(image_paths) / height / width)
    G_std = np.sqrt(G_stds / len(image_paths) / height / width)
    R_std = np.sqrt(R_stds / len(image_paths) / height / width)
    # print(B_std,G_std,R_std)
    return B_mean,G_mean,R_mean,B_std,G_std,R_std
    # return 0.406,0.456,0.485,0.225,0.224,0.229

# define the dataset class
class PIPELINE_Dataset(torch.nn.Module):
    def __init__(self,height,width,B_mean,G_mean,R_mean,B_std,G_std,R_std,image_paths,labels):
        self.pip_trainsform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize((height, width)),
            # tv.transforms.Normalize(mean=[R_mean, G_mean, B_mean], std=[R_std, G_std, B_std]),
            tv.transforms.Normalize(mean=[B_mean, G_mean, R_mean], std=[B_std, G_std, R_std]),
        ])
        self.image_paths=image_paths
        self.labels=labels
    def __len__(self):
        return len(self.image_paths)
    def num_of_samples(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        image=cv.imread(self.image_paths[idx])
        # cv.imshow("raw",image)
        # print(image)
        # image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        # print(image)
        # cv.imshow("defect",image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        sample={'image':self.pip_trainsform(image),'label':torch.tensor(self.labels[idx])}
        # print(sample['image'])
        return sample
# define the pipeline model class

if __name__=="__main__":
    image_paths=[]
    labels=[]
    root_path="E:/pipeline/pipeimages/"
    subroot_names=os.listdir(root_path)
    for subroot_name in subroot_names:
        file_path = os.path.join(root_path, subroot_name)
        image_names = os.listdir(file_path)
        for image_name in image_names:
            image_path = os.path.join(file_path, image_name)
            image_paths.append(image_path)
            labels.append(int(subroot_name,10))
    train_image_paths,test_image_paths,train_labels,test_labels=train_test_split(image_paths,labels,test_size=0.2,random_state=42)
    height=200
    width=200
    # B_mean,G_mean,R_mean,B_std,G_std,R_std=calmeanstd(image_paths,height,width)
    # mean: 0.5180366271293186    0.5097805521608085    0.5413765879449826
    # std: 0.2194831033076909 0.2171821565891425 0.22535597874502292
    B_mean_train,G_mean_train,R_mean_train,B_std_train,G_std_train,R_std_train=calmeanstd(train_image_paths,height,width)
    # mean: 0.518402138245449    0.5101072355830509    0.5418920548041813
    # std: 0.21845120412545 0.21620868937206825 0.2241930401942555
    train_ds=PIPELINE_Dataset(height,width,B_mean_train,G_mean_train,R_mean_train,B_std_train,G_std_train,R_std_train,train_image_paths,train_labels)
    train_dl=DataLoader(train_ds,batch_size=16,shuffle=True,drop_last=True)
    for train_batch in train_dl:
        inputs_train=train_batch['image'].cuda()
        labels_train=train_batch['label'].cuda()
    print(len(train_ds),len(train_dl),len(train_batch),len(inputs_train),len(labels_train))
    B_mean_test,G_mean_test,R_mean_test,B_std_test,G_std_test,R_std_test=calmeanstd(test_image_paths,height,width)
    # mean:     0.5165745826647972        0.5084738184718389        0.5393147205081874
    # std:     0.22355709795402462 0.22102832417268378 0.22993736454244862
    test_ds=PIPELINE_Dataset(height,width,B_mean_test,G_mean_test,R_mean_test,B_std_test,G_std_test,R_std_test,test_image_paths,test_labels)
    test_dl=DataLoader(test_ds,batch_size=16,shuffle=True,drop_last=True)
    for test_batch in test_dl:
        inputs_test=test_batch['image'].cuda()
        labels_test=test_batch['label'].cuda()
    print(len(test_ds),len(test_dl),len(test_batch),len(inputs_test),len(labels_test))


