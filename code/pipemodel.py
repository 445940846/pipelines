from pipdataset import calmeanstd,PIPELINE_Dataset
import os
import torch
import time
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,classification_report
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv

class PIPEResNet18(torch.nn.Module):
    def __init__(self):
        super(PIPEResNet18,self).__init__()
        self.cnn_layers=tv.models.resnet18(pretrained=True)
        num_ftrs=self.cnn_layers.fc.in_features
        self.cnn_layers.fc=torch.nn.Linear(num_ftrs,4)
    def forward(self,x):
        out=self.cnn_layers(x)
        return out

def train(epoch):
    model.train()
    train_index=0
    epoch_losses=0.0
    train_start=time.time()
    for train_batch in train_dl:
        train_inputs=train_batch['image'].cuda()
        train_labels=train_batch['label'].cuda()
        train_outputs=model(train_inputs)
        train_labels=train_labels.long()
        step_losses=criterion(train_outputs,train_labels)
        epoch_losses+=step_losses
        optimizer.zero_grad()
        step_losses.backward()
        optimizer.step()
        # writer.add_scalar("train_step_loss",step_losses/len(train_inputs),epoch*len(train_dl)+train_index)
        # print("\rTrain-->Epoch: {} [batch: {}/{} loss: {}]".format(epoch,train_index,len(train_dl),step_losses.item()),end='')
        train_index+=1
    # run this code when set drop_last=False
    # writer.add_scalar("train_epoch_loss",epoch_losses/len(train_ds),epoch)
    # run this code when set drop_last=True
    # writer.add_scalar("train_epoch_loss",epoch_losses/len(train_dl)/len(train_inputs),epoch)
    train_last=time.time()-train_start
    return train_last,epoch_losses/len(train_ds)

def test(epoch):
    model.eval()
    test_index=0
    epoch_accuracy=0.0
    real_labels=[]
    pred_labels=[]
    test_start=time.time()
    with torch.no_grad():
        for test_batch in test_dl:
            real_labels.extend(test_batch['label'].numpy())
            test_inputs=test_batch['image'].cuda()
            test_labels=test_batch['label'].cuda()
            test_outputs=model(test_inputs)
            test_outputs_cpu=test_outputs.cpu()
            pred=torch.argmax(test_outputs,dim=1)
            pred_labels.extend(pred.cpu().numpy())
            step_accuracy=(pred==test_labels).sum().item()
            # print(len(test_ds),len(test_dl),len(test_inputs))
            epoch_accuracy+=step_accuracy
            # writer.add_scalar("test_step_accuracy",step_accuracy/len(test_inputs),epoch*len(test_dl)+test_index)
            # writer.add_scalar("test_step_accuracy",precision_score(test_labels.cpu(),pred.cpu(),average='micro'),epoch*len(test_dl)+test_index)
            # print("\rTest-->Epoch: {} [batch: {}/{} accuracy: {}]".format(epoch,test_index,len(test_dl),step_accuracy/len(test_inputs)))
            # print("\rTest-->Epoch: {} [batch: {}/{} precision_micro: {}]".format(epoch,test_index,len(test_dl),precision_score(test_labels.cpu(),pred.cpu(),average='micro')))
            test_index+=1
        # run this code when set drop_last=False
        # writer.add_scalar("test_epoch_accuracy",epoch_accuracy/len(test_ds),epoch)
        # run this code when set drop_last=True
        # writer.add_scalar("test_epoch_accuracy", epoch_accuracy / len(test_dl) / len(test_inputs), epoch)
        test_last=time.time()-test_start
    return test_last,epoch_accuracy/len(test_ds),real_labels,pred_labels

if __name__=="__main__":
    image_paths=[]
    labels=[]
    train_time=0.0
    test_time=0.0
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
    B_mean_train,G_mean_train,R_mean_train,B_std_train,G_std_train,R_std_train=calmeanstd(train_image_paths,height,width)
    train_ds=PIPELINE_Dataset(height,width,B_mean_train,G_mean_train,R_mean_train,B_std_train,G_std_train,R_std_train,train_image_paths,train_labels)
    train_dl=DataLoader(train_ds,batch_size=64,shuffle=True,drop_last=True)
    B_mean_test,G_mean_test,R_mean_test,B_std_test,G_std_test,R_std_test=calmeanstd(test_image_paths,height,width)
    test_ds=PIPELINE_Dataset(height,width,B_mean_test,G_mean_test,R_mean_test,B_std_test,G_std_test,R_std_test,test_image_paths,test_labels)
    test_dl=DataLoader(test_ds,batch_size=64,shuffle=True,drop_last=False)
    writer=SummaryWriter("E:/pipeline/log/")
    writer_iter=iter(train_dl)
    writer_batch=writer_iter.next()
    writer_images,writer_labels=writer_batch['image'],writer_batch['label']
    ii=0
    # for image in writer_images:
    #     image=image.numpy().transpose((1,2,0))
    #     image=cv.cvtColor(image,cv.COLOR_RGB2BGR)
    #     print(image)
    #     cv.imshow("raw",image)
    #     cv.imwrite("E:/pipeline/log/3/"+str(ii)+".jpg",image)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    #     ii+=1
    for label in writer_labels:
        print(label)
    writer_images_grid=tv.utils.make_grid(writer_images)
    writer.add_image("four_fashion_pipleline_images",writer_images_grid)
    model=PIPEResNet18().cuda()
    writer.add_graph(model,writer_images.cuda())
    # print(model)
    criterion=torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    epoches=100
    best_accuracy=0.0
    for epoch in range(epoches):
        train_last,epoch_loss=train(epoch)
        print("\ntraining time in epoch %d: %f"%(epoch,train_last))
        train_time+=train_last
        test_last,epoch_accuracy,real_labels,pred_labels=test(epoch)
        print("\ntest time in epoch %d: %f"%(epoch,test_last))
        test_time+=test_last
        if best_accuracy<=epoch_accuracy:
            best_accuracy=epoch_accuracy
            print("Saving the model...")
            # print(real_labels)
            # print(real_labels.reshape(-1))
            # print(np.shape(real_labels))
            # print(pred_labels)
            # print(np.shape(pred_labels))
            confusion=confusion_matrix(real_labels,pred_labels)
            print(confusion)
            print(classification_report(real_labels,pred_labels))
            torch.save(model.state_dict(),"PIPEResNet18.pt")
        else:
            print("Do not save the model...")
    print(train_time)
    print(test_time)
    writer.close()
