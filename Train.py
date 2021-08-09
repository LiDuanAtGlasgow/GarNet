#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torch.utils.data.sampler import BatchSampler
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from itertools import combinations
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
import pandas as pd
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import cv2
import time
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
import math
import numbers
import csv
import torchvision.models as models

cuda=torch.cuda.is_available()

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic=True

pars=argparse.ArgumentParser()
pars.add_argument('--train_mode',type=int,default=0,help='train modes')
par=pars.parse_args()


train_file='./train_file/'
test_file='./vali_file/'
img_path='img/'
csv_path='target/target.csv'

model_path='./Model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

class PhySNet_Dataset(Dataset):
    def __init__(self,train:bool=True,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None,opt=1)->None:
        super(PhySNet_Dataset,self).__init__()
        self.train=train
        self.train_file=train_file
        self.test_file=test_file
        self.img_path=img_path
        self.csv_path=csv_path
        if self.train:
            data_file=self.train_file
            data=pd.read_csv(data_file+self.csv_path)
            self.train_data=data.iloc[:,0]
            self.train_labels=data.iloc[:,opt]
        else:
            data_file=self.test_file
            data=pd.read_csv(data_file+self.csv_path)
            self.test_data=data.iloc[:,0]
            self.test_labels=data.iloc[:,opt]
        self.transform=transform
        self.target_transform=target_transform
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        if self.train:
            imgs_path=self.train_file+self.img_path+self.train_data[index]
            target=int(self.train_labels[index])
        else:
            imgs_path=self.test_file+self.img_path+self.test_data.iloc[index]
            target=int(self.test_labels.iloc[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            target=self.target_transform(target)
        noise=0.01*torch.rand_like(img)
        img=img+noise
        return img, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        self.train_file=self.mnist_dataset.train_file
        self.test_file=self.mnist_dataset.test_file
        self.img_path=self.mnist_dataset.img_path

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.to_numpy())
            self.label_to_indices = {label: np.where(self.train_labels.to_numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.to_numpy())
            self.label_to_indices = {label: np.where(self.test_labels.to_numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = cv2.imread(self.train_file+self.img_path+self.train_data[index],0), self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = cv2.imread(self.train_file+self.img_path+self.train_data[positive_index],0)
            img3 = cv2.imread(self.train_file+self.img_path+self.train_data[negative_index],0)
        else:
            img1 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][0]],0)
            img2 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][1]],0)
            img3 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][2]],0)

        img1 = Image.fromarray(img1, mode='L')
        img2 = Image.fromarray(img2, mode='L')
        img3 = Image.fromarray(img3, mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        img1=img1+0.01*torch.rand_like(img1)
        img2=img2+0.01*torch.rand_like(img2)
        img3=img3+0.01*torch.rand_like(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

class Bayesian_Dataset(Dataset):
    def __init__(self,file_path,csv_path,opt=1,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(Bayesian_Dataset,self).__init__()
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.test_data=data.iloc[:,0]
        self.test_labels=data.iloc[:,opt]
        self.labels=data.iloc[:,opt]
        self.transform=transform
        self.img_path='img/'
        self.csv_path='target/'
        self.data=data.iloc[:,0]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=self.imgs_path+self.data[index]
        target=int(self.labels[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet,self).__init__()
        self.convnet=nn.Sequential(
            nn.Conv2d(1,32,5),
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32,64,5,2),
            nn.PReLU()
        )
        self.fc=nn.Sequential(
            nn.Linear(64*61*61,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )
    
    def forward(self,x):
        output=self.convnet(x)
        output=output.reshape(output.size()[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

def frozon (model):
    for param in model.parameters():
        param.requires_grad=False
    return model

class AlexNet_Embedding(nn.Module):
    def __init__(self):
        super(AlexNet_Embedding,self).__init__()
        self.model=frozon(models.alexnet(pretrained=True))
        self.model.features[0]=nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.model.classifier=nn.Sequential(
            nn.Linear(256*6*6,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )
    
    def forward(self,x):
        output=self.model(x)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

class ResNet18_Embedding(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_Embedding,self).__init__()
        modeling=frozon(models.resnet18(pretrained=True))
        modules=list(modeling.children())[:-2]
        self.features=nn.Sequential(*modules)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        output=self.features(x)
        output=output.reshape(output.shape[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2,self).__init__()
    
    def forward(self,x):
        output=super(EmbeddingNetL2,self).forward(x)
        output=output.pow(2).sum(1,keepdim=True).sqrt()
        return output

    def get_emdding(self,x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self,embedding_net):
        super(TripletNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2,x3):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        output3=self.embedding_net(x3)
        return output1,output2,output3
    
    def get_emdding(self,x):
        return self.embedding_net(x)

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletAccuracy(nn.Module):
    def __init__(self):
        super(TripletAccuracy,self).__init__()
    
    def forward(self,anchor,positive,negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        return distance_positive<distance_negative




def fit(train_loader,val_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,metrics=[],start_epoch=0):
    accuracy_metric=TripletAccuracy()
    for epoch in range(0,start_epoch):
        scheduler.step()
    for epoch in range(start_epoch,n_epochs):
        scheduler.step()
        train_loss,metrics,accuracy=train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric)
        message='Epoch {}/{}. Train set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,train_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        val_loss,metrics,accuracy=test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric)
        val_loss/=len(val_loader)
        message+='\nEpoch {}/{}. Validation set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,val_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        print (message)

def train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric):
    for metric in metrics:
        metric.reset()
    model.train()
    losses=[]
    total_loss=0
    counter=0
    n=0

    for batch_idx,(data,target) in enumerate(train_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        optimizer.zero_grad()
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        losses.append(loss.item())
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        accuracies=accuracy_metric(*loss_inputs)
        n+=len(accuracies)
        for acc_idx in range (len(accuracies)):
            if accuracies[acc_idx]:
                counter+=1

        for metric in metrics:
            metric(outputs,target,loss_outputs)
        
        if batch_idx%log_interval==0:
            message='Train:[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(batch_idx*len(data[0]),len(train_loader.dataset),100*batch_idx/len(train_loader),np.mean(losses))
            for metric in metrics:
                message+='\t{}:{}'.format(metric.name(),metric.value())
            
            print (message)
            losses=[]
    accuracy=(counter/n)*100
    total_loss/=batch_idx+1
    return total_loss,metrics,accuracy

def test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
    model.eval()
    val_loss=0
    counter=0
    n=0
    for batch_idx,(data,target) in enumerate(val_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        val_loss+=loss.item()

        accuracies=accuracy_metric(*loss_inputs)
        n+=len(accuracies)
        for acc_idx in range(len(accuracies)):
            if accuracies[acc_idx]:
                counter+=1
        for metric in metrics:
            metric(outputs,target,loss_outputs)
    accuracy=(counter/n)*100
    return val_loss,metrics,accuracy

mean,std=0.00586554,0.03234654
train_dataset=PhySNet_Dataset(train=True,transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]
),opt=2)
test_dataset=PhySNet_Dataset(train=False,transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]),opt=2)

physnet_classes=['1','2','3']
colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#7f7f7f','#bcbd22','#17becf','#585957','#7f7f7f']
numbers=[1,2,3]
fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
print ('physnet_claasses:',len(physnet_classes))
print ('colors:',len(colors))

def plot_centre(embeddings,targets,xlim=None,ylim=None):
    plt.plot(figsize=(10,10))
    for i in range (len(physnet_classes)):
        inds=np.where(targets==numbers[i])[0]
        embedding_x=embeddings[inds,0].mean()
        embedding_y=embeddings[inds,1].mean()
        plt.scatter(embedding_x,embedding_y,alpha=0.5,color=colors[i])
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.legend(physnet_classes)
    plt.savefig(fig_path+'{:f}.png'.format(time.time()))
    plt.show()
def standard_label(embeddings,targets,numbers=10):
    f=open('./standard_labels.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('Name','mean_x','mean_y','std_x','std_y'))
    for i in range(numbers):
        inds=np.where(targets==i+1)[0]
        embedding_x=embeddings[inds,0].mean()
        embedding_y=embeddings[inds,1].mean()
        embedding_stdx=embeddings[inds,0].std()
        embedding_stdy=embeddings[inds,1].std()
        csv_writer.writerow((i+1,embedding_x,embedding_y,embedding_stdx,embedding_stdy))
def plot_embeddings(embeddings,targets,xlim=None,ylim=None):
    plt.figure(figsize=(10,10))
    for i in range (len(physnet_classes)):
        inds=np.where(targets==numbers[i])[0]
        plt.scatter(embeddings[inds,0],embeddings[inds,1],alpha=0.5,color=colors[i])
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.legend(physnet_classes)
    plt.savefig(fig_path+'{:f}.png'.format(time.time()))
    plt.show()
def extract_embeddings(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            k+=len(images)
    return embeddings,labels
def similarity_distance(embeddings,targets,standard_labels,test_labels,standard_std,test_std,number=5):
    f=open('./accuracies.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('Name','Accuracy'))
    for t in range(len(test_labels)):
            acc=0
            total=0
            data=test_labels[t]
            distances=np.zeros(len(standard_labels))
            sub_dist=np.zeros(len(standard_labels))
            for n in range(len(standard_labels)):
                dist=np.sum(np.power(standard_labels[n]-data,2))
                distances[n]=dist
            print ('distances:',distances)
            inds=np.where(targets==t+(number+1))[0]
            for i in range (len(inds)):
                    data=embeddings[inds[i]]
                    distances_local=np.zeros(number)
                    for n in range(number):
                        dist_local=np.sum(np.power(standard_labels[n]-data,2))
                        distances_local[n]=dist_local*1+distances[n]*0
                    pred_label=np.argmin(distances_local)
                    if pred_label==t:
                        acc+=1#Location Search
                    total+=1
            accs=(acc/total)*100
            print (f'[{t+1}]Accuracy:{accs}')

##########################################################################
batch_size=256
kwargs={'num_workers':1,'pin_memory':True} if cuda else {}
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)


if par.train_mode==1:
    triplet_train_dataset=TripletMNIST(train_dataset)
    triplet_test_dataset=TripletMNIST(test_dataset)
    batch_size=28
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    triplet_train_loader=DataLoader(triplet_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    triplet_test_loader=DataLoader(triplet_test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    margin=1
    embedding_net=ResNet18_Embedding()
    print ('embeding_net:',embedding_net)
    model=TripletNet(embedding_net)
    if cuda:
        model=model.cuda()
    lr=1e-3
    params=[]
    print ('---------Params-----------')
    for name,param in model.named_parameters():
        if param.requires_grad==True:
            print ('name:',name)
            params.append(param)
    print ('--------------------------')
    optimizer=optim.Adam(params,lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=30
    log_interval=100
    loss_fn=TripletLoss(margin)

    fit(triplet_train_loader,triplet_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval)
    torch.save(model,model_path+'%f.pth'%time.time())
    train_embeddings_triplet,train_labels_triplet=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_triplet,train_labels_triplet)
    val_embeddings_triplet,val_labels_triplet=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_triplet,val_labels_triplet)
print ('--finished!--')