#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import sys
from matplotlib.contour import ContourSet
import numpy
import scipy
from numpy.random import seed
from pandas.core.frame import DataFrame
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import pandas as pd
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
from scipy.stats import f
from shapely import geometry
import time

cuda=torch.cuda.is_available()

numpy.random.seed(42)

class SGEResNet18_Embedding(nn.Module):
    def __init__(self) -> None:
        super(SGEResNet18_Embedding,self).__init__()
        modeling=sge_resnet18()
        modules=list(modeling.children())
        self.features=nn.Sequential(*modules)[:-2]
        self.fc=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        output=self.features(x)
        #print ('output:',output.shape)
        output=output.reshape(output.shape[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

def ellipse_error_circling(embedding,labels,n=5):
    plt.figure(figsize=(10,10))
    color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
    label_plottings=['pant','shirt','sweater','towel','tshirt']
    contours=[]
    standard_points=np.zeros((n,2))
    for i in range (n):
        inds=numpy.where(labels==i+1)[0]
        data=embedding[inds]*10
        x=data[:,0].mean()
        y=data[:,1].mean()
        pdf=scipy.stats.kde.gaussian_kde(data.T)
        q,w=numpy.meshgrid(range(-40,60,1), range(-40,40,1))
        r=pdf([q.flatten(),w.flatten()])
        s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), 5)
        r.shape=(80,100)
        cont=plt.contour(range(-40,60,1), range(-40,40,1), r, [s],colors=color[i])
        cont_location=[]
        for line in cont.collections[0].get_paths():
            cont_location.append(line.vertices)
        cont_location=numpy.array(cont_location)[0]
        contours.append(cont_location)
        plt.plot(x,y,'o',color=color[i],label=label_plottings[i])
        standard_points[i,:]=(x,y)
    for t in range (n):
        inds=numpy.where(labels==t+n+1)
        data=embedding[inds]*10
        x=data[:,0].mean()
        y=data[:,1].mean()
        mean_value_point=numpy.array([x,y])
        contains_acc=[]
        for m in range (n):
            line = geometry.LineString(contours[m])
            point = geometry.Point(x,y)
            polygon = geometry.Polygon(line)
            if polygon.contains(point):
                contains_acc.append(m)
        if len(contains_acc)==0:
            print ('do not belong to any circles')
        elif len(contains_acc)==1:
            if contains_acc[0]==t:
                print ('True')
            else:
                print ('False')
        else:
            dists=np.zeros((len(contains_acc),2))
            for h in range(len(contains_acc)):
                standard_point=standard_points[contains_acc[h]]
                dis=np.sum(np.power(standard_point-mean_value_point,2))
                dists[h,0]=dis
                dists[h,1]=contains_acc[h]
            min_val=np.argmin(dists[:,0])
            if dists[min_val,1]==t:
                print ('True')
            else:
                print ('False')
        #plt.plot(x,y,'r*',color=color[t+n],label=t+n+1)
    plt.title('Epllise Error Circle')
    plt.legend()
    plt.show()

def point_changes(embedding,labels,n=5,len_inds=2000):
    plt.figure(figsize=(10,10))
    if n==5:
        labellings=['pant','shirt','sweater','tower','t-shirt','pant','shirt','sweater','tower','t-shirt']
        color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
    if n==3:
        labellings=['light','medium','heavy','light','medium','heavy']
        color=['#1f77b4','#ff7f01','#2ca02c','#1f77b4','#ff7f01','#2ca02c']
    contours=[]
    standard_points=np.zeros((n,2))
    start_time=time.time()
    for i in range (n):
            inds=numpy.where(labels==i+1)[0]
            data=embedding[inds]*10
            x=data[:,0].mean()
            y=data[:,1].mean()
            pdf=scipy.stats.kde.gaussian_kde(data.T)
            q,w=numpy.meshgrid(range(-40,60,1), range(-40,40,1))
            r=pdf([q.flatten(),w.flatten()])
            s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), 5)
            r.shape=(80,100)
            cont=plt.contour(range(-40,60,1), range(-40,40,1), r, [s],colors=color[i])
            cont_location=[]
            for line in cont.collections[0].get_paths():
                cont_location.append(line.vertices)
            cont_location=numpy.array(cont_location)[0]
            contours.append(cont_location)
            plt.plot(x,y,'o',color=color[i],label='train:'+labellings[i])
            standard_points[i,:]=(x,y)
    for m in range (len_inds):
        points=[]
        for t in range (n):
            inds=numpy.where(labels==t+n+1)
            data=embedding[inds]*10
            x_mean=data[:m+1,0].mean()
            y_mean=data[:m+1,1].mean()
            x=data[:m,0]
            y=data[:m,1]
            x_current=data[m,0]
            y_current=data[m,1]
            point_scatter =plt.scatter(x,y,alpha=0.15,color=color[t+n])
            point,=plt.plot(x_current,y_current,'r*',color='green',label=labellings[t+n],markersize=20)
            plt.title('Epllise Error Circle')
            plt.legend()
            points.append(point)
            points.append(point_scatter)
        if m%int(len_inds/10)==0:
            print (f'[Batch: {m+1}/{len_inds}] has been finished, time:{time.time()-start_time}')
        plt.savefig('./plottings/point_'+str(m+1).zfill(4)+'.png')
        for point in points:
            point.remove()

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

model=torch.load('./Model/robot_depth-v9.pth')
print ('------------------')
for name,param in model.named_parameters():
    print ('name: \t',name)
    param.requires_grad=False
print ('-------------------')

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

class T_Dataset(Dataset):
    def __init__(self,file_path,csv_path,opt=1,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.labels=data.iloc[:,opt]
        self.transform=transform
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

physnet_classes=['pant','shirt','sweater','towel','tshirt']
colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#7f7f7f','#bcbd22','#17becf','#585957','#7f7f7f']
numbers=[1,2,3,4,5]
fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
print ('physnet_claasses:',len(physnet_classes))
print ('colors:',len(colors))

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


mean,std=0.00586554,0.03234654
img_path='./Database/depth/'
csv_path='./explore.csv'
batch_size=32
dataset=T_Dataset(file_path=img_path,csv_path=csv_path,opt=1,transform=T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize((mean),(std))
]))
kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,**kwargs)
embeddings,labels=extract_embeddings(dataloader,model)
plot_embeddings(embeddings,labels)
ellipse_error_circling(embeddings,labels,n=5)
#point_changes(embeddings,labels,n=5)
print ('finished!')