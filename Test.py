#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv
import scipy.stats
from shapely import geometry
from continuous_perception import continuous_perception,continuous_perception_plotting,early_stop, early_stop_plotting,early_stop_anime

np.random.seed(42)
torch.manual_seed(42) 

cuda=torch.cuda.is_available()
model_path='./Model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

physnet_classes=['1','2','3','4','5','11','12','13','14','15']
colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
numbers=[1,2,3,4,5,11,12,13,14,15]
mean,std=0.00586554,0.03234654
print ('physnet_claasses:',len(physnet_classes))
print ('colors:',len(colors))

def extract_embeddings(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        video_labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target, video_label in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            video_labels[k:k+len(images)]=video_label.numpy()
            k+=len(images)
    return embeddings,labels,video_labels

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

class GarNet_Dataset(Dataset):
    def __init__(self,file_path,csv_path,opt=1,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(GarNet_Dataset,self).__init__()
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.labels=data.iloc[:,opt]
        self.transform=transform
        self.data=data.iloc[:,0]
        self.video_labels=data.iloc[:,3]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=self.imgs_path+self.data[index]
        target=int(self.labels[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        video_label=int(self.video_labels[index])
        return img, target, video_label
    
    def __len__(self):
        return len(self.data)

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

def bandwidth_investigation(embedding,labels,n=5):
    bandwidth_value=5
    plt.figure(figsize=(10,10))
    if n==5:
        color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
        label_plottings=['pant','shirt','sweater','towel','tshirt']
        len_inds=[600,600,600,600,600]
    if n==3:
        color=['#1f77b4','#ff7f01','#2ca02c','#1f77b4','#ff7f01','#2ca02c']
        label_plottings=['light','medium','heavy']
        len_inds=[600,1200,1200]
    contours=[]
    standard_points=np.zeros((n,2))
    f=open('./garnet_cp_tracking.csv','w')
    csv_writer=csv.writer(f)
    if n==5:
        csv_writer.writerow(('No.','Pant','Shirt','Sweater','Towel','T-shirt','Accuracy'))
    if n==3:
        csv_writer.writerow(('No.','Light','Medium','Heavy','Accuracy'))
    for i in range (n):
        inds=np.where(labels==i+1)[0]
        data=embedding[inds]*10
        x=data[:,0].mean()
        y=data[:,1].mean()
        pdf=scipy.stats.kde.gaussian_kde(data.T)
        q,w=np.meshgrid(range(-40,40,1), range(-10,50,1))
        r=pdf([q.flatten(),w.flatten()])
        s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), bandwidth_value)
        r.shape=(60,80)
        cont=plt.contour(range(-40,40,1), range(-10,50,1), r, [s],colors=color[i])
        cont_location=[]
        for line in cont.collections[0].get_paths():
            cont_location.append(line.vertices)
        cont_location=np.array(cont_location)[0]
        contours.append(cont_location)
        plt.plot(x,y,'o',color=color[i],label=label_plottings[i])
        standard_points[i,:]=(x,y)
    plt.show()
    
    start_time=time.time()
    for t in range (n):
        numbering=n
        acc=0
        total=0
        for idx in range (len_inds[t]):
            inds=np.where(labels==t+numbering+1)
            data=embedding[inds]*10
            x=data[:idx+1,0].mean()
            y=data[:idx+1,1].mean()
            mean_value_point=np.array([x,y])
            contains_acc=[]
            for m in range (n):
                line = geometry.LineString(contours[m])
                point = geometry.Point(x,y)
                polygon = geometry.Polygon(line)
                if polygon.contains(point):
                    contains_acc.append(m)
            if len(contains_acc)==1:
                if contains_acc[0]==t:
                    acc+=1
            elif len(contains_acc)>1:
                dists=np.zeros((len(contains_acc),2))
                for h in range(len(contains_acc)):
                    standard_point=standard_points[contains_acc[h]]
                    dis=np.sum(np.power(standard_point-mean_value_point,2))
                    dists[h,0]=dis
                    dists[h,1]=contains_acc[h]
                min_val=np.argmin(dists[:,0])
                if dists[min_val,1]==t:
                    acc+=1
            total+=1
            accuracy=100*(acc/total)
            dists_for_tracking=[]
            for dist_idx in range (n):
                standard_point=standard_points[dist_idx]
                dist_for_tracking=np.sum(np.power(standard_point-mean_value_point,2))
                dists_for_tracking.append(dist_for_tracking)
            #print ('dists_for_tracking:',dists_for_tracking[0])
            if n==5:
                csv_writer.writerow((idx+1,dists_for_tracking[0],dists_for_tracking[1],dists_for_tracking[2],
                dists_for_tracking[3],dists_for_tracking[4],accuracy))
            if n==3:
                csv_writer.writerow((idx+1,dists_for_tracking[0],dists_for_tracking[1],dists_for_tracking[2],
                accuracy))
        accuracies=100*(acc/total)
        print ('[Category]'+label_plottings[t]+'[Accuracy]'+str(accuracies)+'[Time]'+str(time.time()-start_time))
        start_time=time.time()

def garnet_cp_plotting(cp_data,samp_number=600,number_category=5):
    samps_number=np.array([0,600,1800,3000])
    for i in range (number_category):
        if number_category==5:
            names=['Pant','Shirt','Sweater','Towel','T-shirt']
            df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'pant':cp_data[i*samp_number:(i+1)*samp_number,1],
            'shirt':cp_data[i*samp_number:(i+1)*samp_number,2],'sweater':cp_data[i*samp_number:(i+1)*samp_number,3],
            'towel':cp_data[i*samp_number:(i+1)*samp_number,4],'tshirt':cp_data[i*samp_number:(i+1)*samp_number,5],
            'acc':cp_data[i*samp_number:(i+1)*samp_number,6]})
            plt.figure()
            subplot=plt.subplot()
            subplot.plot('x','pant',data=df,color='red',label='pant')
            subplot.plot('x','shirt',data=df,color='blue',label='shirt')
            subplot.plot('x','sweater',data=df,color='green',label='sweater')
            subplot.plot('x','towel',data=df,color='purple',label='towel')
            subplot.plot('x','tshirt',data=df,color='gray',label='tshirt')
            subplot.set_xlabel('Input')
            subplot.set_ylabel('Decision Distance')
            subplot2=subplot.twinx()
            subplot2.plot('x','acc',data=df,color='grey',linewidth=10,label='Accuracy',linestyle='dotted')
            subplot2.set_ylabel('Accuracy(%)')
            subplot2.set_ylim([0,100])
            plt.legend()
            plt.title(names[i])
            plt.savefig('./'+names[i]+'_category.png')
            plt.show()
        if number_category==3:
            names=['Light','Medium','Heavy']
            df=pd.DataFrame({'x':cp_data[samps_number[i]:samps_number[i+1],0],'light':cp_data[samps_number[i]:samps_number[i+1],1],
            'medium':cp_data[samps_number[i]:samps_number[i+1],2],'heavy':cp_data[samps_number[i]:samps_number[i+1],3],
            'acc':cp_data[samps_number[i]:samps_number[i+1],4]})
            plt.figure()
            subplot=plt.subplot()
            subplot.plot('x','light',data=df,color='red',label='light')
            subplot.plot('x','medium',data=df,color='blue',label='medium')
            subplot.plot('x','heavy',data=df,color='green',label='heavy')
            subplot.set_xlabel('Input')
            subplot.set_ylabel('Decision Distance')
            subplot2=subplot.twinx()
            subplot2.plot('x','acc',data=df,color='grey',linewidth=10,label='Accuracy',linestyle='dotted')
            subplot2.set_ylabel('Accuracy(%)')
            subplot2.set_ylim([0,100])
            plt.legend()
            plt.title(names[i])
            plt.savefig('./'+names[i]+'_category.png')
            plt.show()

def bandwidth_investigations():
    band_90=[0,0,0]
    band_75=[25,30,20]
    band_50=[75,85,90]
    band_25=[100,100,90]
    band_5=[100,100,100]
    band_1=[100,100,100]
    x=[1,2,3]

    df=pd.DataFrame({'x':x,'band_90':band_90,'band_75':band_75,'band_50':band_50,'band_25':
    band_25,'band_5':band_5,'band_1':band_1})

    plt.plot('x','band_90',data=df,label='10% Bandwidth',linestyle='dotted')
    plt.plot('x','band_75',data=df,label='25% Bandwidth',linestyle='dotted')
    plt.plot('x','band_50',data=df,label='50% Bandwidth',linestyle='dotted')
    plt.plot('x','band_25',data=df,label='75% Bandwidth',linestyle='dotted')
    plt.plot('x','band_5',data=df,label='95% Bandwidth')
    plt.plot('x','band_1',data=df,label='99% Bandwidth',linestyle='dotted')
    plt.legend(loc='best')
    plt.title('Bandwidth V.S Accuracy')
    plt.xlabel('Bandwidth')
    plt.ylabel('Accuracy')
    plt.xticks(x,['lights','mediums','heavies'])
    plt.savefig('./bandwidth/bandwidth.png')
    plt.show()

batch_size=32
kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
model=torch.load(model_path+'robot_depth_weight-v9.pth')
file_path='./Database/depth'
data='/'
csv_path='./explore.csv'
dataset=GarNet_Dataset(file_path+data,csv_path,transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]),opt=2)
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,**kwargs)
embeddings,labels, video_labels=extract_embeddings(dataloader,model)
plot_embeddings(embeddings,labels)
continuous_perception(embeddings,labels,video_labels,n=3)
#bandwidth_investigation(embeddings,labels,n=5)
#garnet_cp_drawing=pd.read_csv('./garnet_cp_tracking.csv').to_numpy()
#continuous_perception_plotting(garnet_cp_drawing,number_category=3)
#bandwidth_investigations()
#early_stop(embeddings,labels,video_labels,n=3)
#garnet_cp_drawing=pd.read_csv('./garnet_cp_tracking.csv').to_numpy()
#early_stop_plotting(garnet_cp_drawing,number_category=3)
#early_stop_anime(garnet_cp_drawing,number_category=3)
print ('--finsihed!--')