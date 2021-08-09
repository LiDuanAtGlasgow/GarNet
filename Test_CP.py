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

np.random.seed(42)
torch.manual_seed(42) 

cuda=torch.cuda.is_available()
model_path='./Model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

physnet_classes=['1','2','3','4','5','6','7','8','9','10']
colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#7f7f7f','#bcbd22','#17becf','#585957','#232b08']
numbers=[1,2,3,4,5,6,7,8,9,10]
mean,std=0.00586554,0.03234654
print ('physnet_claasses:',len(physnet_classes))
print ('colors:',len(colors))

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
        self.target_transform=target_transform
        self.train=False
        self.train_file='./test_session/'
        self.test_file='./test_session/'
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
        if self.target_transform is not None:
            target=self.target_transform(target)
        return img, target
    
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

def cp_drawing(embeddings,targets,standard_labels,test_labels,standard_std,test_std,number=5):
    f=open('./cp_tracking.csv','w')
    csv_writer=csv.writer(f)
    if number==5:
        csv_writer.writerow(('No.','Pant','Shirt','Sweater','Towel','T-shirt','Accuracy'))
    if number==3:
        csv_writer.writerow(('No.','Light','Medium','Heavy','Accuracy'))
    for t in range(len(test_labels)):
            acc=0
            total=0
            inds=np.where(targets==t+(number+1))[0]
            globals=np.zeros((len(inds),2))
            test_global=np.array([0,0])
            for i in range (len(inds)):
                    data=embeddings[inds[i]]
                    distances_decision=np.zeros(number)
                    for n in range(number):
                        dist_local=np.sum(np.power(standard_labels[n]-data,2))
                        globals[i,:]=data
                        glo_location=np.array([globals[:,0].sum()/(i+1),globals[:,1].sum()/(i+1)])
                        dist_global=np.sum(np.power(standard_labels[n]-glo_location,2))
                        distances_decision[n]=dist_local*0+dist_global*1
                        test_global=glo_location
                    pred=np.argmin(distances_decision)
                    if pred==t:
                        acc+=1
                    total+=1
                    accuracy=100*(acc/total)
                    if number==5:
                        csv_writer.writerow((i,distances_decision[0],distances_decision[1],distances_decision[2],distances_decision[3],distances_decision[4],accuracy))
                    if number==3:
                        csv_writer.writerow((i,distances_decision[0],distances_decision[1],distances_decision[2],accuracy))
            print (f'[{t+1}] global_location: {test_global}')
            if number==5:
                print (f'[{t+1}] Decision_Distances:{distances_decision[0]}/{distances_decision[1]}/{distances_decision[2]}/{distances_decision[3]}/{distances_decision[4]},Accuracy:{accuracy}')
            if number==3:
                print (f'[{t+1}] Decision_Distances:{distances_decision[0]}/{distances_decision[1]}/{distances_decision[2]},Accuracy:{accuracy}')

def cp_plotting(cp_data,samp_number=2000,number_category=5):
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
            subplot2.plot('x','acc',data=df,color='black',label='Accuracy',linestyle='--')
            subplot2.set_ylabel('Accuracy(%)')
            plt.legend()
            plt.title(names[i])
            plt.savefig('./'+names[i]+'_category.png')
            plt.show()
        if number_category==3:
            names=['Light','Medium','Heavy']
            df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'light':cp_data[i*samp_number:(i+1)*samp_number,1],
            'medium':cp_data[i*samp_number:(i+1)*samp_number,2],'heavy':cp_data[i*samp_number:(i+1)*samp_number,3],
            'acc':cp_data[i*samp_number:(i+1)*samp_number,4]})
            plt.figure()
            subplot=plt.subplot()
            subplot.plot('x','light',data=df,color='red',label='light')
            subplot.plot('x','medium',data=df,color='blue',label='medium')
            subplot.plot('x','heavy',data=df,color='green',label='heavy')
            subplot.set_xlabel('Input')
            subplot.set_ylabel('Decision Distance')
            subplot2=subplot.twinx()
            subplot2.plot('x','acc',data=df,color='black',label='Accuracy',linestyle='--')
            subplot2.set_ylabel('Accuracy(%)')
            plt.legend()
            plt.title(names[i])
            plt.savefig('./'+names[i]+'_category.png')
            plt.show()

def cp_anime(cp_data,samp_number=2000,number_category=5):
    samps_number=np.array([0,4000,6000,10000])
    for i in range (number_category-1):
        i=i+1
        if number_category==5:
            names=['Pant','Shirt','Sweater','Towel','T-shirt']
            df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'pant':cp_data[i*samp_number:(i+1)*samp_number,1],
            'shirt':cp_data[i*samp_number:(i+1)*samp_number,2],'sweater':cp_data[i*samp_number:(i+1)*samp_number,3],
            'towel':cp_data[i*samp_number:(i+1)*samp_number,4],'tshirt':cp_data[i*samp_number:(i+1)*samp_number,5],
            'acc':cp_data[i*samp_number:(i+1)*samp_number,6]})
            start_time=time.time()
            for t in range (samp_number):
                x=df['x'].to_numpy()[:t+1]
                pant=df['pant'].to_numpy()[:t+1]
                shirt=df['shirt'].to_numpy()[:t+1]
                sweater=df['sweater'].to_numpy()[:t+1]
                towel=df['towel'].to_numpy()[:t+1]
                tshirt=df['tshirt'].to_numpy()[:t+1]
                acc=df['acc'].to_numpy()[:t+1]
                plt.figure()
                subplot=plt.subplot()
                subplot.plot(x,acc,color='black',label='Accuracy')
                subplot.plot(x,pant,color='red',label='pant')
                subplot.plot(x,shirt,color='blue',label='shirt')
                subplot.plot(x,sweater,color='green',label='sweater')
                subplot.plot(x,towel,color='purple',label='towel')
                subplot.plot(x,tshirt,color='gray',label='tshirt')
                subplot.set_xlabel('Input')
                subplot.set_ylabel('Decision Distance')
                subplot2=subplot.twinx()
                subplot2.plot(x,acc,color='black',label='Accuracy',linestyle='--')
                subplot2.set_ylabel('Accuracy(%)')
                plt.legend()
                plt.title(names[i])
                plt.xlim([1,samp_number])
                plt.savefig('./cp_anime/shape/'+names[i]+'_'+str(t+1).zfill(4)+'.png')
                plt.close()
                if t%int(samp_number/10)==0:
                    print (f'[{names[i]}][{t+1}/{samp_number}] has been finished, time={time.time()-start_time}')
        if number_category==3:
            names=['Light','Medium','Heavy']
            df=pd.DataFrame({'x':cp_data[samps_number[i]:samps_number[i+1],0],'light':cp_data[samps_number[i]:samps_number[i+1],1],
            'medium':cp_data[samps_number[i]:samps_number[i+1],2],'heavy':cp_data[samps_number[i]:samps_number[i+1],3],
            'acc':cp_data[samps_number[i]:samps_number[i+1],4]})
            start_time=time.time()
            for t in range(samps_number[i+1]-samps_number[i]):
                x=df['x'].to_numpy()[:t+1]
                light=df['light'].to_numpy()[:t+1]
                medium=df['medium'].to_numpy()[:t+1]
                heavy=df['heavy'].to_numpy()[:t+1]
                acc=df['acc'].to_numpy()[:t+1]
                plt.figure()
                subplot=plt.subplot()
                subplot.plot(x,light,color='red',label='light')
                subplot.plot(x,medium,color='blue',label='medium')
                subplot.plot(x,heavy,color='green',label='heavy')
                subplot.set_xlabel('Input')
                subplot.set_ylabel('Decision Distance')
                subplot2=subplot.twinx()
                subplot2.plot(x,acc,color='black',label='Accuracy',linestyle='--')
                subplot2.set_ylabel('Accuracy(%)')
                plt.legend()
                plt.title(names[i])
                plt.xlim([1,samps_number[i+1]-samps_number[i]])
                plt.savefig('./cp_anime/weight/'+names[i]+'_'+str(t+1).zfill(4)+'.png')
                plt.close()
                if t%int((samps_number[i+1]-samps_number[i])/10)==0:
                    print (f'[{names[i]}][{t+1}/{samps_number[i+1]-samps_number[i]}] has been finished, time={time.time()-start_time}')

batch_size=32
kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
model=torch.load(model_path+'robot_depth-v9.pth')
file_path='./Database/depth'
data='/'
csv_path='./explore.csv'
dataset=Bayesian_Dataset(file_path+data,csv_path,transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]),opt=1)
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,**kwargs)
embeddings,labels=extract_embeddings(dataloader,model)
plot_embeddings(embeddings,labels)
standard_label(embeddings,labels,numbers=10)
universe_data=pd.read_csv('./standard_labels.csv')
standard_labels=np.zeros((5,2))
test_labels=np.zeros((5,2))
standard_std=np.zeros((5,2))
test_std=np.zeros((5,2))
#standard_labels=np.zeros((3,2))
#test_labels=np.zeros((3,2))
#standard_std=np.zeros((3,2))
#test_std=np.zeros((3,2))
for i in range (len(standard_labels)):
    for j in range (len(standard_labels[i])):
        standard_labels[i][j]=universe_data.iloc[i,j+1]
        standard_std[i][j]=universe_data.iloc[i,j+3]
for i in range (len(test_labels)):
    for j in range (len(test_labels[i])):
        test_labels[i][j]=universe_data.iloc[i+5,j+1]
        test_std[i][j]=universe_data.iloc[i+5,j+3]
#for i in range (len(standard_labels)):
#    for j in range (len(standard_labels[i])):
#        standard_labels[i][j]=universe_data.iloc[i,j+1]
#        standard_std[i][j]=universe_data.iloc[i,j+3]
#for i in range (len(test_labels)):
#    for j in range (len(test_labels[i])):
#       test_labels[i][j]=universe_data.iloc[i+3,j+1]
#       test_std[i][j]=universe_data.iloc[i+3,j+3]
print ('standard_labels:',standard_labels)
print ('test_labels:',test_labels)
cp_drawing(embeddings,labels,standard_labels,test_labels,standard_std,test_std,number=5)
cp_data=pd.read_csv('./cp_tracking.csv').to_numpy()
cp_plotting(cp_data,number_category=5)
#cp_anime(cp_data,number_category=5)