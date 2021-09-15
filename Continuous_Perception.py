#type:ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import scipy.stats
from shapely import geometry
import os

def continuous_perception(embedding,labels,video_labels,n=5):
    bandwidth_value=90
    plt.figure(figsize=(10,10))
    num_video=50
    len_video=60
    if n==5:
        color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
        label_plottings=['pant','shirt','sweater','towel','tshirt']
    if n==3:
        color=['#1f77b4','#ff7f01','#2ca02c','#1f77b4','#ff7f01','#2ca02c']
        label_plottings=['light','medium','heavy']
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
        q,w=np.meshgrid(range(-40,60,1), range(-40,40,1))
        r=pdf([q.flatten(),w.flatten()])
        s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), bandwidth_value)
        r.shape=(80,100)
        cont=plt.contour(range(-40,60,1), range(-40,40,1), r, [s],colors=color[i])
        cont_location=[]
        for line in cont.collections[0].get_paths():
            cont_location.append(line.vertices)
        cont_location=np.array(cont_location)[0]
        contours.append(cont_location)
        plt.plot(x,y,'o',color=color[i],label=label_plottings[i])
        standard_points[i,:]=(x,y)
    plt.show()

    start_time=time.time()
    i=0
    for video_idx in range (num_video):
        if n==5:
            if video_idx+1<=10:
                video_idx=video_idx+41
                category_idx=0
            elif video_idx+1>10 and video_idx+1<=20:
                video_idx=video_idx+71
                category_idx=1
            elif video_idx+1>20 and video_idx+1<=30:
                video_idx=video_idx+101
                category_idx=2
            elif video_idx+1>30 and video_idx+1<=40:
                video_idx=video_idx+131
                category_idx=3
            elif video_idx+1>40:
                video_idx=video_idx+161
                category_idx=4
        if n==3:
            if video_idx+1<=10:
                video_idx=video_idx+41
                category_idx=2
            elif video_idx+1>10 and video_idx+1<=20:
                video_idx=video_idx+71
                category_idx=1
            elif video_idx+1>20 and video_idx+1<=30:
                video_idx=video_idx+101
                category_idx=2
            elif video_idx+1>30 and video_idx+1<=40:
                video_idx=video_idx+131
                category_idx=0
            elif video_idx+1>40:
                video_idx=video_idx+161
                category_idx=1
        acc=0
        total=0
        inds=np.where(video_labels==video_idx)[0]
        video_data=embedding[inds]*10
        for idx in range (len_video):
            i+=1
            x=video_data[:idx+1,0].mean()
            y=video_data[:idx+1,1].mean()
            #x=video_data[idx,0]
            #y=video_data[idx,1]
            mean_value_point=np.array([x,y])
            contains_acc=[]
            for m in range (n):
                line = geometry.LineString(contours[m])
                point = geometry.Point(x,y)
                polygon = geometry.Polygon(line)
                if polygon.contains(point):
                    contains_acc.append(m)
            if len(contains_acc)==1:
                if contains_acc[0]==category_idx:
                    acc+=1
            elif len(contains_acc)>1:
                dists=np.zeros((len(contains_acc),2))
                for h in range(len(contains_acc)):
                    standard_point=standard_points[contains_acc[h]]
                    dis=np.sum(np.power(standard_point-mean_value_point,2))
                    dists[h,0]=dis
                    dists[h,1]=contains_acc[h]
                min_val=np.argmin(dists[:,0])
                if dists[min_val,1]==category_idx:
                    acc+=1
            total+=1
            accuracy=100*(acc/total)
            dists_for_tracking=[]
            for dist_idx in range (n):
                standard_point=standard_points[dist_idx]
                dist_for_tracking=np.sum(np.power(standard_point-mean_value_point,2))
                dists_for_tracking.append(dist_for_tracking)
            if n==5:
                csv_writer.writerow((i,dists_for_tracking[0],dists_for_tracking[1],dists_for_tracking[2],
                dists_for_tracking[3],dists_for_tracking[4],accuracy))
            if n==3:
                csv_writer.writerow((i,dists_for_tracking[0],dists_for_tracking[1],dists_for_tracking[2],
                accuracy))
        i=0
        accuracies=100*(acc/total)
        print ('[video_idx]',video_idx,'[Category]',label_plottings[category_idx],'[Accuracy]',str(accuracies),'[Time]', str(time.time()-start_time))
        start_time=time.time()

def continuous_perception_plotting(cp_data,samp_number=60,number_category=5):
    num_video=51
    if number_category==5:
        file_path='./continuous_perception/shape/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range (num_video):
                if i==50:
                    samp_number=20
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
                plt.title('video'+str(i).zfill(4))
                plt.savefig('./continuous_perception/shape/'+'video'+str(i).zfill(4)+'.png')
    if number_category==3:
        file_path='./continuous_perception/weight/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range (num_video):
            if i ==50:
                samp_number=20
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
            subplot2.plot('x','acc',data=df,color='grey',linewidth=10,label='Accuracy',linestyle='dotted')
            subplot2.set_ylabel('Accuracy(%)')
            subplot2.set_ylim([0,100])
            if i==50:
                subplot.set_xlim([1,20])
            plt.legend()
            plt.title('video'+str(i).zfill(4))
            plt.savefig('./continuous_perception/weight/'+'video'+str(i).zfill(4)+'.png')

def early_stop(embedding,labels,video_labels,n=5):
    bandwidth_value=5
    plt.figure(figsize=(10,10))
    num_video=50
    len_video=60
    if n==5:
        color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
        label_plottings=['pants','shirts','sweaters','towels','t-shirts']
    if n==3:
        color=['#1f77b4','#ff7f01','#2ca02c','#1f77b4','#ff7f01','#2ca02c']
        label_plottings=['lights','mediums','heavies']
    contours=[]
    standard_points=np.zeros((n,2))
    f=open('./garnet_cp_tracking.csv','w')
    csv_writer=csv.writer(f)
    if n==5:
        csv_writer.writerow(('no.','pants','shirts','sweaters','towels','t-shirts','accuracy'))
    if n==3:
        csv_writer.writerow(('no.','lights','mediums','heavies','accuracy'))
    for i in range (n):
        inds=np.where(labels==i+1)[0]
        data=embedding[inds]*10
        x=data[:,0].mean()
        y=data[:,1].mean()
        pdf=scipy.stats.kde.gaussian_kde(data.T)
        q,w=np.meshgrid(range(-40,60,1), range(-10,50,1))
        r=pdf([q.flatten(),w.flatten()])
        s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), bandwidth_value)
        r.shape=(60,100)
        cont=plt.contour(range(-40,60,1), range(-10,50,1), r, [s],colors=color[i])
        cont_location=[]
        for line in cont.collections[0].get_paths():
            cont_location.append(line.vertices)
        cont_location=np.array(cont_location)[0]
        contours.append(cont_location)
        plt.plot(x,y,'o',color=color[i],label=label_plottings[i])
        standard_points[i,:]=(x,y)
    plt.legend(loc='best')
    plt.show()

    start_time=time.time()
    i=0
    acc=0
    for video_idx in range (num_video):
        if n==5:
            if video_idx+1<=10:
                video_idx=video_idx+41
                category_idx=0
            elif video_idx+1>10 and video_idx+1<=20:
                video_idx=video_idx+71
                category_idx=1
            elif video_idx+1>20 and video_idx+1<=30:
                video_idx=video_idx+101
                category_idx=2
            elif video_idx+1>30 and video_idx+1<=40:
                video_idx=video_idx+131
                category_idx=3
            elif video_idx+1>40:
                video_idx=video_idx+161
                category_idx=4
        if n==3:
            if video_idx+1<=10:
                video_idx=video_idx+41
                category_idx=2
            elif video_idx+1>10 and video_idx+1<=20:
                video_idx=video_idx+71
                category_idx=1
            elif video_idx+1>20 and video_idx+1<=30:
                video_idx=video_idx+101
                category_idx=2
            elif video_idx+1>30 and video_idx+1<=40:
                video_idx=video_idx+131
                category_idx=0
            elif video_idx+1>40:
                video_idx=video_idx+161
                category_idx=1
        total=0
        inds=np.where(video_labels==video_idx)[0]
        video_data=embedding[inds]*10
        point_count=np.zeros(n)
        for idx in range (len_video):
            i+=1
            x=video_data[:idx+1,0].mean()
            y=video_data[:idx+1,1].mean()
            #x=video_data[idx,0]
            #y=video_data[idx,1]
            mean_value_point=np.array([x,y])
            contains_acc=[]
            for m in range (n):
                line = geometry.LineString(contours[m])
                point = geometry.Point(x,y)
                polygon = geometry.Polygon(line)
                if polygon.contains(point):
                    contains_acc.append(m)
            if len(contains_acc)==1:
                for select_idx in range (n):
                    if contains_acc[0]==select_idx:
                        point_count[select_idx]+=1
            elif len(contains_acc)>1:
                dists=np.zeros((len(contains_acc),2))
                for h in range(len(contains_acc)):
                    standard_point=standard_points[contains_acc[h]]
                    dis=np.sum(np.power(standard_point-mean_value_point,2))
                    dists[h,0]=dis
                    dists[h,1]=contains_acc[h]
                min_val=np.argmin(dists[:,0])
                for select_idx in range (n):
                    if dists[min_val,1]==select_idx:
                        point_count[select_idx]+=1
            total+=1
            dists_for_tracking=[]
            pred=100
            frame_counts=np.zeros(len(point_count))
            for select_idx in range (len(point_count)):
                count_percentage=point_count[select_idx]/total
                frame_counts[select_idx]=count_percentage*100
                if i>=20:
                    if count_percentage>=0.8:
                        pred=select_idx
            if n==5:
                if pred<100:
                    csv_writer.writerow((i,0,0,0,0,0,pred))
                else:
                    csv_writer.writerow((i,frame_counts[0],frame_counts[1],frame_counts[2],frame_counts[3],
                    frame_counts[4],pred))
            if n==3:
                if pred<100:
                    csv_writer.writerow((i,0,0,0,pred))
                else:
                    csv_writer.writerow((i,frame_counts[0],frame_counts[1],frame_counts[2],pred))
        i=0
        if pred<100:
            print ('[video_idx]',video_idx,'[Category]',label_plottings[category_idx],'[Pred]',str(label_plottings[pred]),'[Time]', str(time.time()-start_time))
        else:
            print ('[video_idx]',video_idx,'[Category]',label_plottings[category_idx],'[Pred]','Failed','[Time]', str(time.time()-start_time))
        start_time=time.time()

def early_stop_plotting(cp_data,samp_number=60,number_category=5):
    num_video=50
    if number_category==5:
        file_path='./continuous_perception/shape/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range (num_video):
                if i ==50: 
                    samp_number=20
                df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'pant':cp_data[i*samp_number:(i+1)*samp_number,1],
                'shirt':cp_data[i*samp_number:(i+1)*samp_number,2],'sweater':cp_data[i*samp_number:(i+1)*samp_number,3],
                'towel':cp_data[i*samp_number:(i+1)*samp_number,4],'tshirt':cp_data[i*samp_number:(i+1)*samp_number,5]})
                plt.figure()
                subplot=plt.subplot()
                subplot.plot('x','pant',data=df,color='red',label='pant')
                subplot.plot('x','shirt',data=df,color='blue',label='shirt')
                subplot.plot('x','sweater',data=df,color='green',label='sweater')
                subplot.plot('x','towel',data=df,color='purple',label='towel')
                subplot.plot('x','tshirt',data=df,color='gray',label='tshirt')
                subplot.set_xlabel('Input')
                subplot.set_ylabel('Percentage (%)')
                plt.legend()
                plt.title('video'+str(i).zfill(4))
                plt.savefig('./continuous_perception/shape/'+'video'+str(i).zfill(4)+'.png')
    if number_category==3:
        file_path='./continuous_perception/weight/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range (num_video):
            df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'light':cp_data[i*samp_number:(i+1)*samp_number,1],
                'medium':cp_data[i*samp_number:(i+1)*samp_number,2],'heavy':cp_data[i*samp_number:(i+1)*samp_number,3]})
            plt.figure()
            subplot=plt.subplot()
            subplot.plot('x','light',data=df,color='red',label='lights')
            subplot.plot('x','medium',data=df,color='blue',label='mediums')
            subplot.plot('x','heavy',data=df,color='green',label='heavies')
            subplot.set_xlabel('Input')
            subplot.set_ylabel('Percentage (%)')
            plt.legend()
            plt.title('video'+str(i).zfill(4))
            plt.savefig('./continuous_perception/weight/'+'video'+str(i).zfill(4)+'.png')

def early_stop_anime(cp_data,samp_number=60,number_category=5):
    num_video=50
    if number_category==5:
        file_path='./continuous_perception/anime/shape/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range (num_video):
                df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'pant':cp_data[i*samp_number:(i+1)*samp_number,1],
                'shirt':cp_data[i*samp_number:(i+1)*samp_number,2],'sweater':cp_data[i*samp_number:(i+1)*samp_number,3],
                'towel':cp_data[i*samp_number:(i+1)*samp_number,4],'tshirt':cp_data[i*samp_number:(i+1)*samp_number,5]})
                start_time=time.time()
                for t in range (samp_number):
                    x=df['x'].to_numpy()[:t+1]
                    pant=df['pant'].to_numpy()[:t+1]
                    shirt=df['shirt'].to_numpy()[:t+1]
                    sweater=df['sweater'].to_numpy()[:t+1]
                    towel=df['towel'].to_numpy()[:t+1]
                    tshirt=df['tshirt'].to_numpy()[:t+1]
                    plt.figure()
                    subplot=plt.subplot()
                    subplot.plot(x,pant,color='red',label='pant')
                    subplot.plot(x,shirt,color='blue',label='shirt')
                    subplot.plot(x,sweater,color='green',label='sweater')
                    subplot.plot(x,towel,color='purple',label='towel')
                    subplot.plot(x,tshirt,color='gray',label='tshirt')
                    subplot.set_xlabel('Input frame')
                    subplot.set_ylabel('Percentage (%)')
                    plt.legend()
                    #plt.title(names[i])
                    plt.xlim([1,samp_number])
                    plt.savefig('./continuous_perception/anime/shape/'+str(i+1).zfill(2)+'_video_'+str(i+1).zfill(2)+'_frame_'+str(t+1).zfill(4)+'.png')
                    plt.close()
                    if t%int(samp_number/10)==0:
                        print (f'[{str(i+1).zfill(2)}][{t+1}/{samp_number}] has been finished, time={time.time()-start_time}')
                        start_time=time.time()
    if number_category==3:
        file_path='./continuous_perception/anime/weight/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range (num_video):
            df=pd.DataFrame({'x':cp_data[i*samp_number:(i+1)*samp_number,0],'light':cp_data[i*samp_number:(i+1)*samp_number,1],
                'medium':cp_data[i*samp_number:(i+1)*samp_number,2],'heavy':cp_data[i*samp_number:(i+1)*samp_number,3]})
            start_time=time.time()
            for t in range(samp_number):
                x=df['x'].to_numpy()[:t+1]
                light=df['light'].to_numpy()[:t+1]
                medium=df['medium'].to_numpy()[:t+1]
                heavy=df['heavy'].to_numpy()[:t+1]
                plt.figure()
                subplot=plt.subplot()
                subplot.plot(x,light,color='red',label='light')
                subplot.plot(x,medium,color='blue',label='medium')
                subplot.plot(x,heavy,color='green',label='heavy')
                subplot.set_xlabel('Input')
                subplot.set_ylabel('Percentage (%)')
                plt.legend()
                #plt.title(names[i])
                plt.ylim([0,105])
                plt.xlim([1,samp_number])
                plt.savefig('./continuous_perception/anime/weight/'+str(i+1).zfill(2)+'_video_'+str(i+1).zfill(2)+'_frame_'+str(t+1).zfill(4)+'.png')
                plt.close()
                if t%int((samp_number/10))==0:
                    print (f'[{str(i+1).zfill(2)}][{t+1}/{samp_number}] has been finished, time={time.time()-start_time}')
