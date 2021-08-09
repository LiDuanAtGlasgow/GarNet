#type:ignore
import csv
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import glob
import os
import time

train_f=open('./train_label.csv','w')
train_csv_writer=csv.writer(train_f)
train_csv_writer.writerow(('name','shape','weight','video_no'))
test_f=open('./test_label.csv','w')
test_csv_writer=csv.writer(test_f)
test_csv_writer.writerow(('name','shape','weight','video_no'))

names=['jeans','shirts','sweaters','towels','t-shirts']
num_garments=4
num_videos=10
num_frames=60
source_file='./mask'
target_file='./rgb/'
database_file='./Database/rgb/'
if not os.path.exists(database_file):
    os.makedirs(database_file)

start_time=time.time()
for name_idx in range (len(names)):
    for garment_idx in range (num_garments):
        for video_idx in range (num_videos):
            files=source_file+'/'+str(names[name_idx])+'/'+str(names[name_idx])+'_'+str(garment_idx+1)+'/video_'+str(video_idx+1).zfill(2)+'/'
            aim_folder=target_file+'robot_mask/'+str(names[name_idx])+'/'+str(names[name_idx])+'_'+str(garment_idx+1)+'/video_'+str(video_idx+1).zfill(2)+'/'
            if not os.path.exists(aim_folder):
                os.makedirs(aim_folder)
            count=0
            #print ('files:',files)
            for file in sorted(glob.glob(files+'*.png'),key=str.lower):
                #print ('file:',file)
                count+=1
                image=cv2.imread(file)
                cv2.imwrite(aim_folder+str(count).zfill(4)+'.png',image)
    print ('Category',names[name_idx],'has been finished, time:',str(time.time()-start_time))
    start_time=time.time()
'''
start_time=time.time()
for name_idx in range (len(names)):
    for garment_idx in range (num_garments-1):
        for video_idx in range (num_videos):
            for frame_idx in range (num_frames):
                image_dir='./Sources/rgb/'+str(names[name_idx])+'/'+str(names[name_idx])+'_'+str(garment_idx+1)+'/video_'+str(video_idx+1).zfill(2)+'/'+str(frame_idx+1).zfill(4)+'.png'
                #print ('image_dir:',image_dir)
                image=cv2.imread(image_dir)
                name=str(names[name_idx])+'_'+str(garment_idx+1)+'_video_'+str(video_idx+1).zfill(2)+'_'+str(frame_idx+1).zfill(4)+'.png'
                video_no=video_idx+1
                shape=name_idx+1
                if name_idx==0 or name_idx==2:
                    weight=3
                elif name_idx==1 or name_idx==4:
                    weight=2
                elif name_idx==3:
                    weight=1
                else:
                    print ('weight errors!')
                train_csv_writer.writerow((name,shape,weight,video_no))
                cv2.imwrite(database_file+name,image)
    print ('[Train] Category',names[name_idx],'has been finished, time:',str(time.time()-start_time))
    start_time=time.time()
'''
'''
start_time=time.time()
for name_idx in range (len(names)):
    for garment_idx in range (1):
        for video_idx in range (num_videos):
            for frame_idx in range (num_frames):
                image_dir='./Sources/rgb/'+str(names[name_idx])+'/'+str(names[name_idx])+'_'+str(4)+'/video_'+str(video_idx+1).zfill(2)+'/'+str(frame_idx+1).zfill(4)+'.png'
                #print ('image_dir:',image_dir)
                image=cv2.imread(image_dir)
                name=str(names[name_idx])+'_'+str(4)+'_video_'+str(video_idx+1).zfill(2)+'_'+str(frame_idx+1).zfill(4)+'.png'
                video_no=(name_idx+1)*40+garment_idx*10+video_idx+1
                shape=name_idx+1+5
                if name_idx==0 or name_idx==2:
                    weight=3+3
                elif name_idx==1 or name_idx==4:
                    weight=2+3
                elif name_idx==3:
                    weight=1+3
                else:
                    print ('weight errors!')
                test_csv_writer.writerow((name,shape,weight,video_no))
                cv2.imwrite(database_file+name,image)
    print ('[Test] Category',names[name_idx],'has been finished, time:',str(time.time()-start_time))
    start_time=time.time()
'''
print ('--finished!--')