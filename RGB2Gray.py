#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import time

source_file='./Database/rgb/'
categories=['jeans','shirts','sweaters','towels','t-shirts']
num_garment=4
num_video=10
num_frame=60

start_time=time.time()
for category in categories:
    for garment_idx in range (num_garment):
        for video_idx in range (num_video):
            for frame_idx in range (num_frame):
                image_path=source_file+category+'_'+str(garment_idx+1)+'_video_'+str(video_idx+1).zfill(2)+'_'+str(frame_idx+1).zfill(4)+'.png'
                image=cv2.imread(image_path)
                gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                cv2.imwrite(image_path,gray_image)
    print ('[Category]',category,'has been finished, time:',str(time.time()-start_time))
    start_time=time.time()
print ('--finished!--')
    