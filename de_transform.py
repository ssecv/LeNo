import os
import cv2

root='./data/ECSSD/AS'
save='./data/ECSSD/newAS'
mean,std=[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
for root, dirs, files in os.walk(root):
        for file in files:
                # filename = os.path.splitext()[0]
                img=cv2.imread(root+'/'+file)
                a=img/255
                a=a*std+mean
                cv2.imwrite(root+'/'+file,a)