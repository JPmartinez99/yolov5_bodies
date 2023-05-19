
import os
import sys
import numpy as np

import cv2

from PIL import Image
from sklearn.utils import shuffle
import json

PATH_REAL = "/home/juanpe/datasets/SeaDronesSee"
PATH_SYN = "/home/juanpe/datasets/SyntheticBodiesAtSea"
folder = 'annotations'

    
dir_ann = os.path.join(PATH_REAL, folder)
for file in os.listdir(dir_ann):
    print("file: ", file)
    f = open(os.path.join(dir_ann, file))
    data = json.load(f)
    f.close()

data_images = data['images']
data_ann = data['annotations']
'''
n=0
for i in data_images:
    image = cv2.imread(os.path.join(PATH_REAL, 'images', 'train', i['file_name']))
    print("{} - Imagen: {}".format(n,i['file_name']))
    n_ann = 0
    n+=1
    for j in data_ann:
        #print("Anotacion: ", j)
        
        ann_name = str(j['image_id']) + '.png'
        if(i['file_name'] == ann_name):
            n_ann+=1
            print("Ann: ", j['bbox'])
            cv2.rectangle(image, (j['bbox'][0],j['bbox'][1]), (j['bbox'][0]+j['bbox'][2], j['bbox'][1]+j['bbox'][3]), (0, 0, 255), 2)
    dim = (1424, 960)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print("N annotaciones: ", n_ann)
    cv2.imshow("image", image)
    cv2.waitKey(0)
'''
im='1046.png'
image = cv2.imread(os.path.join(PATH_REAL, 'images', 'test', im))
n_ann = 0
for j in data_ann:
        ann_name = str(j['image_id']) + '.png'
        if(im == ann_name):
            n_ann+=1
            print("Ann: ", j['bbox'])
            cv2.rectangle(image, (j['bbox'][0],j['bbox'][1]), (j['bbox'][0]+j['bbox'][2], j['bbox'][1]+j['bbox'][3]), (0, 0, 255), 2)
dim = (1424, 960)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
print("N annotaciones: ", n_ann)
cv2.imshow("image", image)
cv2.waitKey(0)




