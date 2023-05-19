import cv2
import numpy as np
from PIL import Image
import os



path_label = os.path.join('/home/juanpe/datasets/SeaDronesSee_yolo/labels', 'training', 'r2.txt')
labels = []     
with open (path_label) as f:
    for line in f:
        print("line: ", line)
        line = list(map(float, line.split()))
        print("line map: ", line)
        labels.append(line)
    print("labels: ", labels)