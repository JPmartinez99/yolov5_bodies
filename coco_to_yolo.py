import json
import cv2
import os
import matplotlib.pyplot as plt
import shutil

# Paths of the datasets
IMAGES_FOLDER = "/home/juanpe/datasets/SyntheticBodiesAtSea_resize1200/images"
ANNOTATION_FOLDER = "/home/juanpe/datasets/SyntheticBodiesAtSea_resize1200/annotations"

YOLO_FOLDER = "/home/juanpe/datasets/SyntheticBodiesAtSea_resize1200_yolo"
nombre_lab = 's'


# Copy images to destination folder
def load_images_from_folder(folder, subset_img):
  count = 0
  for filename in os.listdir(folder):
        source = os.path.join(folder,filename)
        destination = f"{YOLO_FOLDER}/images/{subset_img}/{nombre_lab}{count}.png"

        try:
            shutil.copy(source, destination)
            print("File copied successfully.")
        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        file_names.append(filename)
        count += 1


# Return the annotation if it is in annotation file
def get_img_ann(image_id):
    img_ann = []
    isFound = False
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return None


# Return image_data if it is in annotation file    
def get_img(filename):
    for img in data['images']:
        if img['file_name'] == filename:
            return img
        
        
################################################################################################
################################################################################################
################################################################################################

file_names = []

for subset in os.listdir(ANNOTATION_FOLDER):
    print("PARTICION: ", subset)
    #Read json file
    ann_file_path = os.path.join(ANNOTATION_FOLDER, subset)
    f = open(ann_file_path)
    data = json.load(f)
    f.close()
    
    file_names = []
    subset_img = ""
    if(subset == "instances_train.json"):
        subset_img = "train"
    elif(subset == "instances_val.json"):
        subset_img = "val"
    else:
        subset_img = "test"
    
    partition_folder = os.path.join(IMAGES_FOLDER, subset_img)
    load_images_from_folder(partition_folder,subset_img)

    # Conversion
    count = 0
    for filename in file_names:
        # Extracting image 
        img = get_img(filename)
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']

        # Get Annotations for this image
        img_ann = get_img_ann(img_id)

        if img_ann:
            # Opening file for current image
            file_object = open(f"{YOLO_FOLDER}/labels/{subset_img}/{nombre_lab}{count}.txt", "a")

            for ann in img_ann:
                current_category = ann['category_id'] - 1 # As yolo format labels start from 0 
                current_bbox = ann['bbox']
                x = current_bbox[0]
                y = current_bbox[1]
                w = current_bbox[2]
                h = current_bbox[3]
                
                # Finding midpoints
                x_centre = (x + (x+w))/2
                y_centre = (y + (y+h))/2
                
                # Normalization
                x_centre = x_centre / img_w
                y_centre = y_centre / img_h
                w = w / img_w
                h = h / img_h
                
                # Limiting upto fix number of decimal places
                x_centre = format(x_centre, '.6f')
                y_centre = format(y_centre, '.6f')
                w = format(w, '.6f')
                h = format(h, '.6f')
                    
                # Writing current object 
                file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

            file_object.close()
            count += 1  # This should be outside the if img_ann block.