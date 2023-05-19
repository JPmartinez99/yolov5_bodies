import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from sklearn.utils import shuffle
import random

random.seed(42)

path_images = '/home/juanpe/datasets/SyntheticBodiesAtSea_yolo/images/training/'
path_labels = '/home/juanpe/datasets/SyntheticBodiesAtSea_yolo/annotations/training'
DEBUG_IMAGES = False
SAVE_PATCHES = True
AREA_FACTOR = 0.2 # 20%

# --------------------------------------------------------------------------
def intersects(bbox1, bbox2):
    return (bbox1[2] >= bbox2[0] and bbox2[2] >= bbox1[0]) and \
           (bbox1[3] >= bbox2[1] and bbox2[3] >= bbox1[1])

def get_intersection(bbox1, bbox2, th_area=-1):
    if intersects(bbox1, bbox2):

        # Get the intersection of the rectangles
        xmin = max(bbox1[0], bbox2[0])
        ymin = max(bbox1[1], bbox2[1])
        xmax = min(bbox1[2], bbox2[2])-1
        ymax = min(bbox1[3], bbox2[3])-1
        
        intersection = [xmin, ymin, xmax, ymax]
        area = (xmax - xmin) * (ymax - ymin)

        if th_area == -1 or area > th_area:
            return intersection, area

    return None, None

# ----------------------------------------------------------------------------
# Pasar el bbox de coordenadas de la imagen a coordenadas del patch
def localize_bbox(original_bbox, target_bbox):  
    return [target_bbox[0]-original_bbox[0], target_bbox[1]-original_bbox[1],
            target_bbox[2]-original_bbox[0], target_bbox[3]-original_bbox[1]]

# ----------------------------------------------------------------------------

# Recibe como parametro un patch de la imagen, las anotaciones (bboxes) de la imagen y el tamaño de ventana
# Calcula los bbox que hay en el patch, y los convierte a coordenadas del patch
def get_patch_bboxes(patch_bbox, ann_bboxes, window):    #reajuste de los bounding box de las ventanas
    overlaid_bboxes = [] # bboxes del patch
    overlaid_areas  = [] # areas de los bboxes
    for ann_bbox in ann_bboxes:
        intersection_bbox, area_bbox = get_intersection(patch_bbox, ann_bbox, th_area=window*AREA_FACTOR) # devuelve la interseccion entre ese bbox y el patch, y  el area
        if intersection_bbox is not None: overlaid_bboxes.append(intersection_bbox)
        if area_bbox is not None: overlaid_areas.append(area_bbox)

    return [localize_bbox(patch_bbox, ann_bbox) for ann_bbox in overlaid_bboxes], overlaid_areas
    # return overlaid_bboxes, overlaid_areas

# ----------------------------------------------------------------------------

def get_random_patches(img, bodies_annotations, nb_patches, window_size, prob_empty):
    patches_X = []
    patches_y = []
    max_height = img.shape[0] - window_size - 1 # para que el patch aleatorio no se salga de la imagen
    max_width = img.shape[1] - window_size - 1
    nb_bodies = len(bodies_annotations)

    for i in range(nb_patches):
        prob = random.random()
        print("prob: ", prob)
        if(prob < prob_empty):     # Completely random window (0.0 <= n < 1.0)
            x = random.randint(0, max_width)
            y = random.randint(0, max_height)
        else:
            bid = random.randint(0, nb_bodies-1)
            bbox = bodies_annotations[bid]
            print("bbox: ", bbox)
            #(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])     # x, y
            min_xbb = int(max(0, bbox[2] - window_size))
            min_ybb = int(max(0, bbox[3] - window_size))
            max_xbb = int(min(max_width, bbox[0]))
            max_ybb = int(min(max_height, bbox[1]))
            print("min_xbb: {}, min_ybb: {}, max_xbb: {}, max_ybb: {}".format(min_xbb, min_ybb, max_xbb, max_ybb))
            #print(min_xbb, max_xbb, ' - ', min_ybb, max_ybb)
            x = random.randint(min_xbb, max_xbb) if min_xbb < max_xbb else max_xbb
            y = random.randint(min_ybb, max_ybb) if min_ybb < max_ybb else max_ybb
            print("Coord patch: x1: {}, y1: {}, x2: {}, y2: {}".format(x,y, x+window_size, y+window_size))
        patch = img[y:y+window_size, x:x+window_size]
        patch_bbox = [x, y, x + window_size, y + window_size]
        boxes, areas = get_patch_bboxes(patch_bbox, bodies_annotations, 256)
        patches_X.append( img[y:y+window_size, x:x+window_size].copy() )
        patches_y.append(boxes)
    return patches_X, patches_y

# ----------------------------------------------------------------------------

def __generate_patches(path_image):
        
        # Get image info
        image  = cv2.imread(path_image)
        assert image is not None
        try:
            height, width, channels = image.shape
        except:
            print('no shape info.')
            return 0
        
        # Get label path
        labels = []     
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        label_path = sb.join(path_image.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'

        with open (label_path) as f:
            for line in f:
                line = list(map(float, line.split()))
                labels.append(line)
        
        # Bounding box coordinates from yolo format
        bboxes = []
        for label in labels:
            x_center, y_center, w, h = float(label[1])*width, float(label[2])*height, float(label[3])*width, float(label[4])*height
            x1 = round(x_center-w/2)
            y1 = round(y_center-h/2)
            x2 = round(x_center+w/2)
            y2 = round(y_center+h/2)  
            bboxes.append([x1,y1,x2,y2])
        

        '''if DEBUG_IMAGES:
            print("Showing input image...", path_image)
            cv2.imshow("Input image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

        patches_X, patches_y = get_random_patches(image,
                                                                bodies_annotations=bboxes,
                                                                nb_patches=8,
                                                                window_size=256,
                                                                prob_empty=0.05)

        
        if DEBUG_IMAGES:
            print('Showing obtained patches...')
            print("Patches_y: ", patches_y)
            print("Patches_X: ", patches_X)
            for i in range(len(patches_X)):
                cv2.imshow("img_x", patches_X[i])
                cv2.waitKey(0)
            cv2.destroyAllWindows()

        patches_X = np.asarray(patches_X).astype('float32')

        if SAVE_PATCHES:
            if not os.path.exists('./patches_tests'):
                os.mkdir('patches_tests')
            for i, patch in enumerate(patches_X):
                cv2.imwrite(f"patches_tests/{i}_test_x.png", patch)
        
        print("Patches_y: ", patches_y)
        for i in range(0,len(patches_X)):
            print("------------------------------------------------------------")
            print("Patches_X: ", patches_X)
        print("Patches_X shape: ", patches_X[0].shape)
        print("Len Patches_X shape: ", len(patches_X))
        
        return patches_X
    
    
path_image = os.path.join(path_images, 'img46.png')    
__generate_patches(path_image)