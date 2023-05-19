import torch
from torch import tensor
from torchvision.ops.boxes import box_iou

# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


###############################################################################
# from utils import get_intersection, get_union
# def IoU(bbox1, bbox2):
#     _, int_area = get_intersection(bbox1, bbox2)
#     _, union_area = get_union(bbox1, bbox2)
#     return int_area / union_area

# def AP(predictions, labels, threshold):
#     lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
#     pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')

#     for prediction, gt in zip(prediction, labels):
#         # 11-points AP
#         AP = 0
#         for i in range(0, 1, 0.1):
#             AP += 
#         sample_lbls = np.unique(gt['labels'].numpy())
###############################################################################

# MAP
# From https://www.kaggle.com/code/kshitijpatil09/pytorch-mean-absolute-precision-calculation/notebook?scriptVersionId=40816383

def align_coordinates(boxes):
    """Align coordinates (x1,y1) < (x2,y2) to work with torchvision `box_iou` op
    Arguments:
        boxes (Tensor[N,4])
    
    Returns:
        boxes (Tensor[N,4]): aligned box coordinates
    """
    x1y1 = torch.min(boxes[:,:2,],boxes[:, 2:])
    x2y2 = torch.max(boxes[:,:2,],boxes[:, 2:])
    boxes = torch.cat([x1y1,x2y2],dim=1)
    return boxes


def calculate_iou(gt, pr, form='pascal_voc'):
    """Calculates the Intersection over Union.

    Arguments:
        gt: (torch.Tensor[N,4]) coordinates of the ground-truth boxes
        pr: (torch.Tensor[M,4]) coordinates of the prdicted boxes
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
        IoU values for every element in boxes1 and boxes2
    """
    if form == 'coco':
        gt = gt.clone()
        pr = pr.clone()

        gt[:,2] = gt[:,0] + gt[:,2]
        gt[:,3] = gt[:,1] + gt[:,3]
        pr[:,2] = pr[:,0] + pr[:,2]
        pr[:,3] = pr[:,1] + pr[:,3]

    gt = align_coordinates(gt)
    pr = align_coordinates(pr)
    
    return box_iou(gt,pr)

def get_mappings(iou_mat, pr_count):  # si hay 2 predicciones, elige la mejor
    mappings = torch.zeros_like(iou_mat)

    if pr_count == 0:
        return mappings

    #first mapping (max iou for first pred_box)
    if not iou_mat[:,0].eq(0.).all():
        # if not a zero column
        mappings[iou_mat[:,0].argsort()[-1],0] = 1

    for pr_idx in range(1, pr_count):
        # Sum of all the previous mapping columns will let 
        # us know which gt-boxes are already assigned
        not_assigned = torch.logical_not(mappings[:,:pr_idx].sum(1)).long()

        # Considering unassigned gt-boxes for further evaluation 
        targets = not_assigned * iou_mat[:,pr_idx]

        # If no gt-box satisfy the previous conditions
        # for the current pred-box, ignore it (False Positive)
        if targets.eq(0).all():
            continue

        # max-iou from current column after all the filtering
        # will be the pivot element for mapping
        pivot = targets.argsort()[-1]
        mappings[pivot,pr_idx] = 1
    return mappings

def calculate_map(gt_boxes, pr_boxes, scores=None, thresh=0.5, form='pascal_voc'):
    # sorting
    if scores is not None:
        pr_boxes = pr_boxes[scores.argsort().flip(-1)]
    iou_mat = calculate_iou(gt_boxes, pr_boxes, form)
    gt_count, pr_count = iou_mat.shape
    
    # thresholding
    iou_mat = iou_mat.where(iou_mat > thresh, tensor(0.).to(device))
    
    mappings = get_mappings(iou_mat, pr_count)

    # mAP calculation
    tp = mappings.sum()
    fp = mappings.sum(0).eq(0).sum()
    fn = mappings.sum(1).eq(0).sum()
    mAP = tp / (tp+fp+fn)
    
    return mAP