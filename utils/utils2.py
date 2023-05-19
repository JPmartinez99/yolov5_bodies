from shapely.geometry import Polygon

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

def get_union(bbox1, bbox2):
    # Convert the bounding box to a Polygon object
    rect1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[2], bbox1[1]), (bbox1[2], bbox1[3]), (bbox1[0], bbox1[3])])
    rect2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[2], bbox2[1]), (bbox2[2], bbox2[3]), (bbox2[0], bbox2[3])])

    union = rect1.union(rect2)
    return union, union.area