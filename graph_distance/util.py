def iou(bb1, bb2):
    x11, y11, x12, y12 = bb1
    x21, y21, x22, y22 = bb2
    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    intersection = x_overlap * y_overlap
    union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - intersection
    iou = intersection / union
    return iou
