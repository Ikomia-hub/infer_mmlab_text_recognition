import numpy as np


def polygon2bbox(pts):
    x = np.min(pts[:, 0])
    y = np.min(pts[:, 1])
    w = np.max(pts[:, 0]) - x
    h = np.max(pts[:, 1]) - y
    return [int(x), int(y), int(w), int(h)]


def bbox2polygon(box):
    x, y, w, h = box
    # starting with left bottom point then anti clockwise
    return [x, y + h, x + w, y + h, x + w, y, x, y]
