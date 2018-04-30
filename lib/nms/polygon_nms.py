from shapely.geometry import Polygon
from shapely.geometry import MultiPoint

import numpy as np

def py_polygon_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms

def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 1
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def get_polygon_areas(dets):
    num = dets.shape[0]
    polygon_points = dets.reshape(-1, 4, 2)
    areas = []
    for i in xrange(num):
        # x1 = dets[i, 0]
        # y1 = dets[i, 1]
        # x2 = dets[i, 2]
        # y2 = dets[i, 3]
        # x3 = dets[i, 4]
        # y3 = dets[i, 5]
        # x4 = dets[i, 6]
        # y4 = dets[i, 7]
        # polygon_points = np.array([x1, x2, x3,])
        area = Polygon(polygon_points[i]).convex_hull.area
        areas.append(area)
    return np.array(areas)

def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2, x3, y3, x4, y4 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    x3 = dets[:, 4]
    y3 = dets[:, 5]
    x4 = dets[:, 6]
    y4 = dets[:, 7]
    scores = dets[:, 8]

    areas = get_polygon_areas(dets[:, 0:8])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])
        xx3 = np.minimum(x3[i], x3[order[1:]])
        yy3 = np.minimum(y3[i], y3[order[1:]])
        xx4 = np.maximum(x4[i], x4[order[1:]])
        yy4 = np.minimum(y4[i], y4[order[1:]])

        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        # inter = w * h
        inter = []
        for h in xrange(xx1.shape[0]):
            inter.append(Polygon([(xx1[h], yy1[h]), (xx2[h], yy2[h]), (xx3[h], yy3[h]), (xx4[h], yy4[h])]).convex_hull.area)
        inter = np.array(inter)
        # inter = Polygon([(xx1, yy1), (xx2, yy2), (xx3, yy3), (xx4, yy4)]).area
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
