import os
import os.path as osp
import math
import numpy as np


def point_in_polygon(x, y, verts):
    try:
        x, y = float(x), float(y)
    except:
        return False
    vertx = [xyvert[0] for xyvert in verts]
    verty = [xyvert[1] for xyvert in verts]

    if not verts or not min(vertx) <= x <= max(vertx) or not min(verty) <= y <= max(verty):
        return False

    nvert = len(verts)
    is_in = False
    for i in range(nvert):
        j = nvert - 1 if i == 0 else i - 1
        if ((verty[i] > y) != (verty[j] > y)) and (
                    x < (vertx[j] - vertx[i]) * (y - verty[i]) / (verty[j] - verty[i]) + vertx[i]):
            is_in = not is_in

    return is_in


def isInsidePolygon(pt, poly):
    c = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l - 1:
        i += 1
        if ((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or (
                        poly[j][0] <= pt[0] and pt[0] < poly[i][0])):
            if (pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (
                        poly[j][0] - poly[i][0]) + poly[i][1]):
                c = not c
        j = i
    return c


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(%d, %d)" % (self.x, self.y)

    def line2(self, another):
        if self.x == another.x:
            step = 1 if self.y < another.y else -1
            y = self.y
            while y != another.y:
                yield Point(self.x, y)
                y += step
        elif self.y == another.y:
            step = 1 if self.x < another.x else -1
            x = self.x
            while x != another.x:
                yield Point(x, self.y)
                x += step
        else:
            d_x = self.x - another.x
            d_y = self.y - another.y
            s_x = 1 if d_x < 0 else -1
            s_y = 1 if d_y < 0 else -1

            if d_y:
                delta = 1. * d_x / d_y
                for i in xrange(0, d_x):
                    yield Point(self.x + i * s_x, self.y + i * s_x / delta)
            elif d_x:
                delta = 1. * d_y / d_x
                for i in xrange(0, d_y):
                    yield Point(self.y + i * s_y / delta, self.y + i * s_y)


def getPointsFromFile(contents):
    Points = []
    for content in contents:
        s = content.split(',')
        x1, y1, x2, y2 = [int(i) for i in s]
        Points.append(Point(x1, y1))
        Points.append(Point(x2, y2))
    return Points


def getCenterPoint(coordinate1, coordinate2):
    return Point((coordinate1.x + coordinate2.x) / 2, (coordinate1.y + coordinate2.y) / 2)


def getDistance(point1, point2):
    return math.sqrt(math.pow(point1.x - point2.x, 2) + math.pow(point1.y - point2.y, 2))


def getVector(point1, point2):
    distance = getDistance(point1, point2)
    theta = math.atan(math.fabs(point1.y - point2.y) / math.fabs(point1.x - point2.x))
    return (distance * math.cos(theta), distance * math.sin(theta))


def getAllPointsAlongLine(point1, point2):
    all_points = []
    for point in point1.line2(point2):
        all_points.append(point)
    return all_points


def getAllPointsInQuadrangles(box):
    xmin = min([box[i][0] for i in range(4)])
    xmax = max([box[i][0] for i in range(4)])
    ymin = min([box[i][1] for i in range(4)])
    ymax = max([box[i][1] for i in range(4)])
    recPoints = [(xk, yk) for xk in range(xmin, xmax + 1) for yk in range(ymin, ymax + 1) if
                 isInsidePolygon((xk, yk), box)]
    return recPoints


def getScore(direction_vector, along_line_points_vectors):
    all_score = 0.0
    length = len(along_line_points_vectors)
    for vector in along_line_points_vectors:
        all_score = all_score + direction_vector[0] * vector[0] + direction_vector[1] * vector[1]
    return all_score / length


def getPointsVectorsFromMap(along_line_points, segmentation_map):
    results = []
    for point in along_line_points:
        results.append(segmentation_map[point.x][point.y])
    return results


def getLongSide(point1, point2, point3, point4):
    center1 = getCenterPoint(point1, point2)
    center2 = getCenterPoint(point2, point3)
    center3 = getCenterPoint(point3, point4)
    center4 = getCenterPoint(point4, point1)
    distance1 = getDistance(center1, center3)
    distance2 = getDistance(center2, center4)
    if distance1 <= distance2:
        return center4, center2, getVector(center4, center2)
    else:
        return center1, center3, getVector(center1, center3)


def getVectorMap(width, height, boxes):
    '''
    This function is used to calculate gt Vector maps
    :param width: 
    :param height: 
    :param boxes: 
    :return: 
    '''
    maps = np.zeros((width, height, 2), dtype=np.float64)
    for box in boxes:
        _, _, vector = getLongSide(box[0], box[1], box[2], box[3])
        pointsInQuadrangle = getAllPointsInQuadrangles(box)
        for point in pointsInQuadrangle:
            if maps[point[0]][point[1]][0] == maps[point[0]][point[1]][1] == 0.0:
                maps[point[0]][point[1]][0] = vector[0]
                maps[point[0]][point[1]][1] = vector[1]
            else:
                # To get thoughrouh vector in cross area
                maps[point[0]][point[1]][0] = (maps[point[0]][point[1]][0] + vector[0]) / 2.0
                maps[point[0]][point[1]][1] = (maps[point[0]][point[1]][1] + vector[1]) / 2.0
    return maps


def getCenterPointsFromFile(contents):
    centerPoints = []
    for content in contents:
        s = content.split(',')
        x1, y1, x2, y2 = [int(i) for i in s]
        centerPoints.append(getCenterPoint(Point(x1, y1), Point(x2, y2)))
    return centerPoints


def longtail(file_path, segmentation_map):
    '''
    :param segmentation_map: (width, height, 2), 2 means the normalized vector of x and y
    :param file_path: path of the file, file goes like:
        x1, y1, x2, y2
        x1, y1, x2, y2
        ....
    :return: the corresponding pair of points with score, [[Point_1, Point_2, Point_3, Point_4, score], ...]
    '''
    with open(file_path, 'r') as f:
        contents = f.readlines()
    contents = [content.strip() for content in contents]
    points = getPointsFromFile(contents)
    centerPoints = getCenterPointsFromFile(contents)
    corresponding_points = []
    for i in range(0, len(centerPoints)):
        for j in range(i, len(centerPoints)):
            line_vector = getVector(centerPoints[i], centerPoints[j])
            all_points_along_line = getAllPointsAlongLine(centerPoints[i], centerPoints[j])
            all_vectors_along_line = getPointsVectorsFromMap(all_points_along_line, segmentation_map)
            score = getScore(line_vector, all_vectors_along_line)
            corresponding_points.append([points[i], points[i+1], points[j], points[j+1], score])
    sorted_points_pairs = sorted(corresponding_points, key=lambda x:x[4])
    return sorted_points_pairs

if __name__ == '__main__':
    print isInsidePolygon((2, 2), [(0, 0), (2, 0), (2, 2), (0, 2)])
    print getAllPointsInQuadrangles([(0, 0), (2, 0), (4, 4), (0, 2)])
