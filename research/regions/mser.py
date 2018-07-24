import cv2
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

###
# IDEA:
# Step 1: Tresholding (adjusted to resolution)
#        --> Adjust to res (or fixed scaling)
# Step 2: Pruning of unlikely / too small samples
#        --> Fully left
# Step 3: Grouping of similar rectangles
# Step 4: Clustering by Y and X axis (differently)
#        --> Clustering by X (max difference: height of letter *2 or similar)
# Step 5: Bounding box of cluster
# Step 6: Non-max supression of clusters
#        --> Fully
# Step 7: Finito!


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
#
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


img = cv2.imread("./images/scan.jpg")
# img = cv2.resize(img, (800, 1200))
vis = img.copy()
# vis = cv2.resize(vis, (533, 800))


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean = np.mean(img, axis=(0, 1))
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 51, mean/4)

max_area = int(img.shape[0]*img.shape[1]/2)

mser = cv2.MSER_create(
    _max_area=int(img.shape[0]*img.shape[1]/2),
    _min_area=0)
regions, _ = mser.detectRegions(img)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]


# cv2.polylines(vis, hulls, 1, (0, 0, 255))
# cv2.imshow('img', vis)
# cv2.waitKey(0)

vis2 = vis.copy()
vis3 = vis.copy()
vis4 = vis.copy()
X = []
YsOnly = []
rects = []
for contour in hulls:
    # image_of_contour(vis, contour)
    # cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    x, y, w, h = cv2.boundingRect(contour)
    # M = cv2.moments(contour)
    # if M["m00"] != 0:
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     YsOnly.append([cY])
    #     X.append([cX, cY])
    # cv2.circle(vis, (cX, cY), 2, (0, 0, 255), -1)
    cv2.rectangle(vis3, (x, y), (x + w, y + h), (0, 0, 255), 1)
    rects.extend([[x-2, y-2, w+2, h+2]]*2)
show(vis3)
exit()

# rects = np.array(rects)
rects = list(rects)
# rects = non_max_suppression_fast(rects, 0.2)
rects = cv2.groupRectangles(rects, 1, 0)
for rect in rects[0]:
    x1, y1, x2, y2 = rect
    # X.append([int(x1 + (x2 - x1)/2), int(y1 + (y2 - y1) / 2)])
    # # X.append([x2, y2])
    # Y.append([int(y1 + (y2-y1)/2)])
    # Y.append([y2])
    cv2.rectangle(vis2, (x1, y1), (x1+x2, y1+y2), (0, 0, 255), 1)

cv2.imshow('img', vis2)
cv2.waitKey(0)

rects = [[x, y, x+w, y+h] for x, y, w, h in rects[0]]


def weighted_euclidian(wX, wY):
    def euclid_comp(a1, a2, b1, b2):
        return 0 if b1 < a1 < b2 or b1 < a2 < b2 else min(np.square(a1 - b1), np.square(a2 - b1),
                                                          np.square(a1 - b2), np.square(a2 - b2))

    def distance(rectA, rectB):
        ax1, ay1, ax2, ay2 = rectA
        bx1, by1, bx2, by2 = rectB
        return np.sqrt(wX*euclid_comp(ax1, ax2, bx1, bx2) + wY*euclid_comp(ay1, ay2, by1, by2))
    return distance


def axis_distance(p1, p2):
    def mid(a1, a2):
        return int(min(a1, a2) + abs(a1 - a2)/2)

    def distance(rectA, rectB):
        a1 = rectA[p1]
        a2 = rectA[p2]
        b1 = rectB[p1]
        b2 = rectB[p2]
        am = mid(a1, a2)
        bm = mid(b1, b2)
        return min(abs(a1 - b1), abs(a1 - b2), abs(a2 - b2), abs(a2 - b1), abs(am - bm))
    return distance


def y_overlap(rectA, rectB):

    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB
    avg_dist = (abs(ay1 - by1) + abs(ay2 - by2))/2
    avg_height = ((ay2 - ay1) + (by2 - by1)) / 2
    return avg_dist / avg_height


def x_dist(rectA, rectB):
    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB
    x_dist = min(abs(ax1 - bx2), abs(ax2 - bx1))
    avg_height = ((ay2 - ay1) + (by2 - by1)) / 2.0
    xd = x_dist / avg_height
    return xd


def squared_y_x(rectA, rectB):
    return np.sqrt(0.1*np.square(x_dist(rectA, rectB) + np.square(y_overlap(rectA, rectB))))


def high_y(rectA, rectB):
    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB
    return abs(max(ay1, ay2) - max(by1, by2))/max(ay2 - ay1, by2 - by1)


# 0.5 for y
db = DBSCAN(eps=0.5, min_samples=1,
            metric=y_overlap).fit(rects)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
colors = {}
label_coords = {}
for idx, rect in enumerate(rects):
    x1, y1, x2, y2 = rect
    if labels[idx] > -1:
        if labels[idx] not in colors:
            colors[labels[idx]] = np.random.randint(0, 255, (3,))
        if labels[idx] not in label_coords:
            label_coords[labels[idx]] = [rect]
        else:
            label_coords[labels[idx]].append(rect)
        _color = (int(colors[labels[idx]][0]),
                  int(colors[labels[idx]][1]), int(colors[labels[idx]][2]))
        # cv2.rectangle(vis, (x1, y1), (x2, y2), _color, 1)
    # else:
        # cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    pass
# cv2.imwrite("./images/output.png", img)
# cv2.imshow('img', vis)
# cv2.waitKey(0)


def cluster_cluster(rects):
    r = DBSCAN(eps=2, min_samples=1,
               metric=x_dist).fit(rects)
    labels = r.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

    colors = {}
    label_coords = {}
    for idx, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        if labels[idx] > -1:
            if labels[idx] not in colors:
                colors[labels[idx]] = np.random.randint(0, 255, (3,))
            if labels[idx] not in label_coords:
                label_coords[labels[idx]] = [rect]
            else:
                label_coords[labels[idx]].append(rect)
            _color = (int(colors[labels[idx]][0]),
                      int(colors[labels[idx]][1]), int(colors[labels[idx]][2]))
            cv2.rectangle(vis, (x1, y1), (x2, y2), _color, 1)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        pass
    # cv2.imwrite("./images/output.png", img)


rects3 = [label_coords[label] for label in label_coords]
for rect in rects3:
    cluster_cluster(rect)
cv2.imshow('img', vis)
cv2.waitKey(0)
exit()
rects2 = []
for label in label_coords:
    coords = np.array(label_coords[label])
    v = [0, 0, 0, 0]
    v[0] = int(np.min(coords[:, 0]))
    v[2] = int(np.max(coords[:, 2]))
    v[1] = int(np.min(coords[:, 1]))
    v[3] = int(np.max(coords[:, 3]))
    rects2.append(v)


for rect in rects2:
    x1, y1, x2, y2 = rect
    cv2.rectangle(vis4, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow('img', vis4)
cv2.waitKey(0)
