import cv2
import numpy as np
from sklearn.cluster import DBSCAN

ORIENTATION_LANDSCAPE = 0
ORIENTATION_PORTRAIT = 1

##################
# REGION DETECTION ALGORITHM
#   1. Thresholding
#


class Region(object):
    def __init__(self, pos, size, img):
        self.img = img
        self.pos = pos
        self.size = size


class RegionExtractor(object):

    def __init__(self, img):
        self.original = img
        self._mser_max_area = int(img.shape[0]*img.shape[1]/2)
        self._mser_min_area = 0
        self._threshold_kernel = 51
        self._cluster_y_eps = 0.5
        self._cluster_x_eps = 2.0
        self._min_region_area = img.shape[0]*img.shape[1]/1000.0

        if np.max(self.original.shape[:2]) < 800 or np.max(self.original.shape[0:2]) < 533:
            print("Please provide the image in a higher resolution")
            exit()

    def _threshold(self, img):
        mean = np.mean(img, axis=(0, 1))
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, self._threshold_kernel, mean/4)

    def _mser(self, img, max_area=14400, min_area=0):
        mser = cv2.MSER_create(_max_area=self._mser_max_area,
                               _min_area=self._mser_min_area)
        regions, _ = mser.detectRegions(img)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        return hulls

    def _duplicate(self, rects):
        res = []
        for rect in rects:
            res.extend([rect]*2)
        return res

    def _hull2rect(self, hull):
        return cv2.boundingRect(hull)

    def _metric_y_overlap(self, rectA, rectB):
        ax1, ay1, ax2, ay2 = rectA
        bx1, by1, bx2, by2 = rectB
        avg_dist = (abs(ay1 - by1) + abs(ay2 - by2))/2
        avg_height = ((ay2 - ay1) + (by2 - by1)) / 2
        return avg_dist / avg_height

    def _metric_x_dist(self, rectA, rectB):
        ax1, ay1, ax2, ay2 = rectA
        bx1, by1, bx2, by2 = rectB
        x_dist = min(abs(ax1 - bx2), abs(ax2 - bx1))
        avg_height = ((ay2 - ay1) + (by2 - by1)) / 2.0
        xd = x_dist / avg_height
        return xd

    def _cluster(self, rects, eps, metric):
        db = DBSCAN(eps=eps, min_samples=1,
                    metric=metric).fit(rects)
        labels = db.labels_
        cluster = []
        cluster_dict = {}
        for idx, rect in enumerate(rects):
            if labels[idx] > -1:
                if labels[idx] not in cluster_dict:
                    cluster_dict[labels[idx]] = []
                cluster_dict[labels[idx]].append(rect)
            else:
                cluster.append([rect])
        cluster.extend(cluster_dict.values())
        return cluster

    def extract(self):
        #########################
        #   1. THRESHOLDING
        ########################
        img = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        img = self._threshold(img)

        #########################
        #   2. MSER
        ########################
        hulls = self._mser(img)
        rects = list([self._hull2rect(hull) for hull in hulls])

        #########################
        #   3. Rectangle Grouping
        ########################

        rects = self._duplicate(rects)
        rects = cv2.groupRectangles(rects, 1, 0)
        rects = [[x, y, x+w, y+h] for x, y, w, h in rects[0]]
        #########################
        #   4. Y-AXIS Clustering
        ########################
        rect_groups = self._cluster(
            rects, self._cluster_y_eps, self._metric_y_overlap)
        #########################
        #   5. X-AXIS Clustering
        ########################
        rect_super_groups = [self._cluster(
            rect_group, self._cluster_x_eps, self._metric_x_dist) for rect_group in rect_groups]

        #########################
        #   6. Clusterboxing
        ########################
        regions = []
        for groups in rect_super_groups:
            for subgroup in groups:
                subgroup = np.array(subgroup)
                v = [0, 0, 0, 0]
                v[0] = int(np.min(subgroup[:, 0]))
                v[2] = int(np.max(subgroup[:, 2]))
                v[1] = int(np.min(subgroup[:, 1]))
                v[3] = int(np.max(subgroup[:, 3]))
                if (v[2]-v[0])*(v[3]-v[1]) > self._min_region_area:
                    regions.append(v)

        #########################
        #   7. Get Region Image
        ########################
        return [self._extract_region(region) for region in regions]

    # OLD
    def _get_orientation(self):
        if self.original.shape[0] > self.original.shape[1]:
            return ORIENTATION_PORTRAIT
        else:
            return ORIENTATION_LANDSCAPE

    def _scale(self, img, height):
        factor = img.shape[0] / height
        width = int(img.shape[1] / factor)
        return cv2.resize(img, (width, height)), factor

    def _extract_region(self, region):
        x1, y1, x2, y2 = region
        return Region(
            (x1, y1), (x2-x1, y2-y1), self.original[y1:y2, x1:x2])


if __name__ == "__main__":
    IMAGE = "./images/rl01.jpg"
    img = cv2.imread(IMAGE)
    re = RegionExtractor(img)
    regions = re.extract()
    for region in regions:
        # cv2.imshow('final_region', region.img)
        # cv2.waitKey(0)
        x, y = region.pos
        w, h = region.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imwrite('output.png', img)
    cv2.imshow('All', img)
    cv2.waitKey(5000)
