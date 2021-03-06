import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from .metrics import x_by_height, y_by_height

from lib.segmentation.Region import Region
from lib.Configuration import Configuration

ORIENTATION_LANDSCAPE = 0
ORIENTATION_PORTRAIT = 1


DEFAULT_CONFIG = {
    "mser_min_area": 0,
    "threshold_kernel": 51,
    "cluster_y_eps": 0.1,
    "cluster_x_eps": 0.3,
    "min_wh_ratio": 0
}


class WordDetector(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULT_CONFIG)

    def _threshold(self, img):
        mean = np.mean(img, axis=(0, 1))
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, self.config["threshold_kernel"], mean/4)

    def _mser(self, img, max_area=14400, min_area=0):
        mser = cv2.MSER_create(_max_area=self._mser_max_area,
                               _min_area=self.config["mser_min_area"])
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

    def _cluster(self, rects, eps, metric):
        if not rects:
            return []
        db = DBSCAN(eps=eps, min_samples=1,
                    metric=metric).fit(rects)
        labels = db.labels_
        cluster = []
        cluster_dict = {}
        for idx, rect in enumerate(rects):
            if labels[idx] > -1 and rect:
                if labels[idx] not in cluster_dict:
                    cluster_dict[labels[idx]] = []
                cluster_dict[labels[idx]].append(rect)
            else:
                cluster.append([rect])
        cluster.extend(cluster_dict.values())
        return cluster

    def _extract_region(self, original, region):
        x1, y1, x2, y2 = region
        return Region(
            pos=(x1, y1), size=(x2-x1, y2-y1), img=original[y1:y2, x1:x2])

    def _combined_metric(self, rectA, rectB):
        if y_by_height(rectA, rectB) < self.config["cluster_y_eps"] and x_by_height(rectA, rectB) < self.config["cluster_x_eps"]:
            return 0
        return 1

    def extract(self, img):
        original = img.copy()
        self._mser_max_area = int(img.shape[0]*img.shape[1]/2)
        self._min_region_area = img.shape[0]*img.shape[1]/(200*200)

        #########################
        #   1. THRESHOLDING
        ########################
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        # rect_groups = self._cluster(
        #     rects, self.config["cluster_x_eps"], x_by_height)
        rect_groups = self._cluster(
            rects, 0.5, self._combined_metric)
        #########################
        #   5. X-AXIS Clustering
        ########################
        # rect_super_groups = [self._cluster(
        #     rect_group, self.config["cluster_y_eps"], y_by_height) for rect_group in rect_groups]
        rect_super_groups = [rect_groups]

        #########################
        #   6. Grouping clusters
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
                if (v[2]-v[0])*(v[3]-v[1]) > self._min_region_area and (v[2]-v[0])/(v[3]-v[1]) > self.config["min_wh_ratio"]:
                    regions.append(v)

        #########################
        #   7. Get Region Image
        ########################
        return [self._extract_region(original, region) for region in regions]


if __name__ == "__main__":
    IMAGE = "./segmentation/images/00007-paper.png"
    # IMAGE = "./segmentation/images/00001-paper.png"
    img = cv2.imread(IMAGE)
    re = RegionExtractor(img)
    regions = re.extract()
    for region in regions:
        # cv2.imshow('final_region', region.img)
        # cv2.waitKey(0)
        x, y = region.pos
        w, h = region.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imwrite('./segmentation/output.png', img)
    # cv2.imshow('All', img)
    # cv2.waitKey(5000)
