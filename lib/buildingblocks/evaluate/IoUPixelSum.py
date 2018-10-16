import numpy as np
import pylev
import cv2

from lib.buildingblocks.evaluate.IoU import IoU

DEFAULT_CONFIG = {
    "threshold": 0,
    "filter_class": True,
    "target_class": 1
}


class IoUPixelSum(IoU):

    def __init__(self, config):
        super().__init__(config, DEFAULT_CONFIG)

    def _score_fn(self, img):
        if img.shape[0] == 0 or img.shape[1] == 0:
            return 0
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.sum(255-img)

    def _calc_score(self, hits, misfire, nofire):
        # GT that has no predicitons, but consist only of 1 char are ignored
        nofire = list(filter(lambda x: len(x["text"]) > 1, nofire))
        total_len = len(hits) + len(nofire) + len(misfire)
        hit_score = 0
        for hit in hits:
            if hit["pred"].img is None or hit["gt"].img is None:
                continue
            gt_score = self._score_fn(hit["gt"].img)
            pred_score = self._score_fn(hit["pred"].img)
            hit_score += min(pred_score, gt_score)/max(pred_score, gt_score)
        if total_len == 0:
            return 1, hits, misfire, nofire
        return hit_score/total_len, hits, misfire, nofire
