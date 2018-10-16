from lib.Configuration import Configuration

DEFAULT_CONFIG = {
    "threshold": 0.5,
    "filter_class": True,
    "target_class": 1
}


class IoU(object):

    def __init__(self, config={}, default_config=DEFAULT_CONFIG):
        self.config = Configuration(config, default_config)

    def _region_to_box(self, region):
        return {
            "x": region.pos[0],
            "y": region.pos[1],
            "w": region.size[0],
            "h": region.size[1]
        }

    def calc(self, boxA, boxB):
        boxA = self._region_to_box(boxA)
        boxB = self._region_to_box(boxB)
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA["x"], boxB["x"])
        yA = max(boxA["y"], boxB["y"])
        xB = min(boxA["w"]+boxA["x"], boxB["w"]+boxB["x"])
        yB = min(boxA["h"]+boxA["y"], boxB["h"]+boxB["y"])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA["w"] * boxA["h"])
        boxBArea = (boxB["w"] * boxB["h"])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def _get_hits(self, groundtruth, predictions):
        hits = []
        misfire = []
        nofire = []
        for gt in groundtruth:
            maxiou = 0
            bestpred = None
            for pred in predictions:
                iou = self.calc(gt, pred)
                if iou > self.config["threshold"] and iou > maxiou:
                    maxiou = iou
                    bestpred = pred
            if bestpred is not None:
                hits.append({
                    "gt": gt,
                    "pred": bestpred,
                    "iou": maxiou
                })
            else:
                nofire.append(gt)

        for pred in predictions:
            maxiou = 0
            for gt in groundtruth:
                iou = self.calc(gt, pred)
                if iou > self.config["threshold"] and iou > maxiou:
                    maxiou = iou
            if maxiou == 0:
                misfire.append(pred)

        return hits, misfire, nofire

    def _calc_score(self, hits, misfire, nofire):
        total_len = len(hits) + len(nofire) + len(misfire)
        if total_len == 0:
            return 1, hits, nofire, misfire
        return len(hits)/total_len, hits, nofire, misfire

    def withinfo(self, groundtruth, predictions):
        if self.config["filter_class"]:
            predictions = list(
                filter(lambda p: p.cls == self.config["target_class"], predictions))
        hits, misfire, nofire = self._get_hits(groundtruth, predictions)
        return self._calc_score(hits, misfire, nofire)

    def __call__(self, groundtruth, predictions):
        return self.withinfo(groundtruth, predictions)[0]
