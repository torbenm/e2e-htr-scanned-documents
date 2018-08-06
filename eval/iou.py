# INTERSECTION OVER UNION


def calc_iou(boxA, boxB):
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


def fullpageiou(groundtruth: list, predictions: list, threshold: float = 0.0) -> list:
    pairs = []
    misfire = 0
    for gt in groundtruth:
        maxiou = 0
        bestpred = None
        for pred in predictions:
            iou = calc_iou(gt, pred)
            if iou > threshold and iou > maxiou:
                maxiou = iou
                bestpred = pred
        pairs.append({
            "gt": gt,
            "pred": bestpred,
            "iou": maxiou
        })
    for pred in predictions:
        maxiou = 0
        for gt in groundtruth:
            iou = calc_iou(gt, pred)
            if iou > threshold and iou > maxiou:
                maxiou = iou
        if maxiou == 0:
            misfire += 1

    return pairs, misfire


if __name__ == "__main__":
    from data import util
    regions = util.loadJson("../paper-notes/data/final/train", "00000-truth")
    print("Full Page Example for exact match")
    line = "{:30}{:30}{:6.2f}%"
    pairs, misfire = fullpageiou(regions, regions)
    for pair in pairs:
        print(line.format(pair["gt"]["text"],
                          pair["pred"]["text"],
                          pair["iou"]*100))
