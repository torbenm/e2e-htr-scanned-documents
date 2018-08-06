from .iou import fullpageiou
import pylev

IOU_THRESHOLD: float = 0.5
LEVENSHTEIN_THRESHOLD: float = 0.25


def levenshtein(gt, pred):
    ln = float(max(len(gt), len(pred)))
    return pylev.levenshtein(gt, pred)/ln


def evaluate(groundtruth, predictions, iou_threshold=IOU_THRESHOLD, levenshtein_threshold=LEVENSHTEIN_THRESHOLD):
    pairs, misfire = fullpageiou(groundtruth, predictions, iou_threshold)
    total_pairs = len(pairs) + misfire
    correct_pairs = 0.0
    for pair in pairs:
        if pair["pred"] is not None:
            dist = levenshtein(pair["gt"]["text"], pair["pred"]["text"])
            pair["dist"] = dist
            if dist < levenshtein_threshold:
                correct_pairs += 1.0 * (1-(dist*2))
    return pairs, correct_pairs/total_pairs


if __name__ == "__main__":
    from data import util
    pred = util.loadJson("./eval", "examplePred")
    gt = util.loadJson("./eval", "exampleGT")
    print("Full Page Example for exact match")
    line = "{:30}{:30}   {:6.2f}%   {:6.2f}%"
    pairs, correctness = evaluate(gt, pred)
    for pair in pairs:
        print(line.format(pair["gt"]["text"],
                          pair["pred"]["text"],
                          pair["dist"]*100,
                          pair["iou"]*100))
    print("Correctness of {:6.2f}".format(correctness))
