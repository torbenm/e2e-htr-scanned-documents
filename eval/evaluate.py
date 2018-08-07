from .iou import fullpageiou
import pylev
import re

IOU_THRESHOLD: float = 0.5
LEVENSHTEIN_THRESHOLD: float = 0.25
PUNCTUATION_REGEX = re.compile(r"([|])(?=[,.;:!?])")
REGULAR_REGEX = re.compile(r"[|]")


def clean_text(text):
    text = PUNCTUATION_REGEX.sub('', text)
    text = REGULAR_REGEX.sub(' ', text)
    return text


def levenshtein(gt, pred):
    ln = float(max(len(gt), len(pred)))
    return pylev.levenshtein(gt, pred)/ln


def score_threshed(dist, threshold):
    return 0 if dist > threshold else 1.0 * (1-(dist*2))


def score_squared_error(dist, threshold):
    return 1.0 - (dist if dist < 1.0 else 1.0)


def evaluate(groundtruth, predictions, iou_threshold=IOU_THRESHOLD, levenshtein_threshold=LEVENSHTEIN_THRESHOLD, score_fn=score_squared_error):
    pairs, misfire = fullpageiou(groundtruth, predictions, iou_threshold)
    total_pairs = len(list(filter(lambda x: x["pred"] is not None or len(
        x["gt"]["text"]) > 1, pairs))) + misfire
    correct_pairs = 0.0
    for pair in pairs:
        if pair["pred"] is not None:
            dist = levenshtein(clean_text(pair["gt"]["text"]).lower(),
                               clean_text(pair["pred"]["text"]).lower())
            pair["dist"] = dist
            correct_pairs += score_fn(dist, levenshtein_threshold)
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
