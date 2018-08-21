import argparse
import cv2
import numpy as np
from time import time
# from segmentation.MSERRegionExtractor import RegionExtractor
from segmentation.WordRegionExtractor import RegionExtractor
from data.RegionDataset import RegionDataset
from data.util import loadJson, rmkdir
from eval.evaluate import evaluate
import executor
import os
import re

# "htrnet-pc-iam-print"
# otf-iam-both-2018-08-07-15-38-49
ALGORITHM_CONFIG = "otf-iam-paper"
# "2018-07-07-14-59-06"  # "2018-07-02-23-46-51"
MODEL_DATE = "2018-08-19-18-20-55"
# 800  # 65
MODEL_EPOCH = 14

DATAPATH = "../paper-notes/data/final"
SUBSET = "dev"


PUNCTUATION_REGEX = re.compile(r"([|])(?=[,.;:!?])")
REGULAR_REGEX = re.compile(r"[|]")

HTR_THRESHOLD = 0.95


class End2End(object):

    def __init__(self, config, model_date, model_epoch, gpu=-1):
        self.model_date = model_date
        self.model_epoch = model_epoch
        self.models_path = os.path.join(
            executor.MODELS_PATH, '{}-{}'.format(config, model_date))
        self.dataset = RegionDataset(None, self.models_path)
        self.dataset.scaling(2.15, 123, 1049)
        self.executor = executor.Executor(
            config, _dataset=self.dataset, verbose=False)
        self.executor.configure(gpu, True, False)

    def _regions(self, img):
        return RegionExtractor(img).extract()

    def _trim_text(self, text):
        return text[:-1] if text.endswith("|") else text

    def _post_process(self, regions, transcriptions):
        result = []
        other = []
        for idx, region in enumerate(regions):
            x, y = region.pos
            w, h = region.size
            obj = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "text": self._trim_text(self.dataset.decompile(transcriptions['trans'][idx])),
                "thresh": transcriptions['class'][idx][0]
            }
            if transcriptions['class'][idx][0] > HTR_THRESHOLD:
                result.append(obj)
            else:
                other.append(obj)
        return result, other

    def _evaluate(self, preds, truthpath):
        gt = loadJson(truthpath)
        return evaluate(gt, preds)

    def _print_transcriptions(self, transcriptions):
        line_format = '{0:70} {1:30}'
        heading = line_format.format('Transcription', 'Classification')
        print(heading)
        print("-"*len(heading))
        for i in range(len(transcriptions['trans'])):
            decompiled = self.dataset.decompile(transcriptions['trans'][i])
            if len(transcriptions['class']) > i:
                is_ht = 'Handwritten' if transcriptions['class'][i][0] > HTR_THRESHOLD else 'Printed'
                is_ht = '{:12} ({:05.2f} %)'.format(
                    is_ht, transcriptions['class'][i][0]*100)

            else:
                is_ht = '?'

            print(line_format.format(decompiled, is_ht))

    def _viz(self, img, box, show_text=True, color=(0, 0, 255)):
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        if show_text:
            cv2.putText(img, box["text"], (x, y-5), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 1)

    def __call__(self, imgpath: str, truthpath: str, viz=False, class_scores=False):
        img = cv2.imread(imgpath)
        regions = self._regions(img)
        self.dataset.set_regions(regions)
        transcriptions = self.executor.transcribe(
            "", self.model_date, self.model_epoch)
        if class_scores:
            self._print_transcriptions(transcriptions)
        preds, nonhts = self._post_process(regions, transcriptions)
        pairs, score = self._evaluate(preds, truthpath)

        if viz:
            for pair in pairs:
                self._viz(img, pair["gt"], False, (0, 255, 0))
            for nonht in nonhts:
                self._viz(img, nonht, False, (255, 0, 0))
            for pred in preds:
                self._viz(img, pred, True)

        return preds, pairs, score, None if not viz else img

    def close(self):
        self.executor.close()

    def paper_notes(self, basepath, num, viz=False, class_scores=False):
        return self(os.path.join(basepath, "{}-paper.png".format(num)), os.path.join(basepath, "{}-truth.json".format(num)), viz, class_scores)


def paper_note(e2e, basepath, num, viz=False, output=False, class_scores=False):
    preds, pairs, score, img = e2e.paper_notes(
        basepath, num, output or viz, class_scores)
    print("{:10}{:6.2f}%".format(num, score*100))
    if output:
        cv2.imwrite("./outputs/{}-output.png".format(num), img)
    if viz:
        cv2.imshow('viz', img)
        cv2.waitKey(0)
    return score


if __name__ == "__main__":

    def get_num(name: str):
        return name.split(".")[0].split("-")[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--config', default=ALGORITHM_CONFIG)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--output', default=False, action='store_true')
    parser.add_argument('--class-scores', default=False, action='store_true')
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument(
        '--model-date', help='date to continue for', default=MODEL_DATE)
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=MODEL_EPOCH, type=int)
    parser.add_argument('--limit', default=-1, type=int)

    parser.add_argument('--datapath', default=DATAPATH)
    parser.add_argument('--subset', default=SUBSET)
    args = parser.parse_args()

    basepath = os.path.join(args.datapath, args.subset)

    os.makedirs("./outputs", exist_ok=True)

    e2e = End2End(args.config, args.model_date, args.model_epoch, args.gpu)

    # files = os.listdir(basepath)
    filenums = [
        "04693",
        "10169",
        "04298",
        "04787",
        "10200",
        "09908",
        "04802",
        "09849",
        "04598",
        "10028",
        "09799"
    ]
    files = list(map(lambda num: "{}-truth.json".format(num), filenums))

    idx = 0
    scores = []
    start = time()
    for file in files:
        if idx >= args.limit and not args.limit == -1:
            break
        if file.endswith("json"):
            num = get_num(file)
            scores.append(paper_note(e2e, basepath, num,
                                     args.visualize, args.output, args.class_scores))
            idx += 1

    e2e.close()
    print("Average score: {:6.2f}%".format(np.mean(scores)*100))
    print("Took {:.2f}s".format(time() - start))
