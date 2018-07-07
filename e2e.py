import argparse
import cv2
from segmentation.RegionExtractor import RegionExtractor
from data.RegionDataset import RegionDataset
import executor
import os
import re

ALGORITHM_CONFIG = "htrnet-pc-iam-print"
MODEL_DATE = "2018-07-02-23-46-51"
MODEL_EPOCH = 65

PUNCTUATION_REGEX = re.compile(r"([|])(?=[,.;:!?])")
REGULAR_REGEX = re.compile(r"[|]")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--config', default=ALGORITHM_CONFIG)
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument(
        '--model-date', help='date to continue for', default=MODEL_DATE)
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=MODEL_EPOCH, type=int)
    args = parser.parse_args()

    #############################
    # 1. EXTRACT REGIONS        #
    #############################
    print("Extracting Regions...")
    img = cv2.imread(args.input)
    re = RegionExtractor(img)
    regions = re.extract()

    #########################################
    # 2. CLASSIFY PRINTED / HANDWRITTEN     #
    #########################################
    print("Classifying & Transcribing Regions...")
    # TODO: this is not so nice...
    models_path = os.path.join(
        executor.MODELS_PATH, '{}-{}'.format("htrnet-pc-iam", args.model_date))
    dataset = RegionDataset(regions, models_path)

    exc = executor.Executor(args.config, _dataset=dataset, verbose=False)
    exc.configure(args.gpu, True, False)
    transcriptions = exc.transcribe("", args.model_date, args.model_epoch)
    # print(transcriptions)

    print('\n')
    print('Transcriptions for {} lines'.format(len(transcriptions['trans'])))
    max_trans_l = max(map(lambda t: len(t), transcriptions['trans']))
    line_format = '{0:'+str(max_trans_l+10)+'} {1:30} {2}'
    heading = line_format.format('Transcription', 'Classification', 'File')
    print(heading)
    print("-"*len(heading))
    for i in range(len(transcriptions['trans'])):
        decompiled = exc.dataset.decompile(transcriptions['trans'][i])
        filename = os.path.basename("")
        if len(transcriptions['class']) > i:
            is_ht = 'Handwritten' if transcriptions['class'][i][0] > 0.25 else 'Printed'
            is_ht = '{:12} ({:05.2f} %)'.format(
                is_ht, transcriptions['class'][i][0]*100)

        else:
            is_ht = '?'

        print(line_format.format(decompiled, is_ht, filename))

    #########################################
    # 4. VISUALIZE TRANSCRIPTIONS           #
    #########################################
    print("Visualizing Transcriptions...")

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2

    for idx, region in enumerate(regions):
        x, y = region.pos
        w, h = region.size
        decompiled = exc.dataset.decompile(
            transcriptions['trans'][idx])
        decompiled = PUNCTUATION_REGEX.sub('', decompiled)
        decompiled = REGULAR_REGEX.sub(' ', decompiled)
        if transcriptions['class'][idx] > 0.25:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(img, decompiled, (x, y-5), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 1)
    cv2.imwrite('output.png', img)
    # cv2.imshow('All', cv2.resize(img, (533, 800)))
    # cv2.waitKey(0)
