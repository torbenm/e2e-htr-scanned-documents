from .dataset import Dataset
import os
from .. import util
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2

TRUTH_FILE = "training_2011.xml"
IMAGE_PATH = "training_2011_gray/images_gray"


class RimesDataset(Dataset):

    identifier = "rimes"
    requires_splitting = True

    def getFilesAndTruth(self, basepath, subset, limit=-1):
        return self._load_ascii_lines(basepath, subset, limit)

    def getDownloadInfo(self):
        error_message = "Unfortunately, automatical downloading of the Rimes database cannot be supported." + \
            "\nPlease visit http://www.a2ialab.com/doku.php?id=rimes_database:start and follow the guidelines.\n" + \
            "Then place the data in data/raw/rimes and run the download script again.\n"

        download_info = {
            "images": {
                "file": "training_2011_gray.tar",
                "error": error_message,
                "tgz": True
            },
            "truth": {
                "file": TRUTH_FILE,
                "error": error_message,
                "tgz": False
            }
        }

        return download_info

    def getIdentifier(self):
        return self.identifier

    def _load_ascii_lines(self, basepath, type, limit=-1):
        parsed = util.loadJson(basepath, "images")
        if limit > -1:
            parsed = parsed[:limit]
        fulltext = ""
        for line in parsed:
            fulltext += line["truth"]
        return parsed, fulltext

    def do_split(self, path):
        split_path = os.path.join(path, "split")
        img_split_path = os.path.join(split_path, "imgs")
        if not os.path.exists(split_path):
            os.makedirs(img_split_path)
            tree = xml.dom.minidom.parse(
                os.path.join(path, TRUTH_FILE)).documentElement
            limit = 100
            images = []
            for page in tree.getElementsByTagName("SinglePage"):
                for paragraph in page.getElementsByTagName("Paragraph"):
                    image_name = os.path.basename(
                        page.getAttribute("FileName"))
                    img_basename = '.'.join(image_name.split('.')[:-1])
                    img_ext = image_name.split('.')[-1]
                    image_path = os.path.join(path, IMAGE_PATH, image_name)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    i = 0
                    if img is not None:
                        for line in paragraph.getElementsByTagName("Line"):
                            output_path = os.path.join(
                                img_split_path, '{}-{}.{}'.format(img_basename, str(i), img_ext))
                            x_l = max(int(line.getAttribute("Left")), 0)
                            x_r = int(line.getAttribute("Right"))
                            y_b = int(line.getAttribute("Bottom"))
                            y_t = max(int(line.getAttribute("Top")), 0)
                            truth = line.getAttribute("Value")
                            crop_img = img[y_t:y_b, x_l:x_r]
                            images.append({
                                "truth": truth,
                                "path": output_path
                            })
                            cv2.imwrite(output_path, crop_img)
                            i += 1
            print(str(len(images))+" Images extracted")
            util.dumpJson(split_path, "images", images)
