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
        with open(os.path.join(basepath, self.identifier, "ascii/{}.txt".format(type)), "r") as lines:
            parsed = []
            fulltext = ""
            i = 0
            while limit == -1 or i < limit:
                line = lines.readline().strip()
                if not line:
                    break
                if line[0] != "#":
                    lsplit = line.split(" ")
                    i = i + 1
                    if lsplit[1] != "err":
                        path = self._get_image_path(basepath, type, lsplit[0])
                        text = ' '.join(lsplit[8:])
                        parsed.append({"path": path, "truth": text})
                        fulltext = fulltext + text
            return parsed, fulltext

    def _get_image_path(self, basepath, subset, identifier):
        idsplit = identifier.split("-")
        return os.path.join(basepath, self.identifier, subset, idsplit[0], "-".join(idsplit[0:2]), identifier + ".png")

    def do_split(self, path):
        split_path = os.path.join(path, "split")
        img_split_path = os.path.join(split_path, "imgs")
        # TODO: remove
        util.rmkdir(img_split_path)
        # if not os.path.exists(split_path):
        # os.makedirs(img_split_path)
        tree = xml.dom.minidom.parse(
            os.path.join(path, TRUTH_FILE)).documentElement
        limit = 5
        for page in tree.getElementsByTagName("SinglePage"):
            for paragraph in page.getElementsByTagName("Paragraph"):
                image_name = os.path.basename(
                    paragraph.getAttribute("FileName"))
                img_basename = '.'.join(image_name.split('.')[:-1])
                image_path = os.path.join(path, IMAGE_PATH, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                for line in paragraph.getElementsByTagName("Line")
                    crop_img = img
                    cv2.imshow("cropped", crop_img)
                    cv2.waitKey(0)
                    limit -= 1
                    if limit == 0:
                        exit()
