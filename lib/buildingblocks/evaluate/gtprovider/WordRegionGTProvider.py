from .GTProvider import GTProvider
from lib.segmentation.Region import Region


class WordRegionGTProvider(GTProvider):

    def _gt2region(self, gt):
        return Region(pos=(gt["x"], gt["y"]), size=(gt["w"], gt["h"]), text=gt["text"], img=self.original[gt["y"]:gt["y"]+gt["h"], gt["x"]:gt["x"]+gt["w"], :])

    def __call__(self, imgfile):
        gts = self._load_gt(imgfile)
        return [self._gt2region(gt) for gt in gts]
