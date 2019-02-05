import sys
from .GTProvider import GTProvider
from lib.segmentation.Region import Region


class ParagraphRegionGTProvider(GTProvider):

    def _side2region(self, side):
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = 0
        max_y = 0
        for gt in side:
            min_x = min(min_x, gt["x"])
            min_y = min(min_y, gt["y"])
            max_x = max(max_x, gt["x"]+gt["w"])
            max_y = max(max_y, gt["y"]+gt["h"])

        return Region(pos=(min_x, min_y), size=(max_x-min_x, max_y-min_y), img=self.original[min_y:max_y, min_x:max_x, :])

    def _group_sides(self, gts):
        sides = {
            "0": [],
            "1": [],
            "2": [],
            "3": []
        }
        [sides[str(gt["side"])].append(gt) for gt in gts]
        return sides

    def __call__(self, imgfile, original):
        self.original = original
        gts = self._load_gt(imgfile)
        sides = self._group_sides(gts)
        regions = []
        for key in sides.keys():
            if(len(sides[key]) > 0):
                regions.append(self._side2region(sides[key]))
        return regions
