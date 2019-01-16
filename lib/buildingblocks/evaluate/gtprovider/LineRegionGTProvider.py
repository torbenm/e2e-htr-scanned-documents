import sys
from .GTProvider import GTProvider
from lib.segmentation.Region import Region


class LineRegionGTProvider(GTProvider):

    def _line2region(self, line):
        sorted_line = sorted(line, key=lambda gt: gt["x"])
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = 0
        max_y = 0
        text = ""
        for gt in sorted_line:
            min_x = min(min_x, gt["x"])
            min_y = min(min_y, gt["y"])
            max_x = max(max_x, gt["x"]+gt["w"])
            max_y = max(max_y, gt["y"]+gt["h"])
            text += "" if len(text) == 0 else "|"
            text += gt["text"]

        return Region(pos=(min_x, min_y), size=(max_x-min_x, max_y-min_y), text=text, cls=1, img=self.original[min_y:max_y, min_x:max_x, :])

    def _group_lines(self, side):
        lines = {}
        for gt in side:
            lines[gt["line"]] = [gt] if gt["line"] not in lines else [
                *lines[gt["line"]], gt]
        return [self._line2region(line) for line in lines.values()]

    def _group_sides(self, gts):
        sides = {
            "0": [],
            "1": [],
            "2": [],
            "3": []
        }
        [sides[str(gt["side"])].append(gt) for gt in gts]
        return sides

    def __call__(self, imgfile, original=None):
        self.original = original
        gts = self._load_gt(imgfile)
        sides = self._group_sides(gts)
        return [r for side in sides.values() for r in self._group_lines(side)]
