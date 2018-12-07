import numpy as np
import cv2
import os

from lib.Configuration import Configuration

DEFAULT_CONFIG = {
    "filled": False,
    "large": False
}


class RegionVisualizer(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULT_CONFIG)

    def __call__(self, image, regions, is_gt=False):
        for region in regions:
            self._viz_region(image, region, is_gt)
        return image

    def _draw_lines(self, image, region, color):
        if(len(region.path) > 0):
            cv2.polylines(image, [np.array(region.path)], 1, color)
        else:
            cv2.rectangle(image, region.pos,
                          region.get_bottom_right(), color, 1)

    def _color(self, region, is_gt=False):
        if is_gt:
            return (0, 255, 0)
        return (0, 0, 255) if region.cls is not None and region.cls == 1 else (255, 0, 0)

    def _draw_text(self, image, region, color):
        if region.text is not None and (region.cls is None or region.cls == 1) and self.config.default("text", True):
            x, y = region.pos
            scale = 2 if self.config["large"] else 1
            thickness = 2 if self.config["large"] else 1
            reloc = 5 * scale
            # place text below if there is not enough space above
            y = y + reloc + region.size[1] if y-(20+reloc) < 0 else y - reloc
            cv2.putText(image, region.text, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        scale, color, thickness)

    def _viz_region(self, image, region, is_gt=False):
        color = self._color(region, is_gt)
        self._draw_lines(image, region, color)
        self._draw_text(image, region, color)

    def store(self, vizimage, original_file):
        if self.config.default("store", False):
            os.makedirs(self.config["store"], exist_ok=True)
            filename = os.path.basename(original_file)
            cv2.imwrite(os.path.join(self.config["store"], filename), vizimage)
