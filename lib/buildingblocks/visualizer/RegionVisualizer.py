import numpy as np
import cv2

from lib.Configuration import Configuration

DEFAULT_CONFIG = {}


class RegionVisualizer(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULT_CONFIG)

    def __call__(self, image, regions):
        for region in regions:
            self._viz_region(image, region)
        return image

    def _draw_lines(self, image, region, color):
        if(len(region.path) > 0):
            cv2.polylines(image, [np.array(region.path)], 1, color)
        else:
            cv2.rectangle(image, region.pos,
                          region.get_bottom_right(), color, 1)

    def _color(self, region):
        return (0, 0, 255) if region.cls is not None and region.cls == 1 else (255, 0, 0)

    def _draw_text(self, image, region, color):
        if region.text is not None and (region.cls is None or region.cls == 1):
            x, y = region.pos
            cv2.putText(image, region.text, (x, y-5), cv2.FONT_HERSHEY_PLAIN,
                        1, color, 1)

    def _viz_region(self, image, region):
        color = self._color(region)
        self._draw_lines(image, region, color)
        self._draw_text(image, region, color)
