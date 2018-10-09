import cv2

from lib.Configuration import Configuration


class RegionVisualizer(object):

    def __init__(self, config={}):
        self.config = Configuration(config)

    def __call__(self, image, regions):
        for region in regions:
            self._viz_region(image, region)
        return image

    def _viz_region(self, image, region):
        color = (255, 0, 0)
        if region._class is not None and region._class == 1:
            color = (0, 0, 255)
        x, y = region.pos
        w, h = region.size
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        if region.text is not None and (region._class is None or region._class == 1):
            cv2.putText(image, region.text, (x, y-5), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 1)
