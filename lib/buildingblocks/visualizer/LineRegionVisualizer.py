import cv2

from lib.Configuration import Configuration


class LineRegionVisualizer(object):

    def __init__(self, config={}):
        self.config = Configuration(config)

    def __call__(self, image, regions):
        for region in regions:
            self._viz_region(image, region)
        return image

    def _viz_region(self, image, region):
        color = (255, 0, 0)
        if region.cls is not None and region.cls == 1:
            color = (0, 0, 255)
        x, y = region.pos
        cv2.polylines(image, )
        cv2.rectangle(image, region.pos, region.get_bottom_right(), color, 1)
        if region.text is not None and (region.cls is None or region.cls == 1):
            cv2.putText(image, region.text, (x, y-5), cv2.FONT_HERSHEY_PLAIN,
                        1, color, 1)
