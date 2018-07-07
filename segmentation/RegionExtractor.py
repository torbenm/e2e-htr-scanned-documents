import cv2
import numpy as np
ORIENTATION_LANDSCAPE = 0
ORIENTATION_PORTRAIT = 1


class Region(object):
    def __init__(self, pos, size, img):
        self.img = img
        self.pos = pos
        self.size = size


class RegionExtractor(object):

    def __init__(self, img):
        self.original = img
        self.tl_padding = [0, 5]  # 5 top-bottom, 0 left-right
        if np.max(self.original.shape[:2]) < 800 or np.max(self.original.shape[0:2]) < 533:
            print("Please provide the image in a higher resolution")
            exit()

    def _get_orientation(self):
        if self.original.shape[0] > self.original.shape[1]:
            return ORIENTATION_PORTRAIT
        else:
            return ORIENTATION_LANDSCAPE

    def _scale(self, img, height):
        factor = img.shape[0] / height
        width = int(img.shape[1] / factor)
        return cv2.resize(img, (width, height)), factor

    def _extract_regions(self, img, hulls, scaling=1, padding=[0, 0], offset=[0, 0], original_region=Region((0, 0), (0, 0), None)):
        regions = []
        for contour in hulls:
            x, y, w, h = cv2.boundingRect(contour)
            x *= scaling
            y *= scaling
            w *= scaling
            h *= scaling
            x = max(int(x) - padding[0] + offset[0], 0)
            y = max(int(y) - padding[1] + offset[1], 0)
            w = int(w) + (padding[0]*2)
            h = int(h) + (padding[1]*2)
            if w / h < 1:
                continue
            regions.append(Region(
                (original_region.pos[0] + x, original_region.pos[1] + y), (w, h), img[y:y+h, x:x+w]))
        return regions

    def _pad_by_height(self, img, color):
        return cv2.copyMakeBorder(
            img, img.shape[0], img.shape[0], img.shape[0], img.shape[0], cv2.BORDER_CONSTANT, value=color)

    def _extract_subregions(self, region, scaling):
        img = cv2.cvtColor(region.img, cv2.COLOR_BGR2GRAY)
        img = self._threshold(img)
        eroding_height = int(2*scaling)
        img = cv2.erode(img, np.ones((eroding_height, 1)))
        img = cv2.dilate(img, np.ones((1, region.img.shape[1]*2)))
        padded = self._pad_by_height(img, (0))
        max_area = int((img.shape[0] * img.shape[1])/3*4)
        min_area = int((img.shape[1] * scaling*3))
        hulls = self._apply_mser(padded, max_area, min_area)

        if len(hulls) <= 1:
            return [region]
        subregions = self._extract_regions(region.img, hulls, padding=[
                                           0, eroding_height], offset=[-region.img.shape[0]]*2, original_region=region)
        return subregions

    def extract(self):
        height = 533 if self._get_orientation() == ORIENTATION_LANDSCAPE else 800
        tl_img = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        tl_img, scaling = self._scale(tl_img, height)
        tl_img = self._threshold(tl_img)
        tl_img = cv2.dilate(tl_img, np.ones((1, 15)))
        hulls = self._apply_mser(tl_img)
        regions = self._extract_regions(
            self.original, hulls, scaling, self.tl_padding)
        final_regions = []
        for region in regions:
            final_regions.extend(self._extract_subregions(region, scaling))
        return final_regions

    def _threshold(self, img):
        mean = np.mean(img, axis=(0, 1))
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, mean/4)

    def _apply_mser(self, img, max_area=14400, min_area=60):
        mser = cv2.MSER_create(_max_area=max_area, _min_area=min_area)
        regions, _ = mser.detectRegions(img)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        return hulls


if __name__ == "__main__":
    IMAGE = "./rl01.jpg"
    img = cv2.imread(IMAGE)
    re = RegionExtractor(img)
    regions = re.extract()

    for region in regions:
        # cv2.imshow('final_region', region.img)
        # cv2.waitKey(0)
        x, y = region.pos
        w, h = region.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imwrite('output.png', img)
    cv2.imshow('All', cv2.resize(img, (533, 800)))
    cv2.waitKey(5000)
