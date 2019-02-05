import numpy as np
import cv2


class Slicer(object):

    def __init__(self, **kwargs):
        self.slice_width = kwargs.get('slice_width', 320)
        self.slice_height = kwargs.get('slice_height', 320)

    def slice(self, img, bg=255.0):
        slices = []
        img = np.reshape(img, [img.shape[0], img.shape[1], 1])
        if self.slice_width == -1 or self.slice_height == -1:
            return [img]

        for w_i in range(int(np.ceil(img.shape[1]/self.slice_width))):
            for h_i in range(int(np.ceil(img.shape[0]/self.slice_height))):
                slc = np.full(
                    (self.slice_height, self.slice_width, 1), fill_value=bg, dtype=np.uint8)
                slc_ = img[(h_i*self.slice_height):min((h_i+1)*self.slice_height, img.shape[0]),
                           (w_i*self.slice_width):min((w_i+1)*self.slice_width, img.shape[1]), :]
                slc[:slc_.shape[0], :slc_.shape[1],
                    :slc_.shape[2]] = slc_
                slices.append(slc)
        return slices

    def merge(self, slices, original_shape):
        rows = (original_shape[0] // self.slice_height)+1
        cols = (original_shape[1] // self.slice_width)+1

        merged_shape = np.zeros(
            (rows*self.slice_height, cols*self.slice_width))
        for idx, slc in enumerate(slices):
            row = idx % rows
            col = idx // rows
            merged_shape[row*self.slice_height:(
                row+1)*self.slice_height, col*self.slice_width:(col+1)*self.slice_width] = slc

        return merged_shape[:original_shape[0], :original_shape[1]]

    def __call__(self, img, bg=255.0):
        return self.slice(img, bg)
