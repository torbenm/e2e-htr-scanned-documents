import cv2
import numpy as np


def pad(array, reference_shape, offsets=None):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """
    offsets = offsets if offsets is not None else [0] * len(reference_shape)
    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                  for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
image = np.transpose(image, [1, 0])

meta = {
    "width": 3767, "height": 130
}

if image.shape[0] != meta["width"] or image.shape[1] != meta["height"]:
    image = pad(image, (meta["width"], meta["height"]))

image = np.transpose(image, [1, 0])

cv2.imwrite("output.png", image)
