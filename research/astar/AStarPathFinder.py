import numpy as np
import cv2
import math

import peakutils
from astar import AStar


def first_nonzero(arr, axis, invalid_val=-1):
    try:
        mask = arr != 0
        im2 = np.where(mask.any(axis=axis),
                       mask.argmax(axis=axis), invalid_val)
        return np.min(im2[im2 != invalid_val])
    except ValueError:
        return 1


def last_nonzero(arr, axis, invalid_val=-1):
    try:
        mask = arr != 0
        val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        im2 = np.where(mask.any(axis=axis), val, invalid_val)
        return np.max(im2[im2 != invalid_val])
    except ValueError:
        return 1


class AStarPathFinder(AStar):

    def __init__(self, img, config={}):
        self.img = img
        self.step = 1
        self.lookahead = 100
        self.start = None
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.c = {
            "d": 150,
            "d2": 50,
            "m": 50,
            "v": 2,
            "n": 1
        }

    def heuristic_cost_estimate(self, current, goal):
        (y1, x1) = current
        (y2, x2) = goal
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        return self.c["d"] * self.D(n2) + self.c["d2"] * self.D2(n2) + self.c["m"] * self.M(n2) + self.c["v"] * self.V(n2) + self.c["n"] * self.N(n1, n2)

    def set_start(self, start):
        self.start = start

    def neighbors(self, node):
        x, y = node
        return[(nx, ny) for nx, ny in[(x, y - self.step), (x, y + self.step), (x - self.step, y), (x + self.step, y)] if 0 <= nx < self.width and 0 <= ny < self.height]

    def D(self, n):
        return 1.0/(1.0+np.min([self.d_u(n), self.d_d(n)]))

    def D2(self, n):
        return 1/(1+np.min([self.d_u(n), self.d_d(n)]) ** 2)

    def M(self, n):
        x, y = np.int32(n)
        return float(self.img[y, x] != 0)

    def V(self, n):
        return np.abs(n[1] - self.start[1])

    def N(self, current, neighbor):
        if (current[0] == neighbor[0] or current[1] == neighbor[1]):
            return 10.0
        else:
            return 14.0

    def d_u(self, n):
        x, y = np.int32(n)
        s = int(np.clip(y-self.lookahead, 0.0, self.height))
        dist = (y-s-1) - last_nonzero(self.img[s:y, x], 0)
        return np.abs(dist)

    def d_d(self, n):
        x, y = np.int32(n)
        e = int(np.clip(y+self.lookahead, 0.0, self.height))
        return np.abs(first_nonzero(self.img[y:e, x], 0))

    def find_path(self, y_start):
        start = (0, y_start)
        goal = (self.width-1, y_start)
        self.set_start(start)
        return self.astar(start, goal)
