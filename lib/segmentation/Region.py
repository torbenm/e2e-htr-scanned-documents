import numpy as np


class Region(object):

    values = None
    cls_score = None

    text = None
    cls = None

    def __init__(self, **kwargs):
        self.path = kwargs.get('path', [])
        if len(self.path) > 0:
            self._pos_and_size_from_path()
        else:
            self.pos = kwargs.get('pos', (0, 0))
            self.size = kwargs.get('size', (0, 0))
        self.img = kwargs.get('img', None)

    def _pos_and_size_from_path(self):
        np_path = np.array(self.path)
        self.pos = (np.min(np_path[:, 0]), np.min(np_path[:, 1]))
        end_pos = (np.max(np_path[:, 0]), np.max(np_path[:, 1]))
        self.size = (end_pos[0] - self.pos[0], end_pos[1] - self.pos[1])

    def translate(self, shift):
        xshift, yshift = shift
        self.pos = (self.pos[0]+xshift, self.pos[0]+yshift)
        self.path = [(x+xshift, y+yshift) for x, y in self.path]

    def set_class(self, score, cls):
        self.cls = cls
        self.cls_score = score

    def set_text(self, values, text):
        self.values = values
        self.text = text

    def remove_img(self):
        self.img = None

    def get_bottom_right(self):
        return (self.pos[0]+self.size[0], self.pos[1]+self.size[1])
