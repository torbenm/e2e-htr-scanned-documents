import re
import numbers
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from config.config import Configuration
import os
from data.steps.pipes.warp import _warp
from data.steps.pipes.crop import _crop
from data.steps.pipes.padding import _pad_pil
from data.steps.pipes.affine import _affine
from data.steps.pipes.convert import _pil2cv2, _cv2pil

DEFAULTS = {
    'fonts': ['Arial', 'Times New Roman'],
    'height': {
        'min': 20,
        'max': 60,
        'center': 40,
        'scale': 5
    },
    'foreground': {
        'low': 0,
        'high': 100
    },
    'background': 255,
    'crop': False,
    'filters': [
        {
            'type': 'blur',
            'prob': 0.1,
            'radius': 1
        },
        {
            'type': 'sharpen',
            'prob': 0.1,
            'radius': 1,
            'threshold': 3,
            'percent': 150
        },
        {
            'type': 'warp',
            'prob': 0.5,
            'deviation': 2.7,
            'grid': [100, 30]
        },
        {
            'type': 'affine',
            'prob': 1.0,
            'config': {}
        }
    ],
    'padding': 0,
    'printing_padding': 0  # padding that is SOLELY applied when printing
}


PUNCTUATION_REGEX = re.compile(r"([|])(?=[,.;:!?])")
REGULAR_REGEX = re.compile(r"[|]")


class PrintGenerator(object):

    FILTERS = {
        'blur': lambda i, c: i.filter(ImageFilter.GaussianBlur(c['radius'])),
        'sharpen': lambda i, c: i.filter(ImageFilter.UnsharpMask(c['radius'], c['percent'], c['threshold'])),
        'warp': lambda i, c: _warp(i, c['grid'], c['deviation']),
        'affine': lambda i, c: PrintGenerator._affine_filter(i, c['config'])
    }

    def __init__(self, config={}):
        self.config = Configuration(config)
        self.default = Configuration(DEFAULTS)
        self.max_size = (0, 0)
        self.max_height = -1
        self.max_width = -1

    def __getitem__(self, key):
        default = self.default.default(key, None)
        return self.config.default(key, default)

    def _random_font(self):
        return self['fonts'][np.random.randint(0, len(self['fonts']))]

    def _random_height(self):
        return max(min(int(np.random.normal(self['height.center', True], self['height.scale', True])), self['height.max', True]), self['height.min', True])

    def _random_foreground(self):
        if self['foreground.low'] == self['foreground.high']:
            return self['foreground.low']
        return np.random.randint(self['foreground.low'], self['foreground.high'])

    def _iterate_height(self, text, fontname, height):
        font = ImageFont.truetype(fontname, size=height)
        size, offset = font.font.getsize(text)
        image_size = (size[0]+offset[0]+self['printing_padding']*2,
                      size[1]+offset[1]+self['printing_padding']*2)
        if self.max_height > -1 and image_size[1] > self.max_height:
            height = int(height*(self.max_height/float(image_size[1])))
            return self._iterate_height(text, fontname, height)
        elif self.max_width > -1 and image_size[0] > self.max_width:
            height = int(height*(self.max_width/float(image_size[0])))
            return self._iterate_height(text, fontname, height)
        else:
            return font, offset, image_size

    def _create_text_image(self, text, font, height, background, foreground):
        font, offset, image_size = self._iterate_height(text, font, height)
        self.max_size = np.max([self.max_size, image_size], axis=0)
        image = Image.new(
            "L", image_size, background)
        draw = ImageDraw.Draw(image)

        draw.text((self['printing_padding'], -(offset[1]/2) +
                   self['printing_padding']), text, font=font, fill=foreground)
        return image

    def _apply_filter(self, image, filter_config):
        if filter_config['prob'] > np.random.rand():
            image = self.FILTERS[filter_config['type']](image, filter_config)
        return image

    def _apply_filters(self, image):
        for _filter in self['filters']:
            image = self._apply_filter(image, _filter)
        return image

    def _crop(self, image, invert):
        if self['crop']:
            image = _pil2cv2(image)
            if not invert:
                image = 255 - image
            image = _crop(image)
            if not invert:
                image = 255 - image
            image = _cv2pil(image)
        return image

    def _pad(self, image, background):
        if self['padding'] > 0:
            image = _pad_pil(image, self['padding'], background)
        return image

    def __call__(self, text, invert=False):
        foreground = self._random_foreground()
        background = self['background']
        if invert:
            foreground = 255 - foreground
            background = 255 - background
        font = self._random_font()
        height = self._random_height()
        image = self._create_text_image(
            text, font, height, background, foreground)
        image = self._apply_filters(image)
        image = self._crop(image, invert)
        image = self._pad(image, invert)
        return image

    @staticmethod
    def clean_text(text):
        text = PUNCTUATION_REGEX.sub('', text)
        text = REGULAR_REGEX.sub(' ', text)
        return text

    @staticmethod
    def _affine_filter(image, config):
        image = _pil2cv2(image)
        image = _affine(image, config)
        return _cv2pil(image)


def generate_printed_samples(train_samples, dev_samples, test_samples, config, invert, path, target_height=-1, target_width=-1):
    count = [config['count']] * \
        3 if isinstance(config['count'], numbers.Number) else config['count']
    printed_train, train_max_size = _generate_printed_samples(
        train_samples, count[0], config, invert, path, target_height, target_width)
    printed_dev, dev_max_size = _generate_printed_samples(
        dev_samples, count[1], config, invert, path, target_height, target_width)
    printed_test, test_max_size = _generate_printed_samples(
        test_samples, count[2], config, invert, path, target_height, target_width)
    return printed_train, printed_dev, printed_test, np.max([train_max_size, dev_max_size, test_max_size], axis=0)


def _generate_printed_samples(ht_samples, count, config, invert, path, target_height=-1, target_width=-1):
    length = min(len(ht_samples), count)
    textsamples = ht_samples[:length]
    generator = PrintGenerator(config)
    generator.max_height = target_height
    generator.max_width = target_width
    full_samples = []
    for idx, sample in enumerate(textsamples):
        text = PrintGenerator.clean_text(sample['truth'])
        image = generator(text, invert)
        printedpath = os.path.join(path, 'printed-{}.png'.format(idx))
        image.save(printedpath)
        full_samples.append({"path": printedpath, "truth": 0})
        full_samples.append({"path": sample['path'], "truth": 1})
    return full_samples, generator.max_size
