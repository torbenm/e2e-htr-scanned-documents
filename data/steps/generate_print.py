import re
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from config.config import Configuration
import os

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
        }
    ],
    'padding': 0
}


PUNCTUATION_REGEX = re.compile(r"([|])(?=[,.;:!?])")
REGULAR_REGEX = re.compile(r"[|]")


class PrintGenerator(object):

    FILTERS = {
        'blur': lambda i, c: i.filter(ImageFilter.GaussianBlur(c['radius'])),
        'sharpen': lambda i, c: i.filter(ImageFilter.UnsharpMask(c['radius'], c['percent'], c['threshold']))
    }

    def __init__(self, config={}):
        self.config = Configuration(config)
        self.default = Configuration(DEFAULTS)
        self.max_size = (0, 0)

    def __getitem__(self, key):
        default = self.default.default(key, None)
        return self.config.default(key, default)

    def _random_font(self):
        return self['fonts'][np.random.randint(0, len(self['fonts']))]

    def _random_height(self):
        return max(min(int(np.random.normal(self['height.center'], self['height.scale'])), self['height.max']), self['height.min'])

    def _random_foreground(self):
        return np.random.randint(self['foreground.low'], self['foreground.high'])

    def _create_text_image(self, text, font, height, background, foreground):
        font = ImageFont.truetype(font, size=height)
        size, offset = font.font.getsize(text)
        image_size = (size[0]+offset[0]+self['padding']*2,
                      size[1]+offset[1]+self['padding']*2)
        self.max_size = np.max([self.max_size, image_size], axis=0)
        image = Image.new(
            "L", image_size, background)
        draw = ImageDraw.Draw(image)

        draw.text((self['padding'], -(offset[1]/2) +
                   self['padding']), text, font=font, fill=foreground)
        return image

    def _apply_filter(self, image, filter_config):
        if filter_config['prob'] > np.random.rand():
            image = self.FILTERS[filter_config['type']](image, filter_config)
        return image

    def _apply_filters(self, image):
        for _filter in self['filters']:
            image = self._apply_filter(image, _filter)
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
        return image

    @staticmethod
    def clean_text(text):
        text = PUNCTUATION_REGEX.sub('', text)
        text = REGULAR_REGEX.sub(' ', text)
        return text


def generate_printed_sampels(ht_samples, config, invert, path):
    length = min(len(ht_samples), config['count'])
    textsamples = ht_samples[:length]
    generator = PrintGenerator(config)
    full_samples = []
    for idx, sample in enumerate(textsamples):
        text = PrintGenerator.clean_text(sample['truth'])
        image = generator(text, invert)
        printedpath = os.path.join(path, 'printed-{}.png'.format(idx))
        image.save(printedpath)
        full_samples.append({"path": printedpath, "truth": 0})
        full_samples.append({"path": sample['path'], "truth": 1})
    return full_samples, generator.max_size