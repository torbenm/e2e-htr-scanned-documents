import re
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from lib.Configuration import Configuration

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
    ]

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
        image = Image.new(
            "L", (size[0]+offset[0], size[1]+offset[1]), background)
        draw = ImageDraw.Draw(image)
        draw.text((0, -(offset[1]/2)), text, font=font, fill=foreground)
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


if __name__ == "__main__":
    texts = [
        'majority|in|Northern|Rhodesia|,|but|the',
        'this does not have somethin below baseline',
    ]

    generator = PrintGenerator()

    for idx, txt in enumerate(texts):
        img = generator(PrintGenerator.clean_text(txt), True)
        name = 'data/demo/printGen/output_{}.png'.format(idx)
        print(name)
        img.save(name)
