from lib.Configuration import Configuration
from lib.buildingblocks.evaluate.gtprovider.LineRegionGTProvider import LineRegionGTProvider
from lib.buildingblocks.evaluate.gtprovider.ParagraphRegionGTProvider import ParagraphRegionGTProvider
from lib.buildingblocks.evaluate.gtprovider.WordRegionGTProvider import WordRegionGTProvider

DEFAULTS = {
    "provider": None
}


class Ceiling(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULTS)

    def __call__(self, image, file):
        provider = self._parse_provider(self.config["provider"])
        return provider(file, image)

    def _parse_provider(self, provider):
        if provider is None:
            return None
        elif provider == "WordRegion":
            return WordRegionGTProvider()
        elif provider == "ParagraphRegion":
            return ParagraphRegionGTProvider()
        elif provider == "LineRegion":
            return LineRegionGTProvider()

    def close(self):
        pass
