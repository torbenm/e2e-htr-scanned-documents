from lib.util.file import readJson


class GTProvider(object):

    def _load_gt(self, imgfile: str):
        # TODO: this is still very focused on the paper notes dataset
        # and probably should be done differently :)
        filepath = imgfile.replace(
            "-stripped.png", "-truth.json").replace("-paper.png", "-truth.json")
        return readJson(filepath)
