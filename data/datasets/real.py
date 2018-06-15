from . import dataset
import os


class RealDataset(dataset.Dataset):

    identifier = "real"

    def getFilesAndTruth(self, basepath, subset, limit=-1):
        return self._load_files(os.path.join(basepath, self.identifier), limit)

    def getDownloadInfo(self):
        return {}

    def getIdentifier(self):
        return self.identifier

    def _load_files(self, basepath, limit=-1):
        # TODO: support limit
        extracted = []
        for subdir, dirs, files in os.walk(basepath):
            if len(files) > 0:
                for file in files:
                    if file[-3:] == "png" or file[-3:] == "PNG":
                        imgpath = os.path.join(subdir, file)
                        extracted.append({"path": imgpath, "truth": ""})
        return extracted, ""

    def _get_image_path(self, basepath, subset, identifier):
        idsplit = identifier.split("-")
        return os.path.join(basepath, self.identifier, subset, idsplit[0], "-".join(idsplit[0:2]), identifier + ".png")
