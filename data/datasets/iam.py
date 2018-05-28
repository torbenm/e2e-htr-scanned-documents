import dataset
import os


class IamDataset(dataset.Dataset):

    identifier = "iam"

    def getFilesAndTruth(self, basepath, subset, limit=-1):
        return self._load_ascii_lines(basepath, subset, limit)

    def getDownloadInfo(self):
        download_info = {
            "lines": {
                "file": "lines.tgz",
                "protocol": "http",
                "url":  "www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz",
                "tgz": True,
                "authenticate": True
            },
            "words": {
                "file": "words.tgz",
                "protocol": "http",
                "url":  "www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz",
                "tgz": True,
                "authenticate": True
            },
            "ascii": {
                "file": "ascii.tgz",
                "protocol": "http",
                "url": "www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz",
                "tgz": True,
                "authenticate": True
            }
        }
        return download_info

    def getIdentifier(self):
        return self.identifier

    def _load_ascii_lines(self, basepath, type, limit=-1):
        with open(os.path.join(basepath, self.identifier, "ascii/{}.txt".format(type)), "r") as lines:
            parsed = []
            fulltext = ""
            i = 0
            while limit == -1 or i < limit:
                line = lines.readline().strip()
                if not line:
                    break
                if line[0] != "#":
                    lsplit = line.split(" ")
                    i = i + 1
                    if lsplit[1] != "err":
                        path = self._get_image_path(basepath, type, lsplit[0])
                        text = ' '.join(lsplit[8:])
                        parsed.append({"path": path, "truth": text})
                        fulltext = fulltext + text
            return parsed, fulltext

    def _get_image_path(self, basepath, subset, identifier):
        idsplit = identifier.split("-")
        return os.path.join(basepath, self.identifier, subset, idsplit[0], "-".join(idsplit[0:2]), identifier + ".png")
