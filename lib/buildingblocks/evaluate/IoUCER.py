from lib.buildingblocks.evaluate.IoU import IoU
import pylev
import re

DEFAULT_CONFIG = {
    "threshold": 0.5,
    "punctuation_regex": re.compile(r"([|])(?=[,.;:!?])"),
    "regular_regex": re.compile(r"[|]"),
    "filter_class": True,
    "target_class": 1
}


class IoUCER(IoU):

    def __init__(self, config):
        super().__init__(config, DEFAULT_CONFIG)

    def _cer(self, gt, pred):
        ln = float(max(len(gt), len(pred)))
        return pylev.levenshtein(gt, pred)/ln

    def _clean_text(self, text):
        text = self.config["punctuation_regex"].sub('', text)
        text = self.config["regular_regex"].sub(' ', text)
        return text

    def _score_fn(self, dist):
        return 1.0 - (dist if dist < 1.0 else 1.0)

    def _calc_score(self, hits, misfire, nofire):
        # GT that has no predicitons, but consist only of 1 char are ignored
        nofire = list(filter(lambda x: len(x.text) > 1, nofire))
        total_len = len(hits) + len(nofire) + len(misfire)
        hit_score = 0
        for hit in hits:
            dist = self._cer(self._clean_text(hit["gt"].text).lower(),
                             self._clean_text(hit["pred"].text).lower())
            hit_score += self._score_fn(dist)
        return ({"ioucer": hit_score/total_len}, hits, misfire, nofire)
