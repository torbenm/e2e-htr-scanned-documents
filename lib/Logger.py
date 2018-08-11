import sys


class Logger(object):

    def progress(self, prefix, step, total):
        percent = (float(step) / total) * 100
        out = '{0} [ {2:100} ] {1:02.2f}% '.format(
            prefix, percent, "|" * int(percent))
        sys.stdout.write("\r" + out)
        sys.stdout.flush()

    def summary(self, prefix, summary):
        summary_part = "{} {:.3f}"
        summary_parts = [summary_part.format(
            key, sumamry[key]) for key in summary]
        msg = "{} | {}".format(prefix, " | ".join(summary_parts))
        sys.stdout.write('\r{:130}\n'.format(msg))
        sys.stdout.flush()
