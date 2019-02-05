import sys


class Logger(object):

    def progress(self, prefix, step, total):
        percent = (float(step) / total) * 100
        out = '{0} [ {2:100} ] {1:02.2f}% '.format(
            prefix, percent, "|" * int(percent))
        self.write(out, False)

    def summary(self, prefix, summary):
        summary_part = "{} {:8.4f}"
        summary_parts = [summary_part.format(
            key, summary[key]) for key in summary]
        msg = "{} | {}".format(prefix, " | ".join(summary_parts))
        self.write(msg)

    def write(self, msg,  newline=True):
        suffix = '\n' if newline else ''
        sys.stdout.write('\r{:130}{}'.format(msg, suffix))
        sys.stdout.flush()
