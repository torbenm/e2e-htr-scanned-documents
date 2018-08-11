import sys


class Logger(object):

    def progress(prefix, step, total):
        percent = (float(step) / total) * 100
        out = '{0} [ {2:100} ] {1:02.2f}% '.format(
            prefix, percent, "|" * int(percent))
        sys.stdout.write("\r" + out)
        sys.stdout.flush()
