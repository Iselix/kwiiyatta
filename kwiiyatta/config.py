import argparse
import sys

import kwiiyatta


class Config:
    def __init__(self, argparser=None):
        if argparser is None:
            argparser = argparse.ArgumentParser()
        argparser.add_argument('--frame-period', type=int, default=5,
                               help='Frame period milli-seconds of vocoder')
        self.parser = argparser

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, args=None):
        if args is None:
            args = sys.argv[1:]
        else:
            args = args + sys.argv[1:]
        self.parser.parse_args(args=args, namespace=self)

    def create_analyzer(self, *args, Analyzer=None, **kwargs):
        if Analyzer is None:
            Analyzer = kwiiyatta.Analyzer
        kwargs['frame_period'] = self.frame_period
        return Analyzer(*args, **kwargs)
