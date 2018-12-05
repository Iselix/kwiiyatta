import argparse
import sys

import pytest

import kwiiyatta


@pytest.fixture(scope='function', autouse=True)
def scope_function():
    sys_argv = sys.argv
    sys.argv = ["kwiiyatta"]
    yield
    sys.argv = sys_argv


def test_empty_argument():
    conf = kwiiyatta.Config()
    conf.parse_args()

    assert conf.frame_period == 5


def test_argument():
    conf = kwiiyatta.Config()
    conf.parse_args(['--frame-period', '10'])

    assert conf.frame_period == 10


def test_pass_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-arg', type=str)
    conf = kwiiyatta.Config(parser)
    conf.parse_args(['--frame-period', '20', '--test-arg', 'hogehoge'])

    assert conf.frame_period == 20
    assert conf.test_arg == 'hogehoge'


def test_sys_argv():
    sys.argv += ['--frame-period', '30']
    conf = kwiiyatta.Config()
    conf.parse_args()

    assert conf.frame_period == 30


def test_sys_argv_and_parse():
    sys.argv += ['--frame-period', '40']
    conf = kwiiyatta.Config()
    conf.parse_args(['--frame-period', '50'])

    assert conf.frame_period == 40


@pytest.mark.parametrize('args,frame_period',
                         [([], 5),
                          (['--frame-period', '10'], 10)])
def test_create_analyzer(args, frame_period):
    print(f'sys argv:{sys.argv!r}')
    conf = kwiiyatta.Config()
    conf.parse_args(args)
    analyzer = conf.create_analyzer(kwiiyatta.Wavdata(0, None))

    assert analyzer.frame_period == frame_period
