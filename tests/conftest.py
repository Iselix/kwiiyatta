import math

import pytest

import pytest_check


pytest_plugins = [
    'pytester',
    'tests.plugin.assert_any',
]


def pytest_addoption(parser):
    parser.addoption("--run-all", action="store_true", default=False,
                     help="run all tests")
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run slow tests")
    parser.addoption("--run-fullset", action="store_true", default=False,
                     help="run tests using fullset data")


def pytest_collection_modifyitems(config, items):
    import tests

    run_all_tests = config.getoption('--run-all')
    tests._skip_fullset = False
    if config.getoption('--lf'):
        lfplugin = config.pluginmanager.get_plugin('lfplugin')
        if len(lfplugin.lastfailed) > 0:
            return
    tests._skip_fullset = not (run_all_tests
                               or config.getoption("--run-fullset"))
    if run_all_tests or config.getoption("--run-slow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest_check.check_func
def check_round_equal(expect, actual, sig_dig=2, eps=1.0):
    eps = min(eps,
              (math.pow(10, math.floor(math.log10(abs(expect))-sig_dig+1))
               if expect != 0 else 0))
    assert expect <= actual < expect + eps


@pytest.fixture
def check(check):
    check.round_equal = check_round_equal
    return check
