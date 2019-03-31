import itertools
import random

import pytest


active = True


def pytest_addoption(parser):
    parser.addoption("--run-all-params", action="store_true", default=False,
                     help="run all parameters of tests")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if (active
            and not config.getoption("--run-all-params")
            and not config.getoption('--lf')):
        skip_param = pytest.mark.skip(
            reason="need --run-all-params option to run")
        paramed_items = {}
        for item in items:
            if (item.originalname is not None
                    and 'parametrize' in item.keywords
                    and 'skip' not in item.keywords):
                key = (item.fspath, item.originalname)
                i = paramed_items.get(key, [])
                i.append(item)
                paramed_items[key] = i

        for items in paramed_items.values():
            if items:
                run = random.randint(0, len(items)-1)
                for item in itertools.chain(items[:run], items[run+1:]):
                    item.add_marker(skip_param)
