import collections
import inspect
import math
import pathlib

import pytest


def pytest_addoption(parser):
    parser.addoption("--force-assert-any",
                     action="store_true", default=False,
                     help="check assert_any tests always")


def pytest_configure(config):
    config.addinivalue_line('markers',
                            'assert_any:'
                            'Use assert_any plugin in test')


def is_assert_any_item(item):
    return 'assert_any' in item.fixturenames or 'assert_any' in item.keywords


def add_failed_item(item):
    global _fixture
    if is_assert_any_item(item):
        _fixture._failed[item.function] = True


def add_skipped_item(item):
    global _fixture
    if is_assert_any_item(item):
        _fixture._skipped_funcs[item.function] = True
        _fixture._skipped_items[item] = True


def check_item_skipped(item):
    if 'skip' in item.keywords:
        add_skipped_item(item)


def pytest_deselected(items):
    for item in items:
        add_skipped_item(item)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    global _always_check
    _always_check = config.getoption("--force-assert-any")

    last_index = {}
    for i in range(len(items)-1, -1, -1):
        item = items[i]
        if is_assert_any_item(item):
            check_item_skipped(item)
            if item.function not in last_index:
                last_index[item.function] = i
        elif isinstance(item, AssertAnyCheckItem):
            if item.function in last_index:
                index = last_index[item.function]
                while items[index].function is not item.function:
                    index -= 1
                items[i:index] = items[i+1:index+1]
                items[index] = item


@pytest.hookimpl(hookwrapper=True)
def pytest_pycollect_makeitem(collector, name, obj):
    outcome = yield
    res = outcome.get_result()
    if isinstance(res, list) and len(res) > 0:
        if is_assert_any_item(res[0]):
            outcome.force_result([
                *res,
                AssertAnyCheckItem(res)
            ])


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    _fixture.current_item = pyfuncitem
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    from _pytest._code.code import ExceptionInfo, TerminalRepr

    class AssertAnyCheckRepr(TerminalRepr):
        def __init__(self):
            super().__init__()
            self.reprs = []

        def append(self, exc_info, item):
            self.reprs.append((exc_info, item))

        def toterminal(self, tw):
            for exc_info, item in self.reprs:
                tw.sep('_ ', item.name, red=True, bold=True)
                (item.repr_failure(ExceptionInfo(exc_info))
                 .toterminal(tw))

    global _fixture
    outcome = yield
    report = outcome.get_result()

    if isinstance(item, AssertAnyCheckItem):
        if report.when == 'call':
            success = True
            repr = AssertAnyCheckRepr()
            if item.function in _fixture._failed:
                success = False
                repr = 'some tests are failed'
            elif (not _always_check
                  and item.function in _fixture._skipped_funcs):
                if item.function in _fixture.results:
                    for result in _fixture.results[item.function].values():
                        if not result.success and result.strict:
                            success = False
                            repr.append(result.excinfo, result.item)

                if success:
                    report.outcome = 'skipped'
                    report.longrepr = (item.location[0], item.lineno,
                                       'some tests are skipped or deselected')
            else:
                if item.function in _fixture.results:
                    for result in _fixture.results[item.function].values():
                        if not result.success:
                            success = False
                            repr.append(result.excinfo, result.item)

                if success:
                    report.outcome = 'passed'
            if not success:
                report.outcome = 'failed'
                report.longrepr = repr
                lfplugin = item.config.pluginmanager.get_plugin('lfplugin')
                for i in item.items:
                    if i not in _fixture._skipped_items:
                        lfplugin.lastfailed[i.nodeid] = True
        elif report.when == 'teardown':
            _fixture.clear(item)
    elif report.when == 'call':
        if report.outcome == 'skipped' and not hasattr(report, 'wasxfail'):
            add_skipped_item(item)
        if report.outcome == 'failed':
            add_failed_item(item)


class AssertAnyCheckItem(pytest.Item):
    def __init__(self, items):
        item = items[0]
        if item.originalname is not None:
            self.name = f'{item.originalname!s}[assert_any]'
        else:
            self.name = f'{item.name!s}[assert_any]'
        super().__init__(
            self.name,
            item.parent,
            item.config,
            item.session,
        )
        self.originalname = (item.originalname if item.originalname is not None
                             else item.name)
        self.fspath, self.lineno, _ = item.reportinfo()
        self.function = item.function
        self.fixturenames = []
        self.items = items

    def runtest(self):
        pass

    def reportinfo(self):
        return self.fspath, self.lineno, self.name


class DummyTraceback:
    def __init__(self, frame):
        self.tb_frame = frame
        self.tb_lineno = frame.f_lineno
        self.tb_next = None


class AssertAnyFixture:
    Result = collections.namedtuple('Result', ('success', 'excinfo', 'item',
                                               'strict'))

    def __init__(self):
        self.results = {}
        self.actuals = {}
        self._skipped_funcs = {}
        self._skipped_items = {}
        self._failed = {}
        self.current_item = None

    def get_traceback(self, depth=0, frame=None):
        if frame is None:
            frame = inspect.currentframe()
            for _ in range(depth+1):
                frame = frame.f_back
        tb = []

        last_current_frame = 0
        while frame is not None:
            tb.append(frame)
            if frame.f_code == self.current_item.function.__code__:
                last_current_frame = len(tb)
            frame = frame.f_back

        return tb[:last_current_frame]

    def get_callee(self, depth=1):
        tb = ''
        for frame in self.get_traceback(depth=depth+1):
            callee = inspect.getframeinfo(frame)
            callee_path = pathlib.Path(callee.filename).resolve()
            tb = (f'{callee_path!s}:{callee.lineno!s}:\n'
                  f'{tb!s}')

        return tb[:-2]

    def get_dummy_traceback(self, frame):
        tbs = self.get_traceback(frame=frame)
        top = None
        for tb in tbs:
            t = DummyTraceback(tb)
            t.tb_next = top
            top = t
        return top

    def excinfo(self):
        import sys
        type, value, tb = sys.exc_info()
        tb = self.get_dummy_traceback(tb.tb_frame)
        return (type, value, tb), self.current_item

    def _get_tb_value(self, tb, dict):
        current_func = self.current_item.function
        if current_func in dict:
            d = dict[current_func]
            if tb in d:
                return d[tb]
        return None

    def _set_tb_value(self, tb, dict, value):
        current_func = self.current_item.function
        d = {}
        if current_func in dict:
            d = dict[current_func]
        else:
            dict[current_func] = d

        d[tb] = value

    def get_result(self, tb):
        return self._get_tb_value(tb, self.results)

    def set_result(self, tb, success, excinfo, strict=False):
        if 'xfail' not in self.current_item.keywords:
            self._set_tb_value(
                tb, self.results,
                self.Result(success, excinfo, self.current_item, strict))

    def get_actual(self, tb):
        return self._get_tb_value(tb, self.actuals)

    def set_actual(self, tb, value):
        self._set_tb_value(tb, self.actuals, value)

    def clear(self, item):
        func = item.function
        for r in self.results.pop(func, {}).values():
            excinfo = r.excinfo
            if excinfo is not None:
                try:
                    excinfo[2].tb_frame.clear()
                except RuntimeError:
                    pass
        self.actuals.pop(func, None)

    def check_assert_any_test(self):
        __tracebackhide__ = True
        if not is_assert_any_item(self.current_item):
            pytest.fail('assert_any tests is required'
                        ' assert_any marker or fixture')

    class AssertAnyContext:
        def __init__(self, fixture, tb, strict, ignore):
            self._fixture = fixture
            self._tb = tb
            self._strict = strict
            self._ignore = ignore

        @property
        def actual(self):
            return self._fixture.get_actual(self._tb)

        @actual.setter
        def actual(self, value):
            self._fixture.set_actual(self._tb, value)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            result = self._fixture.get_result(self._tb)
            success = exc_type is None

            if 'xfail' in self._fixture.current_item.keywords:
                return not self._strict
            if (not self._ignore
                    and (self._strict and not success
                         or result is None
                         or not self._strict and not result.success)):
                exc_info = None
                if not success:
                    traceback = \
                        self._fixture.get_dummy_traceback(traceback.tb_frame)
                    exc_info = (exc_type, exc_value, traceback)

                self._fixture.set_result(self._tb,
                                         success,
                                         exc_info,
                                         strict=self._strict)
            return True

    def __call__(self, *args, **kwargs):
        return self.check(*args, **kwargs)

    def check(self, strict=False, tb=None, ignore=False):
        __tracebackhide__ = True
        self.check_assert_any_test()
        if tb is None:
            tb = self.get_callee()
        return self.AssertAnyContext(self, tb, strict, ignore)

    def strict_check(self, *args, **kwargs):
        kwargs['strict'] = True
        return self.check(*args, **kwargs)

    def check_actual(self, func, actual, *args, tb=None, **kwargs):
        __tracebackhide__ = True
        if tb is None:
            tb = self.get_callee()
        prev_actual = self.get_actual(tb)
        if prev_actual is None or func(actual, prev_actual):
            self.set_actual(tb, actual)
        else:
            kwargs['ignore'] = True
        return self.check(*args, tb=tb, **kwargs)

    def total_max(self, actual, *args, **kwargs):
        __tracebackhide__ = True
        return self.check_actual(lambda a, p: a > p, actual, *args, **kwargs)

    def total_min(self, actual, *args, **kwargs):
        __tracebackhide__ = True
        return self.check_actual(lambda a, p: a < p, actual, *args, **kwargs)

    def between(self, expect_min, actual, expect_max, sig_dig=2):
        __tracebackhide__ = True
        min_eps = (math.pow(10,
                            math.floor(math.log10(abs(expect_min))-sig_dig+1))
                   if expect_min != 0 else 0)

        max_eps = (math.pow(10,
                            math.floor(math.log10(abs(expect_max))-sig_dig+1))
                   if expect_max != 0 else 0)

        with self.total_min(actual, strict=True):
            assert expect_min <= actual, 'in minimum value'
        with self.total_min(actual):
            if expect_min != 0:
                assert expect_min+min_eps > actual, 'in minimum value'
            else:
                assert expect_min == actual, 'in minimum value'

        with self.total_max(actual, strict=True):
            assert expect_max >= actual, 'in maximum value'
        with self.total_max(actual):
            if expect_max != 0:
                assert expect_max-max_eps < actual, 'in maximum value'
            else:
                assert expect_max == actual, 'in maximum value'


@pytest.fixture(scope='module')
def assert_any():
    return _fixture


def check(*args, **kwargs):
    return _fixture.check(*args, **kwargs)


def strict_check(*args, **kwargs):
    return _fixture.strict_check(*args, **kwargs)


def total_max(actual, *args, **kwargs):
    return _fixture.strict_check(actual, *args, **kwargs)


def total_min(actual, *args, **kwargs):
    return _fixture.total_min(actual, *args, **kwargs)


def between(*args, **kwargs):
    __tracebackhide__ = True
    _fixture.between(*args, **kwargs)


_always_check = False
_fixture = AssertAnyFixture()
