import pytest

from tests.plugin import assert_any, run_one_param


conftest_assert_any = """
pytest_plugins = ['tests.plugin.assert_any']
"""

conftest_run_one_param = """
pytest_plugins = ['tests.plugin.run_one_param']
"""

pyfile_run_one_param = """
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(val):
    pass
"""


orig_run_one_param_active = None


def setup_module(module):
    global orig_run_one_param_active
    orig_run_one_param_active = run_one_param.active
    run_one_param.active = True


def teardown_module(module):
    run_one_param.active = orig_run_one_param_active


def test_assert_any(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(2))
def test_not_assert_any(val):
    pass


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    with assert_any():
        assert val == 1 + 1


@pytest.mark.parametrize('val', range(2))
def test_parametrized2(assert_any, val):
    assert_any.between(0, val, 1)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=9)


def test_assert_any_marker(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest

from tests.plugin import assert_any


@pytest.mark.assert_any
@pytest.mark.parametrize('val', range(3))
def test_parametrized(val):
    with assert_any.check():
        assert val == 1 + 1
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=4)


def test_not_assert_any_test(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest

from tests.plugin import assert_any


@pytest.mark.parametrize('val', range(3))
def test_parametrized(val):
    with assert_any.check():
        assert val == 1 + 1
""")

    result = testdir.runpytest()

    result.assert_outcomes(failed=3)


def test_assert_any_between(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    assert_any.between(0, 1, 2)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=3, failed=1)


def test_assert_any_between_sig_dig(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(1, 3))
def test_between1(assert_any, val):
    assert_any.between(0.991, val, 2.09)


@pytest.mark.parametrize('val', range(1, 3))
def test_between1_min(assert_any, val):
    assert_any.between(0.99, val, 2.09)


@pytest.mark.parametrize('val', range(1, 3))
def test_between1_max(assert_any, val):
    assert_any.between(0.991, val, 2.1)


@pytest.mark.parametrize('val', range(1, 3))
def test_between2(assert_any, val):
    assert_any.between(0.91, val, 2.9, sig_dig=1)


@pytest.mark.parametrize('val', range(1, 3))
def test_between2_min(assert_any, val):
    assert_any.between(0.9, val, 2.9, sig_dig=1)


@pytest.mark.parametrize('val', range(1, 3))
def test_between2_min(assert_any, val):
    assert_any.between(0.91, val, 3, sig_dig=1)


@pytest.mark.parametrize('val', range(1, 3))
def test_between3(assert_any, val):
    assert_any.between(0.9991, val, 2.009, sig_dig=3)


@pytest.mark.parametrize('val', range(1, 3))
def test_between3_min(assert_any, val):
    assert_any.between(0.999, val, 2.009, sig_dig=3)


@pytest.mark.parametrize('val', range(1, 3))
def test_between3_min(assert_any, val):
    assert_any.between(0.9991, val, 2.01, sig_dig=3)


def test_between_zero(assert_any):
    assert_any.between(1e-10, 1e-10, 1e-10)


def test_between_zero_neg(assert_any):
    assert_any.between(-1e-10, -1e-10, -1e-10)


def test_between_zero_min(assert_any):
    assert_any.between(0, 1e-10, 1e-10)


def test_between_zero_max(assert_any):
    assert_any.between(-1e-10, -1e-10, 0)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=24, failed=5)


def test_assert_any_between_nan(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    assert_any.between(0, float ('nan'), 2)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=3, failed=1)


def test_assert_any_failed(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    with assert_any():
        assert val == 1 + 2


@pytest.mark.parametrize('val', range(2))
def test_parametrized2(assert_any, val):
    assert_any.between(0, val, 2)


@pytest.mark.parametrize('val', range(2))
def test_parametrized3(assert_any, val):
    assert_any.between(-1, val, 1)


@pytest.mark.parametrize('val', range(2))
def test_parametrized4(assert_any, val):
    assert_any.between(-1, val, 2)


@pytest.mark.parametrize('val', range(2))
def test_parametrized5(assert_any, val):
    assert_any.between(2, val, 3)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=11, failed=5)


def test_assert_any_skipped(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.skip
@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    with assert_any():
        assert val == 1 + 2
""")

    result = testdir.runpytest('-rs')

    result.assert_outcomes(skipped=4)


def test_assert_any_deselected(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    with assert_any():
        assert val == 1 + 2
""")

    result = testdir.runpytest(
        '--deselect', 'test_assert_any_deselected.py::test_parametrized[1]',
        '-rs'
    )

    result.assert_outcomes(passed=2, skipped=1)


def test_assert_any_xfail(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val',
                         [1, 2,
                          pytest.param(
                              3,
                              marks=pytest.mark.xfail)])
def test_xfailed(assert_any, val):
    with assert_any():
        assert val == 3
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=2, xpassed=1, failed=1)


def test_assert_any_xfail_strict(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val',
                         [1, 2,
                          pytest.param(
                              3,
                              marks=pytest.mark.xfail(strict=True))])
def test_xfailed(assert_any, val):
    if val == 2:
        pytest.xfail()
    with assert_any.strict_check():
        assert val != 3
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=2, xfailed=2)


def test_assert_any_strict(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    with assert_any.strict_check():
        assert False
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=3, failed=1)


def test_assert_any_deselected_strict(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    with assert_any.strict_check():
        assert False
""")

    result = testdir.runpytest(
        '--deselect', 'test_assert_any_deselected_strict.py::'
                      'test_parametrized[1]',
    )

    result.assert_outcomes(passed=2, failed=1)


def test_assert_any_deselected_between(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    assert_any.between(0, 1, 2)
""")

    result = testdir.runpytest(
        '--deselect', 'test_assert_any_deselected_between.py::'
                      'test_parametrized[1]',
        '-rs',
        '--capture', 'no'
    )

    result.assert_outcomes(passed=2, skipped=1)


def test_assert_any_with_ff(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(2))
def test_parametrized(assert_any, val):
    assert_any.between(1, 0, 2)


@pytest.mark.parametrize('val', range(2))
def test_parametrized2(assert_any, val):
    assert_any.between(1, 0, 2)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=4, failed=2)

    assert_any._fixture = assert_any.AssertAnyFixture()

    result = testdir.runpytest('--ff')

    result.assert_outcomes(passed=4, failed=2)


def test_assert_any_with_lf(testdir):
    testdir.makeconftest(conftest_assert_any)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    assert_any.between(0, val, 2)


@pytest.mark.parametrize('val', range(2))
def test_parametrized2(assert_any, val):
    assert_any.between(0, val, 2)


@pytest.mark.parametrize(
    'val',
    [0, 1, pytest.param(2, marks=pytest.mark.skip)])
def test_parametrized3(assert_any, val):
    if val == 1:
        pytest.skip()
    assert_any.between(1, val, 2)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=7, failed=2, skipped=2)

    assert_any._fixture = assert_any.AssertAnyFixture()

    result = testdir.runpytest('--lf')

    result.assert_outcomes(passed=3, failed=2)


@pytest.mark.parametrize('dummy', range(10))
def test_run_one_param(testdir, dummy):
    testdir.makeconftest(conftest_run_one_param)
    testdir.makepyfile(pyfile_run_one_param)

    result = testdir.runpytest()

    result.assert_outcomes(passed=1, skipped=2)


def test_run_all_params(testdir):
    testdir.makeconftest(conftest_run_one_param)
    testdir.makepyfile(pyfile_run_one_param)

    result = testdir.runpytest('--run-all-params')

    result.assert_outcomes(passed=3)


def test_run_one_param_with_lf(testdir):
    testdir.makeconftest(conftest_run_one_param)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(val):
    assert False
""")

    result = testdir.runpytest()

    result.assert_outcomes(failed=1, skipped=2)

    result = testdir.runpytest('--lf')

    result.assert_outcomes(failed=1)

    result = testdir.runpytest('--run-all-params')

    result.assert_outcomes(failed=3)

    result = testdir.runpytest('--lf')

    result.assert_outcomes(failed=3)


def test_assert_any_with_run_one_param(testdir):
    testdir.makeconftest("""
pytest_plugins = [
    'tests.plugin.run_one_param',
    'tests.plugin.assert_any',
]
""")

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    assert_any.between(0, 1, 2)
""")

    result = testdir.runpytest('-rs')

    result.assert_outcomes(passed=1, skipped=3)


def test_assert_any_with_run_one_param_failed(testdir):
    testdir.makeconftest("""
pytest_plugins = [
    'tests.plugin.run_one_param',
    'tests.plugin.assert_any',
]
""")

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize('val', range(3))
def test_parametrized(assert_any, val):
    assert_any.between(1, 0, 2)
""")

    result = testdir.runpytest('-rs')

    result.assert_outcomes(passed=1, skipped=2, failed=1)
