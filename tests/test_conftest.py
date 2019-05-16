import pathlib
import shutil


CONFTEST_PATH = pathlib.Path(__file__).parent/'conftest.py'


def test_check_round_equal_passed(testdir):
    shutil.copy(CONFTEST_PATH, testdir.tmpdir)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize(
    'expect,actual',
    [
        (0.111, 0.111),
        (0.111, 0.11209),
        (0.11,  0.11),
        (0.11,  0.119),
        (0.1,   0.1),
        (0.1,   0.109),
        (1,     1),
        (1,     1.09),
        (11,    11),
        (11,    11.9),
        (111,   111),
        (111,   111.9),
    ])
def test_check(check, expect, actual):
    check.round_equal(expect, actual)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=12)


def test_check_round_equal_failed(testdir):
    shutil.copy(CONFTEST_PATH, testdir.tmpdir)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize(
    'expect,actual',
    [
        (0.111, 0.1109),
        (0.111, 0.121),
        (0.11,  0.109),
        (0.11,  0.12),
        (0.1,   0.09),
        (0.1,   0.2),
        (1,     0.9),
        (1,     2),
        (11,    10.9),
        (11,    12),
        (111,   110.9),
        (111,   112),
    ])
def test_check(check, expect, actual):
    check.round_equal(expect, actual)
""")

    result = testdir.runpytest()

    result.assert_outcomes(failed=12)


def test_check_round_equal_sig_dig_passed(testdir):
    shutil.copy(CONFTEST_PATH, testdir.tmpdir)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize(
    'sig_dig,expect,actual',
    [
        (3, 0.11, 0.1109),
        (2, 0.11, 0.119),
        (1, 0.11, 0.20),
        (1, 111, 111.9),
        (3, 111, 111.9),
        (4, 111, 111.09),
    ])
def test_check(check, sig_dig, expect, actual):
    check.round_equal(expect, actual, sig_dig=sig_dig)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=6)


def test_check_round_equal_sig_dig_failed(testdir):
    shutil.copy(CONFTEST_PATH, testdir.tmpdir)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize(
    'sig_dig,expect,actual',
    [
        (3, 0.11, 0.111),
        (2, 0.11, 0.12),
        (1, 0.11, 0.211),
        (1, 111, 112),
        (3, 111, 112),
        (4, 111, 111.1),
    ])
def test_check(check, sig_dig, expect, actual):
    check.round_equal(expect, actual, sig_dig=sig_dig)
""")

    result = testdir.runpytest()

    result.assert_outcomes(failed=6)


def test_check_round_equal_eps_passed(testdir):
    shutil.copy(CONFTEST_PATH, testdir.tmpdir)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize(
    'eps,expect,actual',
    [
        (0.001, 0.11, 0.1109),
        (0.01,  0.11, 0.119),
        (0.1,   0.11, 0.119),
        (0.1, 111, 111.09),
        (1, 111, 111.9),
        (10, 111, 120.9),
    ])
def test_check(check, eps, expect, actual):
    check.round_equal(expect, actual, eps=eps)
""")

    result = testdir.runpytest()

    result.assert_outcomes(passed=6)


def test_check_round_equal_eps_failed(testdir):
    shutil.copy(CONFTEST_PATH, testdir.tmpdir)

    testdir.makepyfile("""
import pytest


@pytest.mark.parametrize(
    'eps,expect,actual',
    [
        (0.001, 0.11, 0.111),
        (0.01,  0.11, 0.12),
        (0.1,   0.11, 0.12),
        (0.1, 111, 111.1),
        (1, 111, 112),
        (10, 111, 121),
    ])
def test_check(check, eps, expect, actual):
    check.round_equal(expect, actual, eps=eps)
""")

    result = testdir.runpytest()

    result.assert_outcomes(failed=6)
