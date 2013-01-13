import numpy as np
from numpy.testing import run_module_suite, assert_equal, assert_almost_equal
from skgeodesy import util


def test_wrap_to_pi():
    # test wrapping
    assert_almost_equal(util.wrap_to_pi(np.pi + 0.1), -np.pi + 0.1)
    assert_almost_equal(util.wrap_to_pi(-np.pi + 0.1), -np.pi + 0.1)
    assert_almost_equal(util.wrap_to_pi(3 * np.pi + 0.1), -np.pi + 0.1)
    assert_almost_equal(util.wrap_to_pi(-3 * np.pi + 0.1), -np.pi + 0.1)

    # test multi-dimensional input
    angles = np.zeros((10, 10, 3))
    assert_almost_equal(util.wrap_to_pi(angles), angles)
    assert_almost_equal(util.wrap_to_pi(angles + 0.1), angles + 0.1)


def test_wrap_to_2pi():
    # test wrapping
    assert_almost_equal(util.wrap_to_2pi(np.pi + 0.1), np.pi + 0.1)
    assert_almost_equal(util.wrap_to_2pi(-np.pi + 0.1), np.pi + 0.1)
    assert_almost_equal(util.wrap_to_2pi(3 * np.pi + 0.1), np.pi + 0.1)
    assert_almost_equal(util.wrap_to_2pi(-3 * np.pi + 0.1), np.pi + 0.1)

    # test multi-dimensional input
    angles = np.zeros((10, 10, 3))
    assert_almost_equal(util.wrap_to_2pi(angles), angles)
    assert_almost_equal(util.wrap_to_2pi(angles + 0.1), angles + 0.1)


def test_deg2dms():
    assert_almost_equal(util.deg2dms(12 + 30 / 60.), [12, 30, 0])
    assert_almost_equal(util.deg2dms(12 + 30 / 60. + 30 / 3600.), [12, 30, 30])
    assert_almost_equal(util.deg2dms(-12 - 30 / 60. - 30 / 3600.),
                        [-12, -30, -30])
    assert_almost_equal(util.dms2deg(util.deg2dms(12.123)), 12.123)


def test_dms2deg():
    assert_almost_equal(util.dms2deg([12, 30, 0]), 12 + 30 / 60.)
    assert_almost_equal(util.dms2deg([12, 30, 30]), 12 + 30 / 60. + 30 / 3600.)
    assert_almost_equal(util.dms2deg([-12, -30, -30]),
                        -12 - 30 / 60. - 30 / 3600.)
    assert_almost_equal(util.deg2dms(util.dms2deg([12, 1, 2])), [12, 1, 2])


if __name__ == '__main__':
    run_module_suite()
