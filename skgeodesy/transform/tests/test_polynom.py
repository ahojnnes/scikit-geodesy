import numpy as np
from numpy.testing import run_module_suite, assert_almost_equal
from skgeodesy import transform
from skgeodesy.transform.polynom import _solve_for_order, _solve_for_num_coeffs


class TestPolynomialTransform(object):

    def test_init(self):
        t = transform.PolynomialTransform(order=5)
        coords = [1, 2, 3]
        assert_almost_equal(t(coords), coords)

        t = t.before(t)
        assert_almost_equal(t(coords), coords)
        t = t.after(t)
        assert_almost_equal(t(coords), coords)


def test_solve_for_order():
    assert _solve_for_order(60) == 3
    assert _solve_for_order(105) == 4
    assert _solve_for_order(360) == 7


def test_solve_for_num_coeffs():
    assert _solve_for_num_coeffs(3) == 60
    assert _solve_for_num_coeffs(4) == 105
    assert _solve_for_num_coeffs(7) == 360


if __name__ == '__main__':
    run_module_suite()
