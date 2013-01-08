import numpy as np
from numpy.testing import run_module_suite, assert_almost_equal
from skgeodesy import transform


class TestPolynomialTransform(object):

    def test_init(self):
        t = transform.PolynomialTransform(order=5)
        coords = [1, 2, 3]
        assert_almost_equal(t(coords), coords)

        t = t.before(t)
        assert_almost_equal(t(coords), coords)
        t = t.after(t)
        assert_almost_equal(t(coords), coords)


if __name__ == '__main__':
    run_module_suite()
