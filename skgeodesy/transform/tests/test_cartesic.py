import numpy as np
from numpy.testing import run_module_suite, assert_almost_equal
from skgeodesy import transform


def test_rotation():
    for axis in range(1, 4):
        for angle in np.linspace(-np.pi / 2, np.pi / 2, 10):
            t = transform.RotationTransform(angle, axis)
            assert_almost_equal(getattr(t, 'angle%d' % axis), angle)



if __name__ == '__main__':
    run_module_suite()
