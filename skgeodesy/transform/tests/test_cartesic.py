import numpy as np
from numpy.testing import run_module_suite, assert_almost_equal
from skgeodesy import transform


def test_rotation_init():
    for axis in range(1, 4):
        for angle in np.linspace(-np.pi / 2, np.pi / 2, 10):
            t = transform.RotationTransform(angle=angle, axis=axis)
            assert_almost_equal(getattr(t, 'angle%d' % axis), angle)


def test_rotation_call():
    # mirror
    t = transform.RotationTransform(angle=np.pi, axis=1)
    assert_almost_equal(t([0, 1, 1]), [0, -1, -1])
    t = transform.RotationTransform(angle=np.pi, axis=2)
    assert_almost_equal(t([1, 0, 1]), [-1, 0, -1])
    t = transform.RotationTransform(angle=np.pi, axis=3)
    assert_almost_equal(t([1, 1, 0]), [-1, -1, 0])

    # 90 degree
    t = transform.RotationTransform(angle=np.pi / 2, axis=1)
    assert_almost_equal(t([0, 1, 1]), [0, 1, -1])
    t = transform.RotationTransform(angle=np.pi / 2, axis=2)
    assert_almost_equal(t([1, 0, 1]), [-1, 0, 1])
    t = transform.RotationTransform(angle=np.pi / 2, axis=3)
    assert_almost_equal(t([1, 1, 0]), [1, -1, 0])

    # 45 degree
    a = np.sqrt(2)
    t = transform.RotationTransform(angle=np.pi / 4, axis=1)
    assert_almost_equal(t([0, 1, 1]), [0, a, 0])
    t = transform.RotationTransform(angle=np.pi / 4, axis=2)
    assert_almost_equal(t([1, 0, 1]), [0, 0, a])
    t = transform.RotationTransform(angle=np.pi / 4, axis=3)
    assert_almost_equal(t([1, 1, 0]), [a, 0, 0])


if __name__ == '__main__':
    run_module_suite()
