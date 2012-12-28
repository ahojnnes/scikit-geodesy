import numpy as np
from numpy.testing import run_module_suite, assert_almost_equal
from skgeodesy import transform


def test_rotation_init():
    axis_idx = ['x', 'y', 'z']
    for axis in range(1, 4):
        for angle in np.linspace(-np.pi / 2, np.pi / 2, 10):
            t = transform.RotationTransform(angle=angle, axis=axis)
            assert_almost_equal(getattr(t, 'r%s' % axis_idx[axis-1]), angle)


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


def test_rotation_inverse():
    t = transform.RotationTransform(angle=1, axis=1)
    tinv = t.inverse()
    coord = [1, 2, 3]
    assert_almost_equal(coord, tinv(t(coord)))


def test_similarity_init():
    angles = np.zeros((3, ))
    translation = np.zeros((3, ))
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 10):
        angles[:] = angle
        for scale in np.linspace(0.1, 4, 10):
            for trans in np.linspace(-100, 100, 30):
                translation[:] = trans
                t = transform.SimilarityTransform(scale=scale, angles=angles,
                                                  translation=translation)
                assert_almost_equal(t.scale, scale)
                assert_almost_equal(t.rx, angle)
                assert_almost_equal(t.ry, angle)
                assert_almost_equal(t.rz, angle)
                assert_almost_equal(t.tx, trans)
                assert_almost_equal(t.ty, trans)
                assert_almost_equal(t.tz, trans)


def test_similarity_call():
    t = transform.SimilarityTransform(angles=(np.pi, np.pi, np.pi),
                                      translation=(10, 10, 10))
    assert_almost_equal(t([1, 1, 1]), [11, 11, 11])
    t = transform.SimilarityTransform(scale=2, angles=(np.pi, np.pi, np.pi),
                                      translation=(10, 10, 10))
    assert_almost_equal(t([1, 1, 1]), [12, 12, 12])
    t = transform.SimilarityTransform(scale=0.5)
    assert_almost_equal(t([1, 1, 1]), [0.5, 0.5, 0.5])


def test_similarity_inverse():
    t = transform.SimilarityTransform(scale=0.1, angles=(1, 2, 3),
                                      translation=(1, 2, 3))
    tinv = t.inverse()
    coord = [1, 2, 3]
    assert_almost_equal(coord, tinv(t(coord)))


if __name__ == '__main__':
    run_module_suite()
