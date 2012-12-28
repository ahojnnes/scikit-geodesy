import numpy as np
from numpy.testing import run_module_suite, assert_almost_equal
from skgeodesy import transform


AXIS_NAMES = ['x', 'y', 'z']


class TestTranslationTransform(object):

    def test_init(self):
        for axis in (1, 2, 3):
            for translation in np.linspace(-100, 100, 20):
                t = transform.TranslationTransform(translation=translation,
                                                   axis=axis)
                assert_almost_equal(getattr(t, 't%s' % AXIS_NAMES[axis-1]),
                                    translation)

    def test_call(self):
        t1 = transform.TranslationTransform(translation=0.1, axis=1)
        assert_almost_equal(t1([1, 1, 1]), [1.1, 1, 1])
        t2 = transform.TranslationTransform(translation=0.1, axis=2)
        assert_almost_equal(t2([1, 1, 1]), [1, 1.1, 1])
        t3 = transform.TranslationTransform(translation=0.1, axis=3)
        assert_almost_equal(t3([1, 1, 1]), [1, 1, 1.1])
        t = t1.before(t2).before(t3)
        assert_almost_equal(t([1, 1, 1]), [1.1, 1.1, 1.1])

    def test_inverse(self):
        t = transform.TranslationTransform(translation=0.1, axis=1)
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestScaleTransform(object):

    def test_init(self):
        for axis in (1, 2, 3):
            for scale in np.linspace(0.1, 4, 10):
                t = transform.ScaleTransform(scale=scale, axis=axis)
                tscale = getattr(t, 's%s' % AXIS_NAMES[axis-1])
                assert_almost_equal(tscale, scale)

    def test_call(self):
        t1 = transform.ScaleTransform(scale=0.1, axis=1)
        assert_almost_equal(t1([1, 1, 1]), [0.1, 1, 1])
        t2 = transform.ScaleTransform(scale=0.1, axis=2)
        assert_almost_equal(t2([1, 1, 1]), [1, 0.1, 1])
        t3 = transform.ScaleTransform(scale=0.1, axis=3)
        assert_almost_equal(t3([1, 1, 1]), [1, 1, 0.1])
        t = t1.before(t2).before(t3)
        assert_almost_equal(t([1, 1, 1]), [0.1, 0.1, 0.1])

    def test_inverse(self):
        t = transform.ScaleTransform(scale=0.1, axis=1)
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestRotationTransform(object):

    def test_init(self):
        for axis in (1, 2, 3):
            for angle in np.linspace(-np.pi / 2, np.pi / 2, 10):
                t = transform.RotationTransform(angle=angle, axis=axis)
                tangle = getattr(t, 'r%s' % AXIS_NAMES[axis-1])
                assert_almost_equal(tangle, angle)

    def test_call(self):
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

    def test_inverse(self):
        t = transform.RotationTransform(angle=1, axis=1)
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestShearTransform(object):

    def test_call(self):
        t1 = transform.ShearTransform(shear=(2, 2), axis=1)
        assert_almost_equal(t1([1, 1, 1]), [1, 3, 3])
        t2 = transform.ShearTransform(shear=(2, 2), axis=2)
        assert_almost_equal(t2([1, 1, 1]), [3, 1, 3])
        t3 = transform.ShearTransform(shear=(2, 2), axis=3)
        assert_almost_equal(t3([1, 1, 1]), [3, 3, 1])

        t = t1.before(t2).before(t3)
        assert_almost_equal(t([1, 1, 1]), [25, 21, 9])
        t = t2.before(t1).before(t3)
        assert_almost_equal(t([1, 1, 1]), [21, 25, 9])
        t = t3.before(t1).before(t2)
        assert_almost_equal(t([1, 1, 1]), [21, 9, 25])

    def test_inverse(self):
        t = transform.ShearTransform(shear=(1, 2), axis=1)
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestEuclideanTransform(object):

    def test_init(self):
        translation = np.zeros((3, ))
        for rot in np.linspace(-np.pi / 2, np.pi / 2, 5):
            angle = (rot, rot, rot)
            for trans in np.linspace(-100, 100, 10):
                translation[:] = trans
                t = transform.EuclideanTransform(angle=angle,
                                                 translation=translation)
                assert_almost_equal(t.rx, rot)
                assert_almost_equal(t.ry, rot)
                assert_almost_equal(t.rz, rot)
                assert_almost_equal(t.tx, trans)
                assert_almost_equal(t.ty, trans)
                assert_almost_equal(t.tz, trans)

    def test_call(self):
        t = transform.EuclideanTransform(angle=(np.pi, np.pi, np.pi),
                                         translation=(10, 10, 10))
        assert_almost_equal(t([1, 1, 1]), [11, 11, 11])

    def test_inverse(self):
        t = transform.EuclideanTransform(angle=(1, 2, 3),
                                         translation=(1, 2, 3))
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestSimilarityTransform(object):

    def test_init(self):
        translation = np.zeros((3, ))
        for rot in np.linspace(-np.pi / 2, np.pi / 2, 5):
            angle = (rot, rot, rot)
            for scale in np.linspace(0.1, 4, 5):
                for trans in np.linspace(-100, 100, 10):
                    translation[:] = trans
                    t = transform.SimilarityTransform(scale=scale, angle=angle,
                                                      translation=translation)
                    assert_almost_equal(t.s, scale)
                    assert_almost_equal(t.rx, rot)
                    assert_almost_equal(t.ry, rot)
                    assert_almost_equal(t.rz, rot)
                    assert_almost_equal(t.tx, trans)
                    assert_almost_equal(t.ty, trans)
                    assert_almost_equal(t.tz, trans)

    def test_call(self):
        t = transform.SimilarityTransform(angle=(np.pi, np.pi, np.pi),
                                          translation=(10, 10, 10))
        assert_almost_equal(t([1, 1, 1]), [11, 11, 11])
        t = transform.SimilarityTransform(scale=2, angle=(np.pi, np.pi, np.pi),
                                          translation=(10, 10, 10))
        assert_almost_equal(t([1, 1, 1]), [12, 12, 12])
        t = transform.SimilarityTransform(scale=0.5)
        assert_almost_equal(t([1, 1, 1]), [0.5, 0.5, 0.5])

    def test_inverse(self):
        t = transform.SimilarityTransform(scale=0.1, angle=(1, 2, 3),
                                          translation=(1, 2, 3))
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestAffineTransform(object):

    def test_init(self):
        translation = np.zeros((3, ))
        for rot in np.linspace(-np.pi / 2, np.pi / 2, 5):
            angle = (rot, rot, rot)
            for sx in np.linspace(0.1, 4, 5):
                for trans in np.linspace(-100, 100, 10):
                    translation[:] = trans
                    sy = sx + 0.1
                    sz = sx + 0.2
                    t = transform.AffineTransform(scale=(sx, sy, sz),
                                                  angle=angle,
                                                  translation=translation)
                    assert_almost_equal(t.sx, sx)
                    assert_almost_equal(t.sy, sy)
                    assert_almost_equal(t.sz, sz)
                    assert_almost_equal(t.rx, rot)
                    assert_almost_equal(t.ry, rot)
                    assert_almost_equal(t.rz, rot)
                    assert_almost_equal(t.tx, trans)
                    assert_almost_equal(t.ty, trans)
                    assert_almost_equal(t.tz, trans)

    def test_inverse(self):
        t = transform.AffineTransform(scale=(1, 2, 3), angle=(1, 2, 3),
                                      translation=(1, 2, 3),
                                      shear=((1, 2), (3, 4), (5, 6)))
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


if __name__ == '__main__':
    run_module_suite()
