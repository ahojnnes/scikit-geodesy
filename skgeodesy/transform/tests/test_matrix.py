import numpy as np
from numpy.testing import run_module_suite, assert_almost_equal
from skgeodesy import transform


class TestTranslationTransform(object):

    def test_init(self):
        for axis in (1, 2, 3):
            for translation in np.linspace(-100, 100, 20):
                t = transform.TranslationTransform(translation=translation,
                                                   axis=axis)
                assert_almost_equal(t.translation[axis - 1], translation)

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
                assert_almost_equal(t.scale[axis - 1], scale)

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
                assert_almost_equal(t.rotation[axis - 1], angle)

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
        t1 = transform.ShearTransform(shear=2, axis=12)
        assert_almost_equal(t1([1, 1, 1]), [1, 3, 1])
        t2 = transform.ShearTransform(shear=2, axis=21)
        assert_almost_equal(t2([1, 1, 1]), [3, 1, 1])
        t3 = transform.ShearTransform(shear=2, axis=32)
        assert_almost_equal(t3([1, 1, 1]), [1, 3, 1])

        t = t1.before(t2).before(t3)
        assert_almost_equal(t([1, 1, 1]), [7, 5, 1])
        t = t2.before(t1).before(t3)
        assert_almost_equal(t([1, 1, 1]), [3, 9, 1])
        t = t3.before(t1).before(t2)
        assert_almost_equal(t([1, 1, 1]), [11, 5, 1])

    def test_inverse(self):
        t = transform.ShearTransform(shear=3, axis=13)
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestPerspectiveTransform(object):

    def test_init(self):
        for axis in (1, 2, 3):
            for perspective in np.linspace(-10, 10, 10):
                t = transform.PerspectiveTransform(perspective=perspective,
                                                   axis=axis)
                assert_almost_equal(t.perspective[axis - 1], perspective)

    def test_call(self):
        t1 = transform.PerspectiveTransform(perspective=3, axis=1)
        assert_almost_equal(t1([1, 1, 1]), [0.25, 0.25, 0.25])
        t2 = transform.PerspectiveTransform(perspective=3, axis=2)
        assert_almost_equal(t2([1, 1, 1]), [0.25, 0.25, 0.25])
        t3 = transform.PerspectiveTransform(perspective=3, axis=3)
        assert_almost_equal(t3([1, 1, 1]), [0.25, 0.25, 0.25])

        t = t1.before(t2).before(t3)
        assert_almost_equal(t([1, 1, 1]), [0.1, 0.1, 0.1])

    def test_inverse(self):
        t = transform.PerspectiveTransform(perspective=3, axis=1)
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
                assert_almost_equal(t.rotation, angle)
                assert_almost_equal(t.translation, translation)

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
                    assert_almost_equal(t.scale, scale)
                    assert_almost_equal(t.rotation, angle)
                    assert_almost_equal(t.translation, translation)

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
        for rot in np.linspace(-np.pi / 2, np.pi / 2, 5):
            angle = (rot, rot, rot)
            for sx in np.linspace(0.1, 4, 5):
                scale = (sx, sx + 0.1, sx + 0.2)
                for trans in np.linspace(-100, 100, 10):
                    translation = (trans, trans, trans)

                    t = transform.AffineTransform(scale=scale,
                                                  angle=angle,
                                                  translation=translation)
                    assert_almost_equal(t.scale, scale)
                    assert_almost_equal(t.rotation, angle)
                    assert_almost_equal(t.translation, translation)

    def test_inverse(self):
        t = transform.AffineTransform(scale=(1, 2, 3), angle=(1, 2, 3),
                                      translation=(1, 2, 3),
                                      shear=((13, 1), (32, 2), (21, 3)))
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


class TestProjectiveTransform(object):

    def test_init(self):
        for rot in np.linspace(-np.pi / 3, np.pi / 3, 6):
            angle = (rot, rot, rot)
            for sx in np.linspace(0.1, 4, 5):
                scale = (sx, sx + 0.1, sx + 0.2)
                for trans in np.linspace(-100, 100, 5):
                    translation = (trans, trans, trans)
                    for persp in np.linspace(-10, 10, 5):
                        perspective = (persp, persp, persp)
                        t = transform.ProjectiveTransform(scale=scale,
                                                      angle=angle,
                                                      translation=translation,
                                                      perspective=perspective)
                        assert_almost_equal(t.scale, scale)
                        assert_almost_equal(t.rotation, angle)
                        assert_almost_equal(t.translation, translation)
                        assert_almost_equal(t.perspective[:3], perspective)

    def test_inverse(self):
        t = transform.ProjectiveTransform(scale=(1, 2, 3), angle=(1, 2, 3),
                                          translation=(1, 2, 3),
                                          shear=((13, 1), (32, 2), (21, 3)),
                                          perspective=(0.1, 0.2, 0.3))
        tinv = t.inverse()
        coord = [1, 2, 3]
        assert_almost_equal(coord, tinv(t(coord)))


if __name__ == '__main__':
    run_module_suite()
