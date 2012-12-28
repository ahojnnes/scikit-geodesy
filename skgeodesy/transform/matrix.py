import numpy as np


def _check_axis(axis):
    if axis not in (1, 2, 3):
        raise ValueError('Axis must be 1, 2 or 3.')


def _extract_shear_axis(axis):
    error = False
    axis = str(axis)
    if len(axis) != 2:
        raise ValueError('Axis must be a combination of 1, 2 and 3, e.g. 12')
    else:
        axis1, axis2 = int(axis[0]), int(axis[1])
        _check_axis(axis1)
        _check_axis(axis2)
        if axis1 == axis2:
            raise ValueError('Axes must not be the same.')
        return axis1, axis2


class MatrixTransform(object):

    def __init__(self, matrix=None):
        """Create cartesic transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix. An identity matrix is created by
            default.

        """

        if matrix is None:
            matrix = np.identity(4, dtype=np.double)
        self.matrix = matrix

    def __call__(self, coords):
        """Apply transform to coordinates.

        Parameters
        ----------
        coords : (N, 2) or (N, 3) array_like
            2D or 3D coordinates. If z-component is not given it is set to 0.

        Returns
        -------
        out : (N, 2) or (N, 3) array_like
            Transformed 2D or 3D coordinates.

        """

        coords = np.array(coords, copy=False)
        input_ndim = coords.ndim
        coords = np.atleast_2d(coords)
        input_is_2D = coords.shape[1] == 2

        if input_is_2D:
            x, y = np.transpose(coords)
            z = np.zeros_like(x)
        else:
            x, y, z = np.transpose(coords)

        src = np.vstack((x, y, z, np.ones_like(x)))
        dst = np.dot(src.transpose(), self.matrix.transpose())

        # rescale to homogeneous coordinates
        dst[:, 0] /= dst[:, 3]
        dst[:, 1] /= dst[:, 3]
        dst[:, 2] /= dst[:, 3]

        if input_is_2D:
            out = dst[:, :2]
        else:
            out = dst[:, :3]
        if input_ndim == 1:
            out = np.squeeze(out)
        return out

    def inverse(self):
        """Return inverse transform.

        Returns
        -------
        inverse : object
            Inverse transform object.

        """
        return self.__class__(matrix=np.linalg.inv(self.matrix))

    def before(self, other):
        """New transform of this transform applied before another transform.

        Parameters
        ----------
        other : transform object
            Any other cartesic transform.

        Returns
        -------
        new : transform object
            New transform containing both transforms applied after each other.

        """

        if not isinstance(other, MatrixTransform):
            raise TypeError('Cannot combine transformations '
                            'of differing types.')
        if type(self) == type(other):
            tform = self.__class__
        else:
            tform = MatrixTransform
        return tform(matrix=other.matrix.dot(self.matrix))

    def after(self, other):
        """New transform of this transform applied after another transform.

        Parameters
        ----------
        other : transform object
            Any other cartesic transform.

        Returns
        -------
        new : transform object
            New transform containing both transforms applied after each other.

        """

        if not isinstance(other, MatrixTransform):
            raise TypeError('Cannot combine transformations '
                            'of differing types.')
        if type(self) == type(other):
            tform = self.__class__
        else:
            tform = MatrixTransform
        return tform(matrix=self.matrix.dot(other.matrix))


class TranslationTransform(MatrixTransform):

    def __init__(self, matrix=None, translation=0, axis=1):
        """Create scale transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scale factor.
        axis : {1, 2, 3}, optional
            Index of rotation axis (x, y, z).

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            _check_axis(axis)
            self.matrix = np.identity(4, dtype=np.double)
            self.matrix[axis - 1, 3] = translation

    @property
    def tx(self):
        """Translation in x-axis direction.

        Returns
        -------
        tx : float
            Translation.

        """

        return self.matrix[0, 3]

    @property
    def ty(self):
        """Translation in y-axis direction.

        Returns
        -------
        ty : float
            Translation.

        """

        return self.matrix[1, 3]

    @property
    def tz(self):
        """Translation in z-axis direction.

        Returns
        -------
        tz : float
            Translation.

        """

        return self.matrix[2, 3]


class ScaleTransform(MatrixTransform):

    def __init__(self, matrix=None, scale=1, axis=1):
        """Create scale transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scale factor.
        axis : {1, 2, 3}, optional
            Index of rotation axis (x, y, z).

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            _check_axis(axis)
            self.matrix = np.identity(4, dtype=np.double)
            self.matrix[axis - 1, axis - 1] *= scale

    @property
    def sx(self):
        """Scale factor in x-axis direction.

        Returns
        -------
        sx : float
            Scale factor.

        """

        return np.sqrt(np.sum(self.matrix[:3, 0]**2))

    @property
    def sy(self):
        """Scale factor in y-axis direction.

        Returns
        -------
        sy : float
            Scale factor.

        """

        return np.sqrt(np.sum(self.matrix[:3, 1]**2))

    @property
    def sz(self):
        """Scale factor in z-axis direction.

        Returns
        -------
        sz : float
            Scale factor.

        """

        return np.sqrt(np.sum(self.matrix[:3, 2]**2))



class RotationTransform(MatrixTransform):

    def __init__(self, matrix=None, angle=0, axis=1):
        """Create rotation transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        angle : float, optional
            Counter-clockwise angle in radians.
        axis : {1, 2, 3}, optional
            Index of rotation axis (x, y, z).

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            _check_axis(axis)
            if axis == 1:
                self.matrix = np.array([[1,              0,             0, 0],
                                        [0,  np.cos(angle), np.sin(angle), 0],
                                        [0, -np.sin(angle), np.cos(angle), 0],
                                        [0,              0,             0, 1]])
            elif axis == 2:
                self.matrix = np.array([[np.cos(angle), 0, -np.sin(angle), 0],
                                        [            0, 1,              0, 0],
                                        [np.sin(angle), 0,  np.cos(angle), 0],
                                        [            0, 0,              0, 1]])
            elif axis == 3:
                self.matrix = np.array([[ np.cos(angle), np.sin(angle), 0, 0],
                                        [-np.sin(angle), np.cos(angle), 0, 0],
                                        [            0,              0, 1, 0],
                                        [            0,              0, 0, 1]])

    @property
    def rx(self):
        """Rotation angle around x-axis.

        Returns
        -------
        rx : float
            Angle in radians.

        """

        a = -self.matrix[2, 1]
        b = self.matrix[2, 2]
        if isinstance(self, ScaleTransform):
            a /= abs(self.sy)
            b /= abs(self.sz)
        return np.arctan2(a, b)

    @property
    def ry(self):
        """Rotation angle around y-axis.

        Returns
        -------
        ry : float
            Angle in radians.

        """

        a = self.matrix[2, 0]
        b = self.matrix[2, 1]
        c = self.matrix[2, 2]
        if isinstance(self, ScaleTransform):
            a /= abs(self.sx)
            b /= abs(self.sy)
            c /= abs(self.sz)
        return np.arctan2(a, np.sqrt(b**2 + c**2))

    @property
    def rz(self):
        """Rotation angle around z-axis.

        Returns
        -------
        rz : float
            Angle in radians.

        """

        a = -self.matrix[1, 0]
        b = self.matrix[0, 0]
        if isinstance(self, ScaleTransform):
            a /= abs(self.sx)
            b /= abs(self.sx)
        return np.arctan2(a, b)


class ShearTransform(MatrixTransform):

    def __init__(self, matrix=None, shear=0, axis=11):
        """Create shear transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        shear : float, optional
            Shear factor.
        axis : {12, 21, 13, 31, 23, 32}, optional
            First digit is the sheared axis, second is the affected axis. E.g.
            `axis=12` shears the x coordinate and changes the y coordinate.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = np.identity(4, dtype=np.double)
            axis1, axis2 = _extract_shear_axis(axis)
            self.matrix[axis2 - 1, axis1 - 1] = shear


class EuclideanTransform(TranslationTransform, RotationTransform):

    def __init__(self, matrix=None, angle=(0, 0, 0), translation=(0, 0, 0)):
        """Create euclidean transform.

        This transform contains translation and rotation.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        angle : (3, ) array_like, optional
            Counter-clockwise angle in radians around x, y and z axis,
            respectively.
        translation : (3, ) array_like, optional
            Translation in x, y and z direction.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            # NOTE: this can be speeded up by combined application of scale and
            # and translation, but for readability the object-oriented approach
            # is chosen

            trans1 = TranslationTransform(translation=translation[0], axis=1)
            trans2 = TranslationTransform(translation=translation[1], axis=2)
            trans3 = TranslationTransform(translation=translation[2], axis=3)
            trans = trans1.before(trans2).before(trans3)

            rot1 = RotationTransform(angle=angle[0], axis=1)
            rot2 = RotationTransform(angle=angle[1], axis=2)
            rot3 = RotationTransform(angle=angle[2], axis=3)
            rot = rot1.before(rot2).before(rot3)

            tform = trans.after(rot)

            self.matrix = tform.matrix


class SimilarityTransform(TranslationTransform, ScaleTransform,
                          RotationTransform):

    def __init__(self, matrix=None, scale=1, angle=(0, 0, 0),
                 translation=(0, 0, 0)):
        """Create similarity transform.

        This transform contains translation, rotation and scaling.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scaling factor.
        angle : (3, ) array_like, optional
            Counter-clockwise angle in radians around x, y and z axis,
            respectively.
        translation : (3, ) array_like, optional
            Translation in x, y and z direction.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            # NOTE: this can be speeded up by combined application of scale and
            # and translation, but for readability the object-oriented approach
            # is chosen

            trans1 = TranslationTransform(translation=translation[0], axis=1)
            trans2 = TranslationTransform(translation=translation[1], axis=2)
            trans3 = TranslationTransform(translation=translation[2], axis=3)
            trans = trans1.before(trans2).before(trans3)

            rot1 = RotationTransform(angle=angle[0], axis=1)
            rot2 = RotationTransform(angle=angle[1], axis=2)
            rot3 = RotationTransform(angle=angle[2], axis=3)
            rot = rot1.before(rot2).before(rot3)

            scale1 = ScaleTransform(scale=scale, axis=1)
            scale2 = ScaleTransform(scale=scale, axis=2)
            scale3 = ScaleTransform(scale=scale, axis=3)
            scale = scale1.before(scale2).before(scale3)

            tform = trans.after(rot.after(scale))

            self.matrix = tform.matrix

    @property
    def s(self):
        """Scale factor.

        Returns
        -------
        s : float
            Scale factor.

        """

        return np.mean((self.sx, self.sy, self.sz))


class AffineTransform(TranslationTransform, ScaleTransform, RotationTransform,
                      ShearTransform):

    def __init__(self, matrix=None, scale=(1, 1, 1), angle=(0, 0, 0),
                 translation=(0, 0, 0), shear=()):
        """Create affine transform.

        This transform contains translation, rotation, scaling and shearing.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scaling factor.
        angle : (3, ) array_like, optional
            Counter-clockwise angle in radians around x, y and z axis,
            respectively.
        translation : (3, ) array_like, optional
            Translation in x, y and z direction.
        shear : (N, 2) array_like, optional
            Shear factors with each row as `(axis, shear)`. See `ShearTransform`
            for usage.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            # NOTE: this can be speeded up by combined application of scale and
            # and translation, but for readability the object-oriented approach
            # is chosen

            trans1 = TranslationTransform(translation=translation[0], axis=1)
            trans2 = TranslationTransform(translation=translation[1], axis=2)
            trans3 = TranslationTransform(translation=translation[2], axis=3)
            trans = trans1.before(trans2).before(trans3)

            shear_tform = MatrixTransform()
            for axis, shear_factor in shear:
                tf = ShearTransform(shear=shear_factor, axis=axis)
                shear_tform = shear_tform.before(tf)
            shear = shear_tform

            rot1 = RotationTransform(angle=angle[0], axis=1)
            rot2 = RotationTransform(angle=angle[1], axis=2)
            rot3 = RotationTransform(angle=angle[2], axis=3)
            rot = rot1.before(rot2).before(rot3)

            scale1 = ScaleTransform(scale=scale[0], axis=1)
            scale2 = ScaleTransform(scale=scale[1], axis=2)
            scale3 = ScaleTransform(scale=scale[2], axis=3)
            scale = scale1.before(scale2).before(scale3)

            tform = trans.after(shear.after(rot.after(scale)))

            self.matrix = tform.matrix
