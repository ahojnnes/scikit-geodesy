import numpy as np


def _check_axis(axis):
    if axis not in (1, 2, 3):
        raise ValueError('Axis must be 1, 2 or 3.')


class CartesicTransform(object):

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

        if not isinstance(other, CartesicTransform):
            raise TypeError('Cannot combine transformations '
                            'of differing types.')
        if type(self) == type(other):
            tform = self.__class__
        else:
            tform = CartesicTransform
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

        if not isinstance(other, CartesicTransform):
            raise TypeError('Cannot combine transformations '
                            'of differing types.')
        if type(self) == type(other):
            tform = self.__class__
        else:
            tform = CartesicTransform
        return tform(matrix=self.matrix.dot(other.matrix))


class TranslationTransform(CartesicTransform):

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


class ScaleTransform(CartesicTransform):

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



class RotationTransform(CartesicTransform):

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

        return np.arctan2(-self.matrix[2, 1], self.matrix[2, 2])

    @property
    def ry(self):
        """Rotation angle around y-axis.

        Returns
        -------
        ry : float
            Angle in radians.

        """

        return np.arctan2(self.matrix[2, 0],
                          np.sqrt(self.matrix[2, 1]**2 + self.matrix[2, 2]**2))

    @property
    def rz(self):
        """Rotation angle around z-axis.

        Returns
        -------
        rz : float
            Angle in radians.

        """

        return np.arctan2(-self.matrix[1, 0], self.matrix[0, 0])


class SimilarityTransform(TranslationTransform, ScaleTransform,
                          RotationTransform):

    def __init__(self, matrix=None, scale=1, angles=(0, 0, 0),
                 translation=(0, 0, 0)):

        """Create similarity transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scaling factor.
        angles : (3, ) array_like, optional
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

            scale1 = ScaleTransform(scale=scale, axis=1)
            scale2 = ScaleTransform(scale=scale, axis=2)
            scale3 = ScaleTransform(scale=scale, axis=3)
            scale = scale1.before(scale2).before(scale3)

            rot1 = RotationTransform(angle=angles[0], axis=1)
            rot2 = RotationTransform(angle=angles[1], axis=2)
            rot3 = RotationTransform(angle=angles[2], axis=3)
            rot = rot1.before(rot2).before(rot3)

            tform = trans.after(scale.after(rot))

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
