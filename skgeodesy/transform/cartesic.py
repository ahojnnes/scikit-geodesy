import numpy as np


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
        coords = np.array(coords, copy=False)
        input_ndim = coords.ndim
        coords = np.atleast_2d(coords)

        if coords.shape[1] == 2:
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

        if input_ndim == 1:
            return np.squeeze(dst[:, :3])
        return dst[:, :3]

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
            else:
                raise ValueError('Axis must be 1, 2 or 3.')

    @property
    def rx(self):
        """Rotation angle around x-axis.

        Returns
        -------
        angle : float
            Angle in radians.

        """

        return np.arctan2(-self.matrix[2, 1], self.matrix[2, 2])

    @property
    def ry(self):
        """Rotation angle around y-axis.

        Returns
        -------
        angle : float
            Angle in radians.

        """

        return np.arctan2(self.matrix[2, 0],
                          np.sqrt(self.matrix[2, 1]**2 + self.matrix[2, 2]**2))

    @property
    def rz(self):
        """Rotation angle around z-axis.

        Returns
        -------
        angle : float
            Angle in radians.

        """

        return np.arctan2(-self.matrix[1, 0], self.matrix[0, 0])


class SimilarityTransform(RotationTransform):

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
            rot1 = RotationTransform(angle=angles[0], axis=1)
            rot2 = RotationTransform(angle=angles[1], axis=2)
            rot3 = RotationTransform(angle=angles[2], axis=3)
            rot = rot1.before(rot2).before(rot3)

            self.matrix = rot.matrix
            self.matrix[:3, :3] *= scale
            self.matrix[:3, 3] = translation

    @property
    def scale(self):
        return np.mean(np.sqrt(np.sum(self.matrix[:3, :3]**2, axis=1)))

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
