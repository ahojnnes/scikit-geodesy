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
            raise TypeError('Cannot combine transformations of differing types.')
        if type(self) == type(other):
            tform = self.__class__
        else:
            tform = CartesicTransform
        return tform(other.matrix.dot(self.matrix))

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
            raise TypeError('Cannot combine transformations of differing types.')
        if type(self) == type(other):
            tform = self.__class__
        else:
            tform = CartesicTransform
        return tform(self.matrix.dot(other.matrix))


class RotationTransform(CartesicTransform):

    def __init__(self, angle=None, axis=None):
        """Create rotation transform.

        Parameters
        ----------
        angle : float, optional
            Counter-clockwise angle in radians.
        axis : {1, 2, 3}, optional
            Index of rotation axis.

        """

        if angle is None and axis is None:
            self.matrix = np.identity(4, dtype=np.double)
        elif (angle is not None and axis is None) \
             or (angle is None and axis is not None):
            raise ValueError('You must specify both angle and axis.')

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
    def angle1(self):
        """Rotation angle aroung axis 1."""

        return np.arctan2(-self.matrix[2, 1], self.matrix[2, 2])

    @property
    def angle2(self):
        """Rotation angle aroung axis 2."""

        return np.arctan2(self.matrix[2, 0],
                          np.sqrt(self.matrix[2, 1]**2 + self.matrix[2, 2]**2))

    @property
    def angle3(self):
        """Rotation angle aroung axis 3."""

        return np.arctan2(-self.matrix[1, 0], self.matrix[0, 0])
