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


def _remove_scale_and_shear(tform):
    if hasattr(tform, 'remove_translation'):
        tform = tform.remove_translation()
    else:
        tform = tform

    removed = tform.matrix.copy()

    # scale factor on x-axis
    scx = np.sqrt(np.sum(removed[:3, 0]**2))

    # normalize column 1
    removed[:, 0] /= scx

    # compute xy shear factor and make column 2 orthogonal to column 1
    shxy  = removed[:, 0].dot(removed[:, 1])
    removed[:, 1] -= shxy * removed[:, 0]

    # scale factor on y-axis
    scy = np.sqrt(np.sum(removed[:3, 1]**2))

    # normalize column 2
    removed[:, 1] /= scy

    # compute xz shear factor and make column 3 orthogonal to column 1
    shxz = removed[:, 0].dot(removed[:, 2])
    removed[:, 2] -= shxz * removed[:, 0]
    # compute yz shear factor and make column 3 orthogonal to column 2
    shyz  = removed[:, 1].dot(removed[:, 2])
    removed[:, 2] -= shyz * removed[:, 1]

    # scale factor on z-axis
    scz = np.sqrt(np.sum(removed[:3, 2]**2))

    # normalize column 3 and xz, yz shears
    removed[:, 2] /= scz
    shxz /= scz
    shyz /= scz

    # negate matrix if determinant is smaller than 0 since we face a coordinate
    # system flip in this case
    if np.linalg.det(removed) < 0:
        removed *= -1
        scx *= -1
        scy *= -1
        scz *= -1

    return (scx, scy, scz), (shxy, shxz, shyz), tform.__class__(matrix=removed)


class MatrixTransform(object):

    def __init__(self, matrix=None):
        """Create matrix transform.

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

        # return as input
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

    def __init__(self, matrix=None, translation=None, axis=None):
        """Create scale transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        translation : (N, ) array_like, optional
            Scale factor.
        axis : (N, ) array_like of {1, 2, 3}, optional
            Index of rotation axis (x, y, z).

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = np.identity(4, dtype=np.double)
            if axis is not None:
                if not hasattr(axis, '__iter__'):
                    axis = (axis, )
                    translation = (translation, )
                for ax, tr in zip(axis, translation):
                    _check_axis(ax)
                    self.matrix[ax - 1, 3] = tr

    def remove_translation(self):
        """Return new transform with translation removed.

        Returns
        -------
        removed : object
            Transform object with translation removed.

        """

        # perspective has to be removed before translation
        if hasattr(self, 'remove_perspective'):
            tform = self.remove_perspective()
        else:
            tform = self

        removed = tform.matrix.copy()
        removed[:3, 3] = 0
        return self.__class__(matrix=removed)

    @property
    def translation(self):
        """Translation in x, y and z-axis direction, respectively.

        Returns
        -------
        tx : float
            Translation in x-axis direction.
        ty : float
            Translation in y-axis direction.
        tz : float
            Translation in z-axis direction.

        """

        return self.matrix[:3, 3]


class ScaleTransform(MatrixTransform):

    def __init__(self, matrix=None, scale=None, axis=None):
        """Create scale transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : (N, ) array_like, optional
            Scale factor.
        axis : (N, ) array_like of {1, 2, 3}, optional
            Index of rotation axis (x, y, z).

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = np.identity(4, dtype=np.double)
            if axis is not None:
                if not hasattr(axis, '__iter__'):
                    axis = (axis, )
                    scale = (scale, )
                for ax, sc in zip(axis, scale):
                    _check_axis(ax)
                    self.matrix[ax - 1, ax - 1] *= sc

    def remove_scale(self):
        """Return new transform with scale removed.

        This also removes the shear from the matrix since this is not separable
        from removing the scale.

        Returns
        -------
        removed : object
            Transform object with scale and shear removed.

        """

        return _remove_scale_and_shear(self)[-1]

    @property
    def scale(self):
        """Scale factors on x, y and z-axis.

        Returns
        -------
        scx : float
            Scale factor on x-axis.
        scy : float
            Scale factor on y-axis.
        scz : float
            Scale factor on z-axis.

        """

        return _remove_scale_and_shear(self)[0]


class RotationTransform(MatrixTransform):

    def __init__(self, matrix=None, angle=None, axis=None):
        """Create rotation transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        angle : (N, ) array_like, optional
            Counter-clockwise angle in radians.
        axis : (N, ) array_like of {1, 2, 3}, optional
            Index of rotation axis (x, y, z).

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = np.identity(4, dtype=np.double)
            if axis is not None:
                if not hasattr(axis, '__iter__'):
                    axis = (axis, )
                    angle = (angle, )
                for ax, an in zip(axis, angle):
                    _check_axis(ax)
                    if ax == 1:
                        rot = np.array([[1,           0,          0, 0],
                                        [0,  np.cos(an), np.sin(an), 0],
                                        [0, -np.sin(an), np.cos(an), 0],
                                        [0,           0,          0, 1]])
                        self.matrix = rot.dot(self.matrix)
                    elif ax == 2:
                        rot = np.array([[np.cos(an), 0, -np.sin(an), 0],
                                        [         0, 1,           0, 0],
                                        [np.sin(an), 0,  np.cos(an), 0],
                                        [         0, 0,           0, 1]])
                        self.matrix = rot.dot(self.matrix)
                    elif ax == 3:
                        rot = np.array([[ np.cos(an), np.sin(an), 0, 0],
                                        [-np.sin(an), np.cos(an), 0, 0],
                                        [          0,          0, 1, 0],
                                        [          0,          0, 0, 1]])
                        self.matrix = rot.dot(self.matrix)

    @property
    def rotation(self):
        if hasattr(self, 'remove_scale'):
            tform = self.remove_scale()
        elif hasattr(self, 'remove_shear'):
            tform = self.remove_shear()
        else:
            tform = self

        rx = np.arctan2(-tform.matrix[2, 1], tform.matrix[2, 2])
        ry = np.arctan2(tform.matrix[2, 0],
                        np.sqrt(tform.matrix[2, 1]**2 + tform.matrix[2, 2]**2))
        rz = np.arctan2(-tform.matrix[1, 0], tform.matrix[0, 0])

        return (rx, ry, rz)


class ShearTransform(MatrixTransform):

    def __init__(self, matrix=None, shear=None, axis=None):
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
            if axis is not None:
                axis1, axis2 = _extract_shear_axis(axis)
                self.matrix[axis2 - 1, axis1 - 1] = shear

    def remove_shear(self):
        """Return new transform with shear removed.

        This also removes the scale from the matrix since this is not separable
        from removing the shear.

        Returns
        -------
        removed : object
            Transform object with shear and scale removed.

        """

        return _remove_scale_and_shear(self)[-1]

    @property
    def shear(self):
        """Shear factors.

        Returns
        -------
        shxy : float
            Shear on x and y affected by shear.
        shxz : float
            Shear on x and z affected by shear.
        shyz : float
            Shear on y and z affected by shear.

        """

        return _remove_scale_and_shear(self)[1]


class PerspectiveTransform(MatrixTransform):

    def __init__(self, matrix=None, perspective=None, axis=None):
        """Create perspective transform.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        perspective : (N, ) array_like, optional
            Perspective factor.
        axis : (N, ) array_like of {1, 2, 3}, optional
            Index of perspective axis.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = np.identity(4, dtype=np.double)
            if axis is not None:
                if not hasattr(axis, '__iter__'):
                    axis = (axis, )
                    perspective = (perspective, )
                for ax, pe in zip(axis, perspective):
                    _check_axis(ax)
                    self.matrix[3, ax - 1] = pe


    def remove_perspective(self):
        """Return new transform with perspective removed.

        Returns
        -------
        removed : object
            Transform object with perspective removed.

        """

        included = np.identity(4, dtype=np.double)
        included[3, :] = self.perspective
        removed = np.linalg.solve(included, self.matrix)
        return self.__class__(matrix=removed)

    @property
    def perspective(self):
        """Perspective factor on x-axis.

        Returns
        -------
        px : float
            Perspective factor on x-axis.
        py : float
            Perspective factor on y-axis.
        pz : float
            Perspective factor on z-axis.
        pw : float
            Perspective factor on 4th coordinate component.

        """

        if np.all(self.matrix[3] == (0, 0, 0, 1)):
            return 0, 0, 0, 1
        removed = self.matrix.copy()
        removed[3] = (0, 0, 0, 1)
        return np.linalg.solve(removed.T, self.matrix[3])


class EuclideanTransform(TranslationTransform, RotationTransform):

    def __init__(self, matrix=None, angle=(0, 0, 0), translation=(0, 0, 0)):
        """Create euclidean transform.

        This transform contains translation and rotation.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        angle : (3, ) array_like, optional
            Counter-clockwise angles in radians around x, y and z axis,
            respectively.
        translation : (3, ) array_like, optional
            Translations in x, y and z direction.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            # NOTE: this can be speeded up by combined application of scale and
            # and translation, but for readability the object-oriented approach
            # is chosen

            trans = TranslationTransform(translation=translation,
                                         axis=(1, 2, 3))
            rot = RotationTransform(angle=angle, axis=(1, 2, 3))

            tform = trans.after(rot)

            self.matrix = tform.matrix


class SimilarityTransform(TranslationTransform, RotationTransform,
                          ScaleTransform):

    def __init__(self, matrix=None, scale=1, angle=(0, 0, 0),
                 translation=(0, 0, 0)):
        """Create similarity transform.

        This transform contains translation, rotation and scaling.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scaling factors.
        angle : (3, ) array_like, optional
            Counter-clockwise angles in radians around x, y and z axis,
            respectively.
        translation : (3, ) array_like, optional
            Translations in x, y and z direction.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            trans = TranslationTransform(translation=translation,
                                         axis=(1, 2, 3))
            rot = RotationTransform(angle=angle, axis=(1, 2, 3))
            scale = ScaleTransform(scale=(scale, scale, scale), axis=(1, 2, 3))

            tform = trans.after(rot.after(scale))

            self.matrix = tform.matrix

    @property
    def scale(self):
        """Scale factor.

        Returns
        -------
        scale : float
            Scale factor.

        """

        return np.mean(super(SimilarityTransform, self).scale)


class AffineTransform(TranslationTransform, RotationTransform, ScaleTransform,
                      ShearTransform):

    def __init__(self, matrix=None, scale=(1, 1, 1), angle=(0, 0, 0),
                 translation=(0, 0, 0), shear=()):
        """Create affine transform.

        This transform contains translation, rotation, scaling and shear.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scaling factors.
        angle : (3, ) array_like, optional
            Counter-clockwise angles in radians around x, y and z axis,
            respectively.
        translation : (3, ) array_like, optional
            Translations in x, y and z direction.
        shear : (N, 2) array_like, optional
            Shear factors with each row as `(axis, shear)`. See `ShearTransform`
            for usage.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            trans = TranslationTransform(translation=translation,
                                         axis=(1, 2, 3))
            rot = RotationTransform(angle=angle, axis=(1, 2, 3))
            scale = ScaleTransform(scale=scale, axis=(1, 2, 3))

            shear_tform = MatrixTransform()
            for axis, shear_factor in shear:
                tf = ShearTransform(shear=shear_factor, axis=axis)
                shear_tform = shear_tform.before(tf)
            shear = shear_tform

            tform = trans.after(shear.after(rot.after(scale)))

            self.matrix = tform.matrix


class ProjectiveTransform(TranslationTransform, RotationTransform,
                          ScaleTransform, ShearTransform, PerspectiveTransform):

    def __init__(self, matrix=None, scale=(1, 1, 1), angle=(0, 0, 0),
                 translation=(0, 0, 0), shear=(), perspective=(0, 0, 0)):
        """Create projective transform.

        This transform contains translation, rotation, scaling, shear
        and perspective.

        Parameters
        ----------
        matrix : (4, 4) array, optional
            Homogeneous transform matrix.
        scale : float, optional
            Scaling factors.
        angle : (3, ) array_like, optional
            Counter-clockwise angles in radians around x, y and z axis,
            respectively.
        translation : (3, ) array_like, optional
            Translations in x, y and z direction.
        shear : (N, 2) array_like, optional
            Shear factors with each row as `(axis, shear)`. See `ShearTransform`
            for usage.
        perspective : (3, ) array_like, optional
            Perspective factors.

        """

        if matrix is not None:
            self.matrix = matrix
        else:
            affine = AffineTransform(matrix=matrix, scale=scale, angle=angle,
                                     translation=translation, shear=shear)

            perspective = PerspectiveTransform(perspective=perspective,
                                               axis=(1, 2, 3))

            tform = perspective.after(affine)

            self.matrix = tform.matrix
