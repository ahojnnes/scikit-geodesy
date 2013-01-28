import numpy as np
from .matrix import AffineTransform


class PiecewiseAffineTransform(object):

    def __init__(self, tesselation, affines):
        """Create piecewise affine transform.

        Parameters
        ----------
        tesselation : `scipy.spatial.Delaunay`
            Delaunay tesselation in 3 dimensions.
        affines : array_like of `skgeodesy.transform.AffineTransform`
            Affine transforms for each simplex.

        """

        self.tesselation = tesselation
        self.affines = affines

    def __call__(self, coords):
        """Apply transform to coordinates.

        Coordinates outside of the mesh are set to nan.

        Parameters
        ----------
        coords : (N, 2) or (N, 3) array_like
            2D or 3D coordinates. If z-component is not given it is set to 0.

        Returns
        -------
        out : (N, 2) or (N, 3) array_like
            Transformed 2D or 3D coordinates.

        """

        # empty input
        if coords.size == 0:
            return coords.copy()

        coords = np.array(coords, copy=False)
        input_ndim = coords.ndim
        coords = np.atleast_2d(coords)
        input_is_2D = coords.shape[1] == 2

        if input_is_2D:
            x, y = np.transpose(coords)
            z = np.zeros_like(x)
        else:
            x, y, z = np.transpose(coords)

        src = np.vstack((x, y, z)).T
        dst = np.empty_like(src)

        # find simplex for each coordinate
        if self.tesselation.ndim == 3:
            tesselation_coords = src
        elif self.tesselation.ndim == 2:
            tesselation_coords = src[:, 0:2]
        elif self.tesselation.ndim == 1:
            tesselation_coords = src[:, 0]
        simplex = self.tesselation.find_simplex(tesselation_coords)

        # coordinates outside of mesh
        dst[simplex == -1, :] = np.nan

        for index in range(len(self.tesselation.vertices)):
            # affine transform for triangle
            affine = self.affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            if np.any(index_mask):
                dst[index_mask, :] = affine(src[index_mask, :])

        # return as input
        if input_is_2D:
            out = dst[:, :2]
        else:
            out = dst[:, :3]
        if input_ndim == 1:
            out = np.squeeze(out)

        return out
