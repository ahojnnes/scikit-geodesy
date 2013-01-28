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

        coords = np.array(coords, copy=False)
        input_ndim = coords.ndim
        coords = np.atleast_2d(coords)
        input_is_2D = coords.shape[1] == 2

        if input_is_2D:
            x, y = np.transpose(coords)
            z = np.zeros_like(x)
        else:
            x, y, z = np.transpose(coords)

        src = np.vstack((x, y, z))
        dst = np.empty_like(src)

        # find simplex for each coordinate
        simplex = self.tesselation.find_simplex(src)

        # coordinates outside of mesh
        dst[simplex == -1, :] = np.nan

        for index in range(len(self.tesselation.vertices)):
            # affine transform for triangle
            affine = self.affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            dst[index_mask, :] = affine(src[index_mask, :])

        # return as input
        if input_is_2D:
            out = dst[:, :2]
        else:
            out = dst[:, :3]
        if input_ndim == 1:
            out = np.squeeze(out)

        return out
