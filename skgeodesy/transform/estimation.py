import numpy as np
from numpy import sin, cos
from .matrix import EuclideanTransform, SimilarityTransform, AffineTransform, \
                    ProjectiveTransform
from .polynom import PolynomialTransform, _solve_for_num_coeffs


class TransformEstimator(object):

    def __init__(self, src, dst, src_weight=None, dst_weight=None):
        """Create total-least-squares transform estimator.

        Parameters
        ----------
        src : (N, 3) array
            Source coordinates.
        dst : (N, 3) array
            Destination coordinates.
        src_weight : (N, 3) array, optional
            Weights for source coordinates for each component separately.
        dst_weight : (N, 3) array, optional
            Weights for destination coordinates for each component separately.

        """

        assert src.shape[0] == dst.shape[0]

        self.src = np.array(src, copy=False, dtype=np.double)
        self.dst = np.array(dst, copy=False, dtype=np.double)

        # number of coordinate pairs
        self._cnum = src.shape[0]

        # weights for residuals
        if src_weight is None:
            src_weight = np.ones_like(self.src, dtype=np.double)
        if dst_weight is None:
            dst_weight = np.ones_like(self.dst, dtype=np.double)
        self.src_weight = np.array(src_weight, copy=False, dtype=np.double)
        self.dst_weight = np.array(dst_weight, copy=False, dtype=np.double)

        # observations
        self.obs = np.hstack([dst[:, 0], dst[:, 1], dst[:, 2],
                              src[:, 0], src[:, 1], src[:, 2]]).T

    def estimate(self, eps=1e-6, max_iter=10, verbose=True):
        """Estimate

        Parameters
        ----------
        eps : float, optional
            Convergence criteria for iteration. Stops if
            `sqrt(sum(dx**2)) < eps)`.
        max_iter : int, optional
            Maximum number of iterations.
        verbose : bool, optional
            Show status for each iteration.

        Returns
        -------
        tform : object
            Estimated transform object.

        Notes
        -----
         * the jacobian matrix is approximated by numerical central derivatives.
         * double precision floating point numbers are used throughout the
           estimation process.
         * initial paramaeters are automatically estimated for computation
           speedup and improved convergence

        """

        if verbose:
            print 'Starting estimation...'

        # weight matrix
        W = self._build_weight_matrix()
        # approximation / initial unknown variables
        x = self._build_initial_x()
        # error vector
        dx = np.empty_like(x)
        dx[:] = np.inf

        i = 0
        while i < max_iter and np.sqrt(np.sum(dx**2)) > eps:
            A = self._build_jacobian_matrix(x)
            w = self._build_error_vector(x)
            # normal equation matrix
            N = np.dot(A.T, np.dot(W, A))
            # error in taylor approximation
            dx = np.dot(np.linalg.pinv(N), np.dot(A.T, np.dot(W, w)))
            # "improve" parameters by approximation error
            x += dx

            i += 1
            if verbose:
                print '%3d. iteration: |dx|=%.10f' % (i, np.sqrt(np.sum(dx**2)))

        return self._build_transform(self._split_x(x)[0])

    def _build_initial_x(self):
        """Build initial unknowns.

        Returns
        -------
        x : (u, array)
            Vector of unknowns.

        """

        x0 = self._estimate_initial_params()
        return np.hstack([x0, self.src[:, 0], self.src[:, 1], self.src[:, 2]])

    def _build_error_vector(self, x):
        """Build error vector.

        Error vector is difference between observations and functional model
        evaluated at given point.

        Parameters
        ----------
        x : (U, ) array
            Vector of unknowns.

        Returns
        -------
        w : (U, ) array
            Error vector.

        """

        params, xyz = self._split_x(x)
        tform = self._build_transform(params)
        dst = tform(xyz)
        f = np.hstack([dst[:, 0], dst[:, 1], dst[:, 2],
                       xyz[:, 0], xyz[:, 1], xyz[:, 2]]).T

        return self.obs - f

    def _build_weight_matrix(self):
        """Build weight matrix.

        Weight matrix weights the errors (residuals).

        Returns
        -------
        W : (U, U) array
            Weight matrix.

        """

        W = np.zeros((6 * self._cnum, 6 * self._cnum), dtype=np.double)

        diag_idxs = np.diag_indices(self._cnum)

        W[:self._cnum,
          :self._cnum][diag_idxs] = self.dst_weight[:, 0]
        W[self._cnum:2 * self._cnum,
          self._cnum:2 * self._cnum][diag_idxs] = self.dst_weight[:, 1]
        W[2 * self._cnum:3 * self._cnum,
          2 * self._cnum:3 * self._cnum][diag_idxs] = self.dst_weight[:, 2]
        W[3 * self._cnum:4 * self._cnum,
          3 * self._cnum:4 * self._cnum][diag_idxs] = self.src_weight[:, 0]
        W[4 * self._cnum:5 * self._cnum,
          4 * self._cnum:5 * self._cnum][diag_idxs] = self.src_weight[:, 1]
        W[5 * self._cnum:,
          5 * self._cnum:][diag_idxs] = self.src_weight[:, 2]

        return W

    def _build_jacobian_matrix(self, x):
        """Build jacobian matrix.

        The jacobian matrix is evaluated numerically using central derivatives.
        The step size is automatically chosen as `x * sqrt(eps)` to avoid
        rounding errors caused by machine precision (double precision floating
        point numbers).

        Parameters
        ----------
        x : (U, ) array
            Vector of unknowns.

        Returns
        -------
        A : (6 * N, U) array
            Jacobian matrix.

        References
        ----------
        ..[1] Numerical Recipes in C - The Art of Scientific Computing - 2nd
              edition, William H. Press et al., Chapter 5.7,
              Cambridge University Press, 1992

        """

        _, xyz = self._split_x(x)

        eps = np.spacing(1)

        # jacobian matrix - x, y, z for source and destination coordinates
        A = np.zeros((6 * self._cnum, x.size), dtype=np.double)

        for p in range(x.size):
            step_size = max(np.sqrt(eps), abs(x[p]) * np.sqrt(eps))

            # backward function evaluation
            x0_bwd = x.copy()
            x0_bwd[p] -= step_size
            tform_bwd = self._build_transform(self._split_x(x0_bwd)[0])
            dst_bwd = tform_bwd(xyz)

            # forward function evaluation
            x0_fwd = x.copy()
            x0_fwd[p] += step_size
            tform_fwd = self._build_transform(self._split_x(x0_fwd)[0])
            dst_fwd = tform_fwd(xyz)

            # central derivative
            dst_diff = (dst_fwd - dst_bwd) / (x0_fwd[p] - x0_bwd[p])

            # fill jacobian matrix column
            A[0:self._cnum, p] = dst_diff[:, 0]
            A[self._cnum:2 * self._cnum, p] = dst_diff[:, 1]
            A[2 * self._cnum:3 * self._cnum, p] = dst_diff[:, 2]

        c3 = 3 * self._cnum
        A[c3:, self._pnum:] = np.identity(c3)

        return A

    def _split_x(self, x):
        """Split unknowns into parameters and adjusted source coordinates.

        Parameters
        ----------
        x : (U, ) array
            Vector of unknowns.

        Returns
        -------
        params : (N, ) array
            Transform parameters.
        xyz : (M, ) array
            Adjusted source coordinates.

        """

        params = x[:self._pnum]
        xx = x[self._pnum:self._pnum + self._cnum]
        yy = x[self._pnum + self._cnum:self._pnum + 2 * self._cnum]
        zz = x[self._pnum + 2 * self._cnum:]
        xyz = np.vstack([xx, yy, zz]).T
        return params, xyz


class MatrixTransformEstimator(TransformEstimator):

    def _estimate_initial_params(self):
        # use total least-squares without weighting as initial solution to
        # improve convergence and speedup the computation
        tform = self.estimate_without_weight()
        return tform.matrix.flat[self._param_idxs]

    def estimate_without_weight(self):
        xs = self.src[:, 0]
        ys = self.src[:, 1]
        zs = self.src[:, 2]
        xd = self.dst[:, 0]
        yd = self.dst[:, 1]
        zd = self.dst[:, 2]

        # params (ordered as in columns of A):
        # a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3
        A = np.zeros((3 * self._cnum, 16))

        # function X: a1, a2, a3, a4, d1, d2, d3
        i0 = 0
        i1 = self._cnum
        A[i0:i1, 0] = xs
        A[i0:i1, 1] = ys
        A[i0:i1, 2] = zs
        A[i0:i1, 3] = 1
        A[i0:i1, 12] = - xd * xs
        A[i0:i1, 13] = - xd * ys
        A[i0:i1, 14] = - xd * zs

        # function Y: b1, b2, b3, b4, d1, d2, d3
        i0 = self._cnum
        i1 = 2 * self._cnum
        A[i0:i1, 4] = xs
        A[i0:i1, 5] = ys
        A[i0:i1, 6] = zs
        A[i0:i1, 7] = 1
        A[i0:i1, 12] = - yd * xs
        A[i0:i1, 13] = - yd * ys
        A[i0:i1, 14] = - yd * zs

        # function Z: c1, c2, c3, c4, d1, d2, d3
        i0 = 2 * self._cnum
        i1 = 3 * self._cnum
        A[i0:i1, 8] = xs
        A[i0:i1, 9] = ys
        A[i0:i1, 10] = zs
        A[i0:i1, 11] = 1
        A[i0:i1, 12] = - zd * xs
        A[i0:i1, 13] = - zd * ys
        A[i0:i1, 14] = - zd * zs

        # d4 (= 1)
        A[:, 15] = - np.hstack([xd, yd, zd]).T

        # select needed parameters
        A = A[:, self._param_idxs + [15]]

        U, S, V = np.linalg.svd(A)

        if S[-2] < 1e-9:
            # search for largest non-zero eigenvalue
            for i in range(1, 16):
                if S[-i] > 1e-9:
                    break
        else: # no rank defect
            i = 1

        matrix = np.zeros((4, 4), dtype=np.double)
        matrix.flat[self._param_idxs + [15]] = V[-i] / V[-i, -1]

        return ProjectiveTransform(matrix=matrix)


class EuclideanTransformEstimator(MatrixTransformEstimator):

    """Euclidean transform estimator.

    Input
    -----

        N source and destination coordinates.

    Functional model (for each coordinate pair)
    ----------------

        X + vX = tx + r11*x + r12*y + r13*z
        Y + vY = ty + r21*x + r22*y + r23*z
        Z + vZ = tz + r31*x + r32*y + r33*z
        x + vx = x
        y + vy = y
        z + vz = z

        with R = r_ij = R3*R2*R1

    Parameters
    ----------

        tx, ty, tz      (translation)
        rx, ry, rz      (rotation)

    Redundancy
    ----------

        r = 3 * N - 6

    """

    _pnum = 6
    # use affine for initial params
    _param_idxs = range(12)

    def _estimate_initial_params(self):
        tform = self.estimate_without_weight()
        tx, ty, tz = tform.translation
        rx, ry, rz = tform.rotation
        s = np.mean(tform.scale)
        return s * tx, s * ty, s * tz, rx, ry, rz

    def _build_transform(self, params):
        tx, ty, tz, rx, ry, rz = params
        return EuclideanTransform(angle=(rx, ry, rz),
                                  translation=(tx, ty, tz))


class SimilarityTransformEstimator(MatrixTransformEstimator):

    """Similarity transform estimator.

    Input
    -----

        N source and destination coordinates.

    Functional model (for each coordinate pair)
    ----------------

        X + vX = tx + s*(r11*x + r12*y + r13*z)
        Y + vY = ty + s*(r21*x + r22*y + r23*z)
        Z + vZ = tz + s*(r31*x + r32*y + r33*z)
        x + vx = x
        y + vy = y
        z + vz = z

        with R = r_ij = R3*R2*R1

    Parameters
    ----------

        tx, ty, tz      (translation)
        rx, ry, rz      (rotation)
        s               (scale)

    Redundancy
    ----------

        r = 3 * N - 7

    """

    _pnum = 7
    # use affine for initial params
    _param_idxs = range(12)

    def _estimate_initial_params(self):
        tform = self.estimate_without_weight()
        tx, ty, tz = tform.translation
        rx, ry, rz = tform.rotation
        s = np.mean(tform.scale)
        return tx, ty, tz, rx, ry, rz, s

    def _build_transform(self, params):
        tx, ty, tz, rx, ry, rz, s = params
        return SimilarityTransform(scale=s, angle=(rx, ry, rz),
                                   translation=(tx, ty, tz))


class AffineTransformEstimator(MatrixTransformEstimator):

    """Affine transform estimator.

    Input
    -----

        N source and destination coordinates.

    Functional model (for each coordinate pair)
    ----------------

        X + vX = a1*x + a2*y + a3*z + a4
        Y + vY = b1*x + b2*y + b3*z + b4
        Z + vZ = c1*x + c2*y + c3*z + c4
        x + vx = x
        y + vy = y
        z + vz = z

    Parameters
    ----------

        a1, a2, a3, a4
        b1, b2, b3, b4
        c1, c2, c3, c4

    Redundancy
    ----------

        r = 3 * N - 12

    """

    _pnum = 12
    _param_idxs = range(12)

    def _build_transform(self, params):
        matrix = np.identity(4, dtype=np.double)
        matrix[:3, :] = params.reshape(3, 4)
        return AffineTransform(matrix=matrix)


class ProjectiveTransformEstimator(MatrixTransformEstimator):

    """Projective transform estimator.

    Input
    -----

        N source and destination coordinates.

    Functional model (for each coordinate pair)
    ----------------

        X + vX = (a1*x + a2*y + a3*z + a4) / (d1*x + d2*y + d3*z + 1)
        Y + vY = (b1*x + b2*y + b3*z + b4) / (d1*x + d2*y + d3*z + 1)
        Z + vZ = (c1*x + c2*y + c3*z + c4) / (d1*x + d2*y + d3*z + 1)
        x + vx = x
        y + vy = y
        z + vz = z

    Parameters
    ----------

        a1, a2, a3, a4
        b1, b2, b3, b4
        c1, c2, c3, c4
        d1, d2, d3[, 1]

    Redundancy
    ----------

        r = 3 * N - 15

    """

    _pnum = 15
    _param_idxs = range(15)

    def _build_transform(self, params):
        matrix = np.hstack([params, 1]).reshape(4, 4)
        return ProjectiveTransform(matrix=matrix)


class PolynomialTransformEstimator(TransformEstimator):

    """Polynomial transform estimator.

    Input
    -----

        N source and destination coordinates.

    Functional model (for each coordinate pair)
    ----------------

        X + vX = sum[j=0:order](
                    sum[i=0:j](
                        sum[k=0:i](
                            a_jik * x**(j - i - k) * y**(i - k) * z**k )))
        Y + vY = sum[j=0:order](
                    sum[i=0:j](
                        sum[k=0:i](
                            b_jik * x**(j - i - k) * y**(i - k) * z**k )))
        Z + vZ = sum[j=0:order](
                    sum[i=0:j](
                        sum[k=0:i](
                            c_jik * x**(j - i - k) * y**(i - k) * z**k )))
        x + vx = x
        y + vy = y
        z + vz = z

    Parameters
    ----------

        a_ijk
        b_ijk
        c_ijk

    Redundancy
    ----------

        r = 3 * N - U

        with U = (order + 1) * (order + 2) * (order * 3) / 2

    """

    def __init__(self, src, dst, order, src_weight=None, dst_weight=None):
        """Create total-least-squares transform estimator.

        Parameters
        ----------
        src : (N, 3) array
            Source coordinates.
        dst : (N, 3) array
            Destination coordinates.
        order : int
            Polynomial order.
        src_weight : (N, 3) array, optional
            Weights for source coordinates for each component separately.
        dst_weight : (N, 3) array, optional
            Weights for destination coordinates for each component separately.

        """

        TransformEstimator.__init__(self, src, dst, src_weight=src_weight,
                                    dst_weight=dst_weight)
        self.order = order
        self._pnum = _solve_for_num_coeffs(order)

    def _estimate_initial_params(self):
        tform = PolynomialTransform(self.order)
        return tform.coeffs.ravel()

    def _build_transform(self, params):
        coeffs = params.reshape(3, self._pnum / 3)
        return PolynomialTransform(self.order, coeffs=coeffs)
