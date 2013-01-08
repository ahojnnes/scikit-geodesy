import numpy as np


def _solve_for_order(num_coeffs):
    """Solve equation for polynomial order.

    Equation (u = total number of coefficients, n = polynomial order):

        u = ((n + 1) * (n + 2) * (n + 3)) / 2

    The equation is solved using Cardano's formula for equations of cubic
    degree.

    """

    # substituted coefficients
    p = -1
    q = -2 * num_coeffs

    # D always > 0
    D = (q / 2)**2 + (p / 3)**3

    u = (-q / 2 + np.sqrt(D))**(1 / 3.)
    v = (-q / 2 - np.sqrt(D))**(1 / 3.)

    # substituted solution
    z = u + v

    # re-substitute
    order = z - 2

    return int(order)


def _solve_for_coeffs(order):
    """Solve equation for number of coefficients.

    """

    return ((order + 1) * (order + 2) * (order + 3)) / 2


class PolynomialTransform(object):

    def __init__(self, order, coeffs=None):
        """Create polynomial transform.

        Parameters
        ----------
        coeffs : (3, N) array, optional
            Polynomial coefficients for x, y and z, respectively. By default
            the coefficients are chosen, so that the transformed coordinates
            equal the original coordinates.

        """

        if coeffs is None:
            # default to transformation which preserves original coordinates
            coeffs = np.zeros((3, _solve_for_coeffs(order) / 3))
            coeffs[0, 1] = coeffs[1, 2] = coeffs[2, 3] = 1
        if coeffs.shape[0] != 3:
            raise ValueError('Invalid shape of polynomial coefficients')
        assert coeffs.size == _solve_for_coeffs(order)
        self.order = order
        self.coeffs = coeffs

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

        dst = np.zeros((x.shape[0], 3))

        order = max(1, _solve_for_order(self.coeffs.size))

        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                for k in range(i + 1):
                    a = x**(j - i - k) * y**(i - k) * z**k
                    dst[:, 0] += self.coeffs[0, pidx] * a
                    dst[:, 1] += self.coeffs[1, pidx] * a
                    dst[:, 2] += self.coeffs[2, pidx] * a
                    pidx += 1

        if input_is_2D:
            out = dst[:, :2]
        else:
            out = dst[:, :3]
        if input_ndim == 1:
            out = np.squeeze(out)

        return out

    def inverse(self):
        raise Exception(
            'There is no explicit way to do the inverse polynomial '
            'transformation. Instead, estimate the inverse transformation '
            'parameters by exchanging source and destination coordinates,'
            'then apply the forward transformation.')

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

        if not isinstance(other, PolynomialTransform):
            raise TypeError('Cannot combine transformations '
                            'of differing types.')
        return PolynomialTransform(order=self.order,
                                   coeffs=self.coeffs * other.coeffs)

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

        return self.before(other)
