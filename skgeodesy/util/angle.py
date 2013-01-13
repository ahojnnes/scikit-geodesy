import numpy as np


def wrap_to_pi(angles):
    """Wrap angles to interval [-pi, pi].

    Parameters
    ----------
    angles : array
        Angles in radians.

    Returns
    -------
    wrapped : array
        Wrapped angles.

    """

    wrapped = np.atleast_1d(np.array(angles, copy=True))

    mask = np.logical_or(wrapped < -np.pi, np.pi < wrapped)
    wrapped[mask] = wrap_to_2pi(wrapped[mask] + np.pi) - np.pi

    return wrapped


def wrap_to_2pi(angles):
    """Wrap angles to interval [0, 2pi].

    Parameters
    ----------
    angles : array
        Angles in radians.

    Returns
    -------
    wrapped : array
        Wrapped angles.

    """

    wrapped = np.atleast_1d(np.array(angles, copy=False))

    mask = wrapped > 0
    np.mod(wrapped, 2 * np.pi, out=wrapped)
    wrapped[np.logical_and(wrapped == 0, mask)] = 2 * np.pi

    return wrapped


def deg2dms(x, out=None):
    """Convert angles in degrees to degrees, minutes and seconds.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and it
        must be of the right shape to hold the output.

    Returns
    -------
    out : (..., 3) ndarray
        Angles as `(deg, min, sec)` as last dimension.

    Examples
    --------
    >>> np.deg2dms(48.5)
    array([ 48., 30., 0.])
    >>> np.deg2dms(-48.5)
    array([-48., -30., -0.])
    >>> np.dms2deg(np.deg2dms(-48.5))
    -48.5

    """

    x = np.asarray(x)

    if out is None:
        out = np.atleast_2d(np.empty(x.shape + (3, ), dtype=np.double))

    # conversion
    xabs = np.abs(x)
    out[..., 0] = np.floor(xabs)
    xabs = (xabs - out[..., 0]) * 60
    out[..., 1] = np.floor(xabs)
    out[..., 2] = (xabs - out[..., 1]) * 60

    # corrections caused by limited machine precision
    eps = np.spacing(1)

    s = out[..., 2]
    smask = ((np.round(s) - s) / 3600) < eps
    out[..., 2][smask] = np.round(out[..., 2][smask])

    mmask = s == -60
    out[..., 1][mmask] = out[..., 1][mmask] - 1
    mmask = s == 60
    out[..., 1][mmask] = out[..., 1][mmask] + 1

    out[..., 2][s == -60] = 0
    out[..., 2][s == 60] = 0

    # apply correct sign
    sign = np.sign(x)
    for i in range(3):
        out[..., i] *= sign

    if x.ndim == 0:
        return out[0]

    return out


def dms2deg(x, out=None):
    """Convert angles in degrees, minutes and seconds to degrees.

    Parameters
    ----------
    x : (..., 3) array_like
        Input array as `(deg, min, sec)` as last dimension.
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and it
        must be of the right shape to hold the output.

    Returns
    -------
    out : ndarray
        Angles in degrees.

    Examples
    --------
    >>> np.dms2deg([48, 30, 0])
    48.5
    >>> np.dms2deg([-48, -30, 0])
    -48.5
    >>> np.deg2dms(np.dms2deg([-48, -30, 0]))
    array([-48., -30., -0.])

    """

    x = np.asarray(x)

    if out is None:
        out = np.atleast_1d(np.empty(x.shape[:-1], dtype=np.double))

    out[:] = x[..., 0] + x[..., 1] / 60. + x[..., 2] / 3600.

    if x.ndim == 1:
        return out[0]

    return out
