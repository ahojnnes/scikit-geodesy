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
