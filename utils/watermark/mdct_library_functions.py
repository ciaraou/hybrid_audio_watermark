import numpy as np

""" COPIED FROM mdct LIBRARY SECTION START """

def mdct(x, odd=True):
    """ Calculate modified discrete cosine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return trans(x, func=np.cos, odd=odd) * np.sqrt(2)


def imdct(X, odd=True):
    """ Calculate inverse modified discrete cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return itrans(X, func=np.cos, odd=odd) * np.sqrt(2)

def trans(x, func, odd=True):
    """ Calculate modified discrete sine/cosine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    func : callable
        The transform kernel function
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(x) // 2
    if odd:
        outlen = N
        offset = 0.5
    else:
        outlen = N + 1
        offset = 0.0

    X = np.zeros(outlen, dtype=np.complex)
    n = np.arange(len(x))

    for k in range(len(X)):
        X[k] = np.sum(
            x * func(
                (np.pi / N) * (
                    n + 0.5 + N / 2
                ) * (
                    k + offset
                )
            )
        )

    if not odd:
        X[0] *= np.sqrt(0.5)
        X[-1] *= np.sqrt(0.5)

    return X * np.sqrt(1 / N)


def itrans(X, func, odd=True):
    """ Calculate inverse modified discrete sine/cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    func : callable
        The transform kernel function
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    if not odd and len(X) % 2 == 0:
        raise ValueError(
            "Even inverse CMDCT requires an odd number "
            "of coefficients"
        )

    X = X.copy()

    if odd:
        N = len(X)
        offset = 0.5
    else:
        N = len(X) - 1
        offset = 0.0

        X[0] *= np.sqrt(0.5)
        X[-1] *= np.sqrt(0.5)

    x = np.zeros(N * 2, dtype=np.complex)
    k = np.arange(len(X))

    for n in range(len(x)):
        x[n] = np.sum(
            X * func(
                (np.pi / N) * (
                    n + 0.5 + N / 2
                ) * (
                    k + offset
                )
            )
        )

    return np.real(x) * np.sqrt(1 / N)

""" COPIED FROM mdct LIBRARY SECTION END """