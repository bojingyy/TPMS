"""Cubic Hermite spline curve construction (paper Eq. 2, Fig. 6)."""
import numpy as np


def hermite_coeffs(p1, d1, p2, d2):
    """Solve P(x) = C0 + C1 x + C2 x^2 + C3 x^3 for
    P(0)=p1, P'(0)=d1, P(1)=p2, P'(1)=d2  (paper Eq. 2).
    p1,d1,p2,d2 are arrays of shape (..., D); returns C0..C3 of the same shape.
    """
    p1 = np.asarray(p1, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    C0 = p1
    C1 = d1
    C2 = 3 * (p2 - p1) - 2 * d1 - d2
    C3 = 2 * (p1 - p2) + d1 + d2
    return C0, C1, C2, C3


def hermite_eval(p1, d1, p2, d2, t):
    """Evaluate the Hermite curve at parameter(s) t in [0,1].
    t: array of shape (N,). Returns array of shape (N, D).
    """
    C0, C1, C2, C3 = hermite_coeffs(p1, d1, p2, d2)
    t = np.asarray(t, dtype=float)[:, None]
    return C0[None, :] + C1[None, :] * t + C2[None, :] * t**2 + C3[None, :] * t**3
