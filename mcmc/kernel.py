"""kernel functions for the MCMC algorithm."""

from numba import jit
import numpy as np


@jit(nopython=True)
def get_squared_diffences(xs: np.ndarray) -> np.ndarray:
    """Get the squared of the differences between all pairs of vectors in `xs`.

    Args:
        xs (np.ndarray): The input array of shape `(n, d)`.

    Returns:
        np.ndarray: The n x n matrix of the squared difference of all vectors in `xs`.
    """
    return (np.expand_dims(xs, 1) - np.expand_dims(xs, 0)) ** 2


@jit(nopython=True)
def get_se_kernel(
    X: np.ndarray,
    l1: float,
    l2: float,
    jitter: float = 1.0e-6,
) -> np.ndarray:
    """Get the squared exponential kernel between two vectors `x1` and `x2`.

    Args:
        X (np.ndarray): The squared difference matrix.
        l1 (fload): The square root of the output variance parameter of the kernel.
        l2 (float): The lengthscale parameter.

    Returns:
        np.ndarray: The squared exponential kernel between `x1` and `x2`.
    """
    return l1**2 * np.exp(-0.5 * (X) / (l2**2)) + jitter * np.eye(X.shape[0])


@jit(nopython=True)
def get_matrix_a(
    M: np.ndarray,
    ind_un: np.array,
    max_size_get_eig: int = 4,
) -> np.ndarray:
    """Perform the Cholesky factorization of the augmented kernel `lambda * I - Q`.

    Args:
        M (np.ndarray): The input matrix to perform Cholesky factorization on the unobserved variables.
        ind_un (np.array): The indices of the unobserved variables.
        max_size_get_eig (int): The maximum size of the matrix to obtain and use eigenvalues.
            If the matrix is larger, the trace is used instead to ensure positive-definiteness.

    Returns:
        np.ndarray: The Cholesky factor of `lambda * I - Q`.
    """
    Q = M[ind_un, :][:, ind_un]
    if Q.shape[0] < max_size_get_eig:
        eigvs = np.real(np.linalg.eigvals(Q.astype(np.complex128)))
        D = (1.05 * np.max(eigvs)) * np.eye(Q.shape[0])
    else:
        D = np.trace(Q) * np.eye(Q.shape[0])

    return np.linalg.cholesky(D - Q).T
