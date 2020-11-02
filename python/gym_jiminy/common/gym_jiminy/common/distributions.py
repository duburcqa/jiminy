""" TODO: Write documentation.
"""
import numpy as np


# For references about Gaussian processes:
#     https://peterroelants.github.io/posts/gaussian-process-tutorial/
#     https://peterroelants.github.io/posts/gaussian-process-kernels/
#     https://stats.stackexchange.com/questions/238655/sampling-from-matrix-variate-normal-distribution-with-singular-covariances  # noqa: E501  # pylint: disable=line-too-long

class PeriodicGaussianProcess:
    """ TODO: Write documentation.
    """
    def __init__(self,
                 mean: np.ndarray,
                 scale: np.ndarray,
                 wavelength: np.ndarray,
                 period: np.ndarray,
                 dt: np.ndarray) -> None:
        """ TODO: Write documentation.
        """
        assert isinstance(mean, np.ndarray) and \
            np.issubdtype(mean.dtype, np.floating), (
                "'mean' must be a real-valued numpy array.")
        assert np.all(scale > 0.0), "'scale' must be strictly positive."
        assert np.all(wavelength > 0.0), (
            "'wavelength' must be strictly positive.")
        assert np.all(period > 0.0), "'period' must be strictly positive."
        assert np.all(dt > 0.0), "'dt' must be strictly positive."

        # Backup some user argument(s)
        self.mean = mean

        # Compute the covariance matrix and associated SVD decomposition.
        # Note that SVD is used instead of Cholesky. Even though it is
        # computionally more expensive, it works with non strictly positive
        # definite matrices contrary to Cholesky.
        scale = scale.reshape((scale.shape[0], 1, 1))
        wavelength = wavelength.reshape((wavelength.shape[0], 1, 1))
        period = period.reshape((period.shape[0], 1, 1))

        t = np.outer(dt.reshape((-1)), np.arange(mean.shape[-1]))
        t_dist_mat = t.reshape((t.shape[0], -1, 1)) - \
            t.reshape((t.shape[0], 1, -1))

        cov = scale ** 2 * np.exp(
            -2.0 / wavelength ** 2 *
            np.sin(np.pi * np.abs(t_dist_mat) / period) ** 2)

        _, s, v = np.linalg.svd(cov)  # u = v.T because cov is symmetric
        cov_sqrt = np.sqrt(s)[..., None] * v  # = np.diag(np.sqrt(s)) @ v in 2D

        self.cov = cov
        self._cov_sqrt = cov_sqrt

    def sample(self) -> np.ndarray:
        """ TODO: Write documentation.
        """
        return (np.random.standard_normal(self.mean.shape) @
                self._cov_sqrt)[..., 0, :] + self.mean
