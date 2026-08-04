from dataclasses import dataclass
from math import erf, sqrt
from typing import Tuple

import numpy as np
from scipy.linalg import solve_triangular


@dataclass
class KernelConfig:
    kind: str = "se"
    variance: float = 1.0
    length_scale: float = 1.0


def _sqeuclidean(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    x1_sq = np.sum(X1**2, axis=1, keepdims=True)
    x2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    return np.maximum(x1_sq + x2_sq - 2.0 * X1 @ X2.T, 0.0)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(float(value) / sqrt(2.0)))


def normal_cdf_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(values / sqrt(2.0)))


def kernel_matrix(X1: np.ndarray, X2: np.ndarray, cfg: KernelConfig) -> np.ndarray:
    scaled = np.sqrt(_sqeuclidean(X1 / cfg.length_scale, X2 / cfg.length_scale))
    if cfg.kind == "se":
        return cfg.variance * np.exp(-0.5 * scaled**2)
    if cfg.kind == "matern52":
        root5 = np.sqrt(5.0) * scaled
        return cfg.variance * (1.0 + root5 + 5.0 * scaled**2 / 3.0) * np.exp(-root5)
    raise ValueError(f"Unsupported kernel kind: {cfg.kind}")


def kernel_diag(X: np.ndarray, cfg: KernelConfig) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.full(X.shape[0], float(cfg.variance), dtype=float)


def kernel_gradient_x2(X1: np.ndarray, X2: np.ndarray, cfg: KernelConfig) -> np.ndarray:

    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    differences = X1[:, None, :] - X2[None, :, :]
    inv_length_sq = 1.0 / float(cfg.length_scale) ** 2
    scaled = np.sqrt(_sqeuclidean(X1 / cfg.length_scale, X2 / cfg.length_scale))
    if cfg.kind == "se":
        kernel = cfg.variance * np.exp(-0.5 * scaled**2)
        return kernel[:, :, None] * differences * inv_length_sq
    if cfg.kind == "matern52":
        root5 = np.sqrt(5.0) * scaled
        coefficient = cfg.variance * (5.0 / 3.0) * (1.0 + root5) * np.exp(-root5)
        return coefficient[:, :, None] * differences * inv_length_sq
    raise ValueError(f"Unsupported kernel kind: {cfg.kind}")


class GaussianProcess1D:
    def __init__(self, kernel_cfg: KernelConfig, noise_var: float, jitter: float = 1e-8):
        self.kernel_cfg = kernel_cfg
        self.noise_var = float(noise_var)
        self.jitter = float(jitter)
        self.X_train = np.empty((0, 1), dtype=float)
        self.y_train = np.empty((0,), dtype=float)
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.L = None
        self.alpha = None
        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.X_train = X
        self.y_train = y
        self.y_mean = float(np.mean(y))
        self.y_scale = float(np.std(y))
        if self.y_scale < 1e-8:
            self.y_scale = 1.0

        y_norm = (y - self.y_mean) / self.y_scale
        K = kernel_matrix(X, X, self.kernel_cfg)
        diagonal = K.diagonal().copy()
        np.fill_diagonal(K, diagonal + self.noise_var + self.jitter)

        self.L = np.linalg.cholesky(K)
        tmp = solve_triangular(self.L, y_norm, lower=True, check_finite=False)
        self.alpha = solve_triangular(
            self.L.T,
            tmp,
            lower=False,
            check_finite=False,
        )
        self.is_fit = True

    def posterior(self, X_star: np.ndarray, return_cov: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fit:
            raise RuntimeError("GP must be fit before calling posterior().")

        X_star = np.asarray(X_star, dtype=float)
        if X_star.ndim == 1:
            X_star = X_star.reshape(1, -1)

        K_trans = kernel_matrix(self.X_train, X_star, self.kernel_cfg)
        mean_norm = K_trans.T @ self.alpha
        v = solve_triangular(
            self.L,
            K_trans,
            lower=True,
            check_finite=False,
        )

        mean = self.y_mean + self.y_scale * mean_norm
        if return_cov:
            K_star = kernel_matrix(X_star, X_star, self.kernel_cfg)
            cov_norm = K_star - v.T @ v
            cov_norm = 0.5 * (cov_norm + cov_norm.T)
            np.fill_diagonal(cov_norm, np.maximum(np.diag(cov_norm), 0.0))
            cov = (self.y_scale**2) * cov_norm
            return mean, cov

        var_norm = kernel_diag(X_star, self.kernel_cfg) - np.sum(v**2, axis=0)
        var = (self.y_scale**2) * np.clip(var_norm, 0.0, None)
        return mean, var

    def mean_std(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, var = self.posterior(X_star, return_cov=False)
        return mean, np.sqrt(np.clip(var, 0.0, None))

    def mean_jacobian(self, X_star: np.ndarray) -> np.ndarray:

        if not self.is_fit:
            raise RuntimeError("GP must be fit before calling mean_jacobian().")
        X_star = np.asarray(X_star, dtype=float)
        if X_star.ndim == 1:
            X_star = X_star.reshape(1, -1)
        gradients = kernel_gradient_x2(self.X_train, X_star, self.kernel_cfg)
        return self.y_scale * np.einsum("n,nqd->qd", self.alpha, gradients)

    def posterior_blocks(self, X_blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if not self.is_fit:
            raise RuntimeError("GP must be fit before calling posterior_blocks().")
        X_blocks = np.asarray(X_blocks, dtype=float)
        if X_blocks.ndim != 3:
            raise ValueError("X_blocks must have shape (batch, block_size, features).")
        batch_size, block_size, feature_count = X_blocks.shape
        X_flat = X_blocks.reshape(batch_size * block_size, feature_count)
        K_trans = kernel_matrix(self.X_train, X_flat, self.kernel_cfg)
        mean_norm = (K_trans.T @ self.alpha).reshape(batch_size, block_size)
        v = solve_triangular(
            self.L,
            K_trans,
            lower=True,
            check_finite=False,
        ).reshape(self.X_train.shape[0], batch_size, block_size)
        scaled = X_blocks / self.kernel_cfg.length_scale
        squared = np.sum(scaled**2, axis=2)
        distances = np.sqrt(np.maximum(squared[:, :, None] + squared[:, None, :] - 2.0 * np.einsum("bid,bjd->bij", scaled, scaled), 0.0))
        if self.kernel_cfg.kind == "se":
            K_blocks = self.kernel_cfg.variance * np.exp(-0.5 * distances**2)
        elif self.kernel_cfg.kind == "matern52":
            root5 = np.sqrt(5.0) * distances
            K_blocks = self.kernel_cfg.variance * (1.0 + root5 + 5.0 * distances**2 / 3.0) * np.exp(-root5)
        else:
            raise ValueError(f"Unsupported kernel kind: {self.kernel_cfg.kind}")
        cov_norm = K_blocks - np.einsum("nbi,nbj->bij", v, v)
        cov_norm = 0.5 * (cov_norm + np.swapaxes(cov_norm, 1, 2))
        diag_idx = np.arange(block_size)
        cov_norm[:, diag_idx, diag_idx] = np.maximum(
            cov_norm[:, diag_idx, diag_idx],
            0.0,
        )
        return self.y_mean + self.y_scale * mean_norm, (self.y_scale**2) * cov_norm


class DynamicsGP:
    def __init__(self, kernel_cfg: KernelConfig, noise_std: float):
        self.gp = GaussianProcess1D(kernel_cfg=kernel_cfg, noise_var=noise_std**2)

    def fit(self, X: np.ndarray, y_next: np.ndarray) -> None:
        self.gp.fit(X, y_next.reshape(-1))

    def predict(self, z: np.ndarray) -> Tuple[float, float]:
        mean, std = self.gp.mean_std(np.asarray(z, dtype=float).reshape(1, -1))
        return float(mean[0]), float(std[0])

    def predict_batch(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        return self.gp.mean_std(Z)

    def mean_jacobian_batch(self, Z: np.ndarray) -> np.ndarray:
        return self.gp.mean_jacobian(Z)

    def rollout_information_gain(self, X_plan: np.ndarray) -> float:
        _, cov = self.gp.posterior(X_plan, return_cov=True)
        eye = np.eye(cov.shape[0])
        matrix = eye + cov / self.gp.noise_var
        sign, logdet = np.linalg.slogdet(matrix + 1e-10 * eye)
        if sign <= 0:
            return float("-inf")
        return 0.5 * float(logdet)

    def rollout_information_gain_batch(self, X_blocks: np.ndarray) -> np.ndarray:

        _, covariances = self.gp.posterior_blocks(X_blocks)
        block_size = covariances.shape[1]
        eye = np.eye(block_size)
        matrices = eye[None, :, :] + covariances / self.gp.noise_var
        signs, logdets = np.linalg.slogdet(matrices + 1e-10 * eye[None, :, :])
        return np.where(signs > 0, 0.5 * logdets, float("-inf"))

    def rollout_uncertainty_score(self, X_plan: np.ndarray, criterion: str) -> float:
        _, cov = self.gp.posterior(X_plan, return_cov=True)
        cov = 0.5 * (cov + cov.T)
        eye = np.eye(cov.shape[0])
        if criterion == "logdet":
            sign, logdet = np.linalg.slogdet(cov + 1e-10 * eye)
            return float(logdet) if sign > 0 else float("-inf")
        if criterion == "trace":
            return float(np.trace(cov))
        if criterion == "maxeig":
            return float(np.max(np.linalg.eigvalsh(cov)))
        raise ValueError(f"Unsupported SAL-NX criterion: {criterion}")

    def time_aware_imspe_score(self, x_query: np.ndarray, X_eval: np.ndarray) -> float:
        x_query = np.asarray(x_query, dtype=float).reshape(1, -1)
        X_eval = np.asarray(X_eval, dtype=float)
        X_joint = np.vstack([X_eval, x_query])
        _, cov = self.gp.posterior(X_joint, return_cov=True)
        m = X_eval.shape[0]
        cross_cov = cov[:m, m]
        query_var = float(max(cov[m, m], 0.0))
        denom = query_var + self.gp.noise_var
        if denom <= 1e-12:
            return 0.0
        return float(np.mean((cross_cov**2) / denom))


class SafetyGP:
    def __init__(self, kernel_cfg: KernelConfig, noise_std: float = 1e-4):
        self.gp = GaussianProcess1D(kernel_cfg=kernel_cfg, noise_var=noise_std**2)
        self._orthant_base_samples = {}

    def fit(self, X: np.ndarray, g_values: np.ndarray) -> None:
        self.gp.fit(X, g_values.reshape(-1))

    def predict(self, z: np.ndarray) -> Tuple[float, float]:
        mean, std = self.gp.mean_std(np.asarray(z, dtype=float).reshape(1, -1))
        return float(mean[0]), float(std[0])

    def predict_batch(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        return self.gp.mean_std(Z)

    def lcb(self, z: np.ndarray, beta_g: float) -> float:
        mean, std = self.predict(z)
        return mean - np.sqrt(beta_g) * std

    def lcb_batch(self, Z: np.ndarray, beta_g: float) -> np.ndarray:
        mean, std = self.predict_batch(Z)
        return mean - np.sqrt(beta_g) * std

    def _orthant_standard_normals(self, dim: int, mc_samples: int) -> np.ndarray:
        key = (int(dim), int(mc_samples))
        cached = self._orthant_base_samples.get(key)
        if cached is None:
            rng = np.random.default_rng(0)
            cached = rng.standard_normal((mc_samples, dim))
            self._orthant_base_samples[key] = cached
        return cached

    def trajectory_safe_probability(self, X_plan: np.ndarray, mc_samples: int = 128) -> float:
        X_plan = np.asarray(X_plan, dtype=float)
        if X_plan.ndim == 1:
            X_plan = X_plan.reshape(1, -1)
        mean_g, cov_g = self.gp.posterior(X_plan, return_cov=True)
        dim = int(X_plan.shape[0])
        if dim == 1:
            std = float(np.sqrt(max(cov_g[0, 0], 0.0)))
            return normal_cdf(float(mean_g[0]) / max(std, 1e-8))

        jitter = 1e-10 * np.eye(dim)
        chol = np.linalg.cholesky(cov_g + jitter)
        base = self._orthant_standard_normals(dim, mc_samples)
        samples = mean_g.reshape(1, -1) + base @ chol.T
        return float(np.mean(np.all(samples >= 0.0, axis=1)))
