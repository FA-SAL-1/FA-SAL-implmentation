import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

from blas_threads import configure_blas_threads

BLAS_THREAD_LIMITS = configure_blas_threads()

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gp_models import DynamicsGP, KernelConfig, SafetyGP, normal_cdf_array
from planning import (
    FARandomTrajectoryPlanner,
    FutureAwareTrajectoryPlanner,
    PlannerConfig,
    NominalMPCTrajectoryPlanner,
    SALNXTrajectoryPlanner,
    TebbeABMTrajectoryPlanner,
)

try:
    from config_rail_pressure import RAIL_PRESSURE_CONFIG
except Exception:
    RAIL_PRESSURE_CONFIG = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "High-Pressure-Fluid-System"


def _config_value(section: str, field: str, fallback):
    if RAIL_PRESSURE_CONFIG is None:
        return fallback
    config_section = getattr(RAIL_PRESSURE_CONFIG, section, None)
    return getattr(config_section, field, fallback)


def _method_config_prefix(method: str) -> str:
    normalized = str(method).lower()
    if normalized in {"sal", "safe_exploration"}:
        return "safe_exploration"
    if normalized in {"random", "random_safe", "fa_random"}:
        return "fa_random"
    if normalized == "myopic_fa_sal":
        return "myopic_fa_sal"
    if normalized == "tebbe_abm":
        return "tebbe"
    if normalized == "salnx":
        return "salnx"
    return "fa_sal"


def _method_alpha(args, method: str) -> float:
    prefix = _method_config_prefix(method)
    return float(getattr(args, f"{prefix}_alpha", getattr(args, "salnx_alpha", 0.2)))


def _method_mc_samples(args, method: str) -> int:
    prefix = _method_config_prefix(method)
    return int(getattr(args, f"{prefix}_mc_samples", getattr(args, "salnx_mc_samples", 128)))


def _method_uncertainty_criterion(args, method: str) -> str:
    prefix = _method_config_prefix(method)
    return str(getattr(args, f"{prefix}_uncertainty_criterion", "logdet"))


def _method_l_ell_quantile(args, method: str) -> float:
    prefix = _method_config_prefix(method)
    return float(getattr(args, f"{prefix}_l_ell_quantile", 0.9))


def _method_safety_margin(args, method: str, estimated_l_ell: Optional[float] = None) -> float:
    prefix = _method_config_prefix(method)
    if prefix not in {"fa_sal", "fa_random", "myopic_fa_sal"}:
        return 0.0
    fallback_l_ell = float(getattr(args, f"{prefix}_l_ell", 0.0))
    l_ell_scale = float(getattr(args, f"{prefix}_l_ell_scale", 0.0))
    if estimated_l_ell is not None and np.isfinite(estimated_l_ell) and float(estimated_l_ell) > 0.0:
        return max(0.0, float(l_ell_scale) * float(estimated_l_ell))
    return max(0.0, fallback_l_ell)


def _finite_domain_beta(round_idx: int, domain_size: int, delta: float) -> float:

    t = int(round_idx) + 1
    if int(domain_size) <= 0:
        raise ValueError("beta domain size must be positive")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("confidence delta must lie in (0, 1)")
    pi_t = (np.pi**2 * float(t) ** 2) / 6.0
    return float(2.0 * np.log(float(domain_size) * pi_t / float(delta)))


def _fa_sal_beta(args, round_idx: int, domain_size: int) -> Tuple[float, float]:
    if args.fa_sal_beta_schedule == "fixed":
        return float(args.fa_sal_beta_f), float(args.fa_sal_beta_g)
    return (
        _finite_domain_beta(round_idx, domain_size, args.delta_f),
        _finite_domain_beta(round_idx, domain_size, args.delta_g),
    )


def _format_sweep_value(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _format_label_value(value: float) -> str:
    return f"{float(value):g}"


def _safe_filename_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_").lower()


def _variant_result_path(args, variant: Dict[str, object]) -> Path:
    result_id = _safe_filename_part(str(variant["result_id"]))
    return args.save_dir / (
        f"rail_pressure_{result_id}_m{int(args.m)}_rounds{int(args.rounds)}_"
        f"trials{int(args.trials)}_seed{int(args.seed)}_results.json"
    )


def _build_method_variants(args) -> List[Dict[str, object]]:
    labels = {
        "fa_sal": "FA-SAL",
        "fa_random": "FA-SAL-Random",
        "myopic_fa_sal": "Myopic FA-SAL",
        "safe_exploration": "Safe Exploration",
        "salnx": "SAL-NX",
        "tebbe_abm": "Tebbe-ABM",
    }
    variants: List[Dict[str, object]] = []
    for method in args.methods:
        method_name = str(method)
        base_label = labels.get(method_name, method_name)
        if method_name in {"fa_sal", "fa_random", "myopic_fa_sal"}:
            prefix = _method_config_prefix(method_name)
            sweep = tuple(float(v) for v in getattr(args, f"{prefix}_l_ell_scale_sweep", ()))
            if sweep:
                for scale in sweep:
                    variants.append(
                        {
                            "label": f"{base_label} (l_ell_scale={_format_label_value(scale)})",
                            "base_method": method_name,
                            "result_id": f"{method_name}_l_ell_scale_{_format_sweep_value(scale)}",
                            "overrides": {f"{prefix}_l_ell_scale": float(scale)},
                        }
                    )
                continue
        if method_name == "salnx":
            sweep = tuple(float(v) for v in getattr(args, "salnx_alpha_sweep", ()))
            if sweep:
                for alpha in sweep:
                    variants.append(
                        {
                            "label": f"{base_label} (alpha={_format_label_value(alpha)})",
                            "base_method": method_name,
                            "result_id": f"salnx_alpha_{_format_sweep_value(alpha)}",
                            "overrides": {"salnx_alpha": float(alpha)},
                        }
                    )
                continue
        variants.append(
            {
                "label": base_label,
                "base_method": method_name,
                "result_id": method_name,
                "overrides": {},
            }
        )
    return variants


def _namespace_with_overrides(args, overrides: Dict[str, object]):
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _endpoint_key(endpoint: np.ndarray) -> Tuple[float, float]:
    endpoint = np.asarray(endpoint, dtype=float).reshape(-1)
    return (round(float(endpoint[0]), 8), round(float(endpoint[1]), 8))


def _iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    union = float(np.sum(predicted | truth))
    if union <= 0.0:
        return 1.0
    return float(np.sum(predicted & truth) / union)


@dataclass
class RailPressureModel:
    offsets: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    output_scale: float
    output_shift: float

    @classmethod
    def from_matlab_file(cls, path: Path) -> "RailPressureModel":
        text = path.read_text(encoding="utf-8", errors="ignore")
        norm_matches = re.findall(
            r"x\((\d+)\)\s*=\s*\(xIn\(\d+\)\s*-\s*([-+0-9.eE]+)\)\s*/\s*([-+0-9.eE]+)",
            text,
        )
        if len(norm_matches) != 10:
            raise ValueError(f"Could not parse 10 normalization constants from {path}.")
        offsets = np.zeros(10, dtype=float)
        scales = np.ones(10, dtype=float)
        for idx, offset, scale in norm_matches:
            offsets[int(idx) - 1] = float(offset)
            scales[int(idx) - 1] = float(scale)

        weights: List[List[float]] = []
        cos_coeffs: List[float] = []
        sin_coeffs: List[float] = []
        fterm_pattern = re.compile(r"fterm\s*=\s*(.*?);", re.DOTALL)
        y_pattern = re.compile(
            r"y=y\+\s*([-+0-9.eE]+)\s*\*\s*cos\(fterm\)\s*\+\s*([-+0-9.eE]+)\s*\*\s*sin\(fterm\)\s*;"
        )
        fterm_matches = list(fterm_pattern.finditer(text))
        y_matches = list(y_pattern.finditer(text))
        if len(fterm_matches) != len(y_matches) or not fterm_matches:
            raise ValueError(f"Could not parse Fourier terms from {path}.")
        for fmatch, ymatch in zip(fterm_matches, y_matches):
            expr = fmatch.group(1)
            coeffs = np.zeros(10, dtype=float)
            for var_idx, coeff in re.findall(r"x\((\d+)\)\*\s*([-+0-9.eE]+)", expr):
                coeffs[int(var_idx) - 1] = float(coeff)
            weights.append(coeffs.tolist())
            cos_coeffs.append(float(ymatch.group(1)))
            sin_coeffs.append(float(ymatch.group(2)))

        scale_match = re.search(r"y\s*=\s*y\s*\*\s*([-+0-9.eE]+)\s*\+\s*([-+0-9.eE]+)\s*;", text)
        if scale_match is None:
            raise ValueError(f"Could not parse output scaling from {path}.")
        return cls(
            offsets=offsets,
            scales=scales,
            weights=np.asarray(weights, dtype=float),
            cos_coeffs=np.asarray(cos_coeffs, dtype=float),
            sin_coeffs=np.asarray(sin_coeffs, dtype=float),
            output_scale=float(scale_match.group(1)),
            output_shift=float(scale_match.group(2)),
        )

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xn = (X - self.offsets.reshape(1, -1)) / self.scales.reshape(1, -1)
        fterms = Xn @ self.weights.T
        y = np.cos(fterms) @ self.cos_coeffs + np.sin(fterms) @ self.sin_coeffs
        return y * self.output_scale + self.output_shift

    def normalize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X - self.offsets.reshape(1, -1)) / self.scales.reshape(1, -1)

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        X_norm = np.asarray(X_norm, dtype=float)
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(1, -1)
        return X_norm * self.scales.reshape(1, -1) + self.offsets.reshape(1, -1)


class RailPressureEnv:
    def __init__(
        self,
        model: RailPressureModel,
        psi_max: float = 18.0,
        lambda_p: float = 10.0,
        m: int = 5,
        pressure_noise_std: float = 1.0,
        safety_noise_std: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.psi_max = float(psi_max)
        self.lambda_p = float(lambda_p)
        self.m = int(m)
        self.pressure_noise_std = float(pressure_noise_std)
        self.safety_noise_std = float(safety_noise_std)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def pressure(self, X: np.ndarray, noise: bool = False) -> np.ndarray:
        y = self.model(X)
        if noise:
            y = y + self.rng.normal(0.0, self.pressure_noise_std, size=np.asarray(y).shape)
        return np.asarray(y, dtype=float)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return self.model.normalize(X)

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        return self.model.denormalize(X_norm)

    def safety_value_from_pressure(self, pressure: np.ndarray, noise: bool = False) -> np.ndarray:
        z = 1.0 - np.exp((np.asarray(pressure, dtype=float) - self.psi_max) / self.lambda_p)
        if noise:
            z = z + self.rng.normal(0.0, self.safety_noise_std, size=np.asarray(z).shape)
        return np.asarray(z, dtype=float)

    def safety_value(self, X: np.ndarray, noise: bool = False) -> np.ndarray:
        pressure = self.pressure(X, noise=False)
        return self.safety_value_from_pressure(pressure, noise=noise)

    @staticmethod
    def initial_history() -> np.ndarray:
        row = np.array([2500, 2500, 2500, 2500, 30, 30, 30, 0.7, 0.7, 0.7], dtype=float)
        return np.repeat(row.reshape(1, -1), 3, axis=0)

    def create_ramp(self, history: np.ndarray, endpoint: Sequence[float]) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        endpoint = np.asarray(endpoint, dtype=float)
        start = history[-1, [0, 4]]
        tau = np.asarray(
            [start + (step / self.m) * (endpoint - start) for step in range(1, self.m + 1)],
            dtype=float,
        )
        extended = np.vstack(
            [
                history,
                np.column_stack(
                    [
                        tau[:, 0],
                        np.zeros((self.m, 3)),
                        tau[:, 1],
                        np.full((self.m, 5), 0.7),
                    ]
                ),
            ]
        )
        rows = np.zeros((self.m, 10), dtype=float)
        for j in range(self.m):
            rows[j, 0] = tau[j, 0]
            rows[j, 1] = extended[-1 - self.m + j, 0]
            rows[j, 2] = extended[-2 - self.m + j, 0]
            rows[j, 3] = extended[-3 - self.m + j, 0]
            rows[j, 4] = tau[j, 1]
            rows[j, 5] = extended[-1 - self.m + j, 4]
            rows[j, 6] = extended[-3 - self.m + j, 4]
            rows[j, 7:] = 0.7
        return rows

    def endpoint_grid(self, speed_points: int, actuation_points: int) -> np.ndarray:
        speeds = np.linspace(2200.0, 2800.0, int(speed_points))
        acts = np.linspace(24.0, 36.0, int(actuation_points))
        return np.asarray([(speed, act) for speed in speeds for act in acts], dtype=float)


def _iter_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def generate_initial_data(env: RailPressureEnv, n_trajectories: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = env.initial_history()
    X_parts = [history.copy()]
    for _ in range(int(n_trajectories)):
        last = history[-1]
        lower = np.array([max(2200.0, last[0] - 10.0 * env.m), max(24.0, last[4] - 5.0 * env.m)])
        upper = np.array([min(2800.0, last[0] + 10.0 * env.m), min(36.0, last[4] + 5.0 * env.m)])
        endpoint = lower + (upper - lower) * env.rng.random(2)
        ramp = env.create_ramp(history, endpoint)
        history = np.vstack([history, ramp])
        X_parts.append(ramp)
    X = np.vstack(X_parts)[3:]
    pressure = env.pressure(X, noise=True)
    safety = env.safety_value_from_pressure(pressure, noise=True)
    return X, pressure, safety, history


def build_evaluation_set(env: RailPressureEnv, n_eval: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    speeds = env.rng.uniform(2200.0, 2800.0, size=(int(n_eval), 4))
    acts = env.rng.uniform(24.0, 36.0, size=(int(n_eval), 3))
    ti = np.full((int(n_eval), 3), 0.7, dtype=float)
    X = np.column_stack([speeds, acts, ti])
    pressure = env.pressure(X, noise=False)
    safety = env.safety_value_from_pressure(pressure, noise=False)
    return X, pressure, safety


def abm_unsafe_upper_bound(
    safe_model: SafetyGP,
    ramp: np.ndarray,
    alpha: float,
    delta: float,
    sample_start: int,
    sample_stages: int,
) -> Tuple[str, float]:
    mean_g, cov_g = safe_model.gp.posterior(np.asarray(ramp, dtype=float), return_cov=True)
    mean_g = np.asarray(mean_g, dtype=float).reshape(-1)
    cov_g = np.asarray(cov_g, dtype=float)
    if not np.all(mean_g > 0.0):
        return "unsafe", 1.0

    sigma_tilde = float(np.sqrt(np.max(np.diag(cov_g) / np.maximum(mean_g**2, 1e-12))))
    if sigma_tilde <= 1e-12:
        return "safe", 0.0

    stages = max(1, int(sample_stages))
    schedule = [int(sample_start * (2 ** idx)) for idx in range(stages)]
    total_samples = max(schedule)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((total_samples, cov_g.shape[0]))
    cov_x = cov_g / np.outer(mean_g, mean_g)
    cov_x = 0.5 * (cov_x + cov_x.T)
    chol = np.linalg.cholesky(cov_x + 1e-10 * np.eye(cov_x.shape[0]))
    borell_samples = np.max(base @ chol.T, axis=1)
    normal = NormalDist()
    last_upper = 1.0

    for stage_idx, sample_count in enumerate(schedule, start=1):
        samples_now = borell_samples[:sample_count]
        p_hat = float(np.mean(samples_now > 1.0))
        half_delta_factor = max(3.0 * float(delta) / (np.pi**2 * stage_idx**2), 1e-12)
        full_delta_factor = max(6.0 * float(delta) / (np.pi**2 * stage_idx**2), 1e-12)
        c_upper = float(np.sqrt(max(0.0, 2.0 * abs(np.log(half_delta_factor)) / sample_count)))
        c_lower = float(np.sqrt(max(0.0, 2.0 * abs(np.log(full_delta_factor)) / sample_count)))
        p_mc_plus = min(1.0, p_hat + np.sqrt(float(alpha) * (1.0 - float(alpha))) * c_upper)
        p_mc_minus = max(0.0, p_hat - 0.25 * c_lower**2 - np.sqrt(float(alpha)) * c_lower)
        chi = min(1.0 - 1e-12, 1.0 - half_delta_factor)
        beta_plus = min(1.0, 0.5 + normal.inv_cdf(chi) / np.sqrt(4.0 * sample_count))
        q_plus = float(np.quantile(samples_now, beta_plus, method="linear"))
        p_borell_plus = 1.0 - normal.cdf((1.0 - q_plus) / sigma_tilde)
        safe_upper = min(p_borell_plus, p_mc_plus)
        last_upper = safe_upper
        if safe_upper <= float(alpha):
            return "safe", float(safe_upper)
        if p_mc_minus >= float(alpha):
            return "unsafe", float(p_mc_minus)
    return "undecided", float(last_upper)


def trajectory_safe_probability_with_margin(
    safe_model: SafetyGP,
    X_plan: np.ndarray,
    mc_samples: int,
    safety_margin: float,
) -> float:
    X_plan = np.asarray(X_plan, dtype=float)
    if X_plan.ndim == 1:
        X_plan = X_plan.reshape(1, -1)
    if safety_margin <= 0.0:
        return safe_model.trajectory_safe_probability(X_plan, mc_samples=mc_samples)

    mean_g, cov_g = safe_model.gp.posterior(X_plan, return_cov=True)
    dim = int(X_plan.shape[0])
    if dim == 1:
        std = float(np.sqrt(max(cov_g[0, 0], 0.0)))
        return normal_cdf(float(mean_g[0] - safety_margin) / max(std, 1e-8))

    jitter = 1e-10 * np.eye(dim)
    chol = np.linalg.cholesky(cov_g + jitter)
    base = safe_model._orthant_standard_normals(dim, mc_samples)
    samples = mean_g.reshape(1, -1) + base @ chol.T
    return float(np.mean(np.all(samples >= float(safety_margin), axis=1)))


def estimate_rail_local_lcb_lipschitz_constant(
    safe_model: SafetyGP,
    points: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    steps: np.ndarray,
    quantile: float = 0.9,
    beta_g: float = 1.0,
    max_points: int = 64,
) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.size == 0:
        return 0.0

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
        points = points[idx]

    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    steps = np.asarray(steps, dtype=float)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    local_norms: List[float] = []

    for z in points:
        z = np.asarray(z, dtype=float)
        center = float(safe_model.lcb(z, beta_g))
        axis_slopes: List[float] = []
        for axis in range(z.shape[0]):
            step = float(steps[axis])
            lower = float(lower_bounds[axis])
            upper = float(upper_bounds[axis])
            if step <= 0.0 or upper <= lower:
                axis_slopes.append(0.0)
                continue

            plus = z.copy()
            minus = z.copy()
            plus[axis] = min(float(z[axis]) + step, upper)
            minus[axis] = max(float(z[axis]) - step, lower)
            candidates = []
            if abs(float(plus[axis]) - float(z[axis])) > 1e-10:
                lcb_plus = float(safe_model.lcb(plus, beta_g))
                candidates.append(abs(lcb_plus - center) / abs(float(plus[axis]) - float(z[axis])))
            if abs(float(minus[axis]) - float(z[axis])) > 1e-10:
                lcb_minus = float(safe_model.lcb(minus, beta_g))
                candidates.append(abs(center - lcb_minus) / abs(float(z[axis]) - float(minus[axis])))
            axis_slopes.append(max(candidates) if candidates else 0.0)
        local_norms.append(float(np.linalg.norm(axis_slopes)))

    if not local_norms:
        return 0.0
    return float(np.quantile(np.asarray(local_norms, dtype=float), quantile))


def select_endpoint(
    method: str,
    dyn_model: DynamicsGP,
    safe_model: SafetyGP,
    env: RailPressureEnv,
    history: np.ndarray,
    endpoints: np.ndarray,
    planner_cfg: PlannerConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float, float, List[object]]:
    trajectory_candidates = [
        (endpoint.copy(), env.normalize(env.create_ramp(history, endpoint)))
        for endpoint in endpoints
    ]
    if method == "myopic_fa_sal":
        planner_cfg = PlannerConfig(**{**planner_cfg.__dict__, "horizon": 1})
        planner = FutureAwareTrajectoryPlanner(dyn_model, safe_model, planner_cfg)
    elif method == "fa_sal":
        planner = FutureAwareTrajectoryPlanner(dyn_model, safe_model, planner_cfg)
    elif method in {"fa_random", "random", "random_safe"}:
        planner = FARandomTrajectoryPlanner(dyn_model, safe_model, planner_cfg, rng=rng)
    elif method in {"salnx", "safe_exploration", "sal"}:
        planner = SALNXTrajectoryPlanner(dyn_model, safe_model, planner_cfg)
    elif method == "tebbe_abm":
        planner = TebbeABMTrajectoryPlanner(dyn_model, safe_model, planner_cfg)
    elif method == "nominal_mpc":
        planner = NominalMPCTrajectoryPlanner(dyn_model, safe_model, planner_cfg, env.psi_max, "upper", planner_cfg.output_target)
    else:
        planner = FutureAwareTrajectoryPlanner(dyn_model, safe_model, planner_cfg)

    endpoint, ramp, info, infos = planner.select_trajectory(trajectory_candidates)
    if endpoint is None or ramp is None or info is None:
        current = history[-1, [0, 4]]
        ramp = env.normalize(env.create_ramp(history, current))
        return current, ramp, float("-inf"), 0.0, infos
    return endpoint, ramp, float(info.acquisition), float(info.trajectory_safety_prob), infos


def evaluate_candidate_set_metrics(
    env: RailPressureEnv,
    infos: Sequence[object],
) -> Dict[str, float]:
    if not infos:
        return {
            "false_certification_rate": 0.0,
            "dead_end_free_recovery_iou": 0.0,
            "feasible_sizes": 0.0,
            "true_future_safe_count": 0.0,
        }

    feasible_set = {
        _endpoint_key(getattr(info, "endpoint"))
        for info in infos
        if bool(getattr(info, "feasible_full_horizon", False))
    }
    true_future_safe = set()
    for info in infos:
        endpoint = getattr(info, "endpoint")
        trajectory = env.denormalize(np.asarray(getattr(info, "trajectory"), dtype=float))
        pressure = env.pressure(trajectory, noise=False)
        if bool(np.all(pressure <= env.psi_max)):
            true_future_safe.add(_endpoint_key(endpoint))

    false_certified = feasible_set - true_future_safe
    union = feasible_set | true_future_safe
    return {
        "false_certification_rate": float(len(false_certified) / len(feasible_set)) if feasible_set else 0.0,
        "dead_end_free_recovery_iou": float(len(feasible_set & true_future_safe) / len(union)) if union else 1.0,
        "feasible_sizes": float(len(feasible_set)),
        "true_future_safe_count": float(len(true_future_safe)),
    }


def run_trial(args, method: str, seed: int) -> Dict[str, object]:
    model = RailPressureModel.from_matlab_file(args.source_dir / "models" / "prist_w_FIR_3step.m")
    env = RailPressureEnv(
        model=model,
        psi_max=args.psi_max,
        lambda_p=args.lambda_p,
        m=args.m,
        pressure_noise_std=args.pressure_noise_std,
        safety_noise_std=args.safety_noise_std,
        rng=np.random.default_rng(seed),
    )
    rng = np.random.default_rng(seed)
    X_train, y_train, z_train, history = generate_initial_data(env, args.n_init_trajectories)
    X_eval, y_eval, z_eval = build_evaluation_set(env, args.n_eval)
    X_train_gp = env.normalize(X_train)
    X_eval_gp = env.normalize(X_eval)
    endpoints = env.endpoint_grid(args.speed_grid_points, args.actuation_grid_points)
    beta_domain_size = (
        int(args.beta_domain_size)
        if int(args.beta_domain_size) > 0
        else int(endpoints.shape[0] * int(args.m))
    )

    dyn_model = DynamicsGP(KernelConfig(kind=args.kernel, variance=1.0, length_scale=args.length_scale), noise_std=args.pressure_noise_std)
    safe_model = SafetyGP(KernelConfig(kind=args.kernel, variance=1.0, length_scale=args.length_scale), noise_std=args.safety_noise_std)

    logs: Dict[str, List[float]] = {
        "rmse": [],
        "dynamics_rmse": [],
        "safety_violation_rate": [],
        "trajectory_failure": [],
        "dead_end_safety_violation_rate": [],
        "safe_coverage_iou": [],
        "safety_set_recovery_iou": [],
        "dead_end_free_recovery_iou": [],
        "false_certification_rate": [],
        "feasible_sizes": [],
        "selected_safety_probability": [],
        "selected_score": [],
        "executed_points": [],
        "estimated_l_ell": [],
        "effective_l_ell": [],
        "beta_f": [],
        "beta_g": [],
    }
    failures = 0.0
    dead_end_failures = 0.0
    executed = 0.0

    for round_idx in _iter_progress(range(int(args.rounds)), desc=f"{method} seed {seed}", leave=False, dynamic_ncols=True):
        dyn_model.fit(X_train_gp, y_train)
        safe_model.fit(X_train_gp, z_train)

        y_pred, _ = dyn_model.predict_batch(X_eval_gp)
        rmse = float(np.sqrt(np.mean((y_pred - y_eval) ** 2)))
        mean_z, std_z = safe_model.predict_batch(X_eval_gp)
        method_alpha = _method_alpha(args, method)
        method_mc_samples = _method_mc_samples(args, method)
        method_uncertainty_criterion = _method_uncertainty_criterion(args, method)
        if method == "fa_sal":
            beta_f, beta_g = _fa_sal_beta(args, round_idx, beta_domain_size)
        else:
            beta_f = 1.0
            beta_g = 1.0
        current = history[-1]
        current_gp = env.normalize(current)[0]
        distance_to_current = np.linalg.norm(X_train_gp - current_gp.reshape(1, -1), axis=1)
        nearest_count = min(48, X_train.shape[0])
        nearest_idx = np.argsort(distance_to_current)[:nearest_count]
        points = np.vstack([current_gp.reshape(1, -1), X_train_gp[nearest_idx]])
        lower_bounds_raw = np.array([2200.0, 2200.0, 2200.0, 2200.0, 24.0, 24.0, 24.0, 0.7, 0.7, 0.7], dtype=float)
        upper_bounds_raw = np.array([2800.0, 2800.0, 2800.0, 2800.0, 36.0, 36.0, 36.0, 0.7, 0.7, 0.7], dtype=float)
        lower_bounds = env.normalize(lower_bounds_raw)[0]
        upper_bounds = env.normalize(upper_bounds_raw)[0]
        steps = np.maximum((upper_bounds - lower_bounds) / 20.0, 1e-6)
        estimated_l_ell = estimate_rail_local_lcb_lipschitz_constant(
            safe_model=safe_model,
            points=points,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            steps=steps,
            quantile=_method_l_ell_quantile(args, method),
            beta_g=float(beta_g),
        )
        method_safety_margin = _method_safety_margin(args, method, estimated_l_ell=estimated_l_ell)
        planner_cfg = PlannerConfig(
            horizon=1 if method == "myopic_fa_sal" else int(args.m),
            beta_f=float(beta_f),
            beta_g=float(beta_g),
            Lx=1.0,
            Ly=1.0,
            Lf=float(args.fa_sal_lf_multiplier) if method == "fa_sal" else 1.0,
            L_ell=float(method_safety_margin),
            alpha_safety=float(method_alpha),
            safe_exploration_alpha=float(method_alpha),
            salnx_criterion=str(method_uncertainty_criterion),
            salnx_joint_mc_samples=int(method_mc_samples),
            tebbe_alpha=float(args.tebbe_alpha),
            tebbe_criterion=str(args.tebbe_uncertainty_criterion),
            tebbe_confidence_delta=float(args.tebbe_confidence_delta),
            tebbe_sample_start=int(args.tebbe_sample_start),
            tebbe_sample_stages=int(args.tebbe_sample_stages),
            fa_beam_width=1,
            output_target=float(args.pressure_target),
            control_r_u=float(args.control_effort_weight),
        )
        pred_safe = normal_cdf_array(mean_z / np.maximum(std_z, 1e-8)) >= 1.0 - method_alpha
        true_safe = z_eval >= 0.0
        iou = _iou(pred_safe, true_safe)

        endpoint, ramp_gp, score, prob, candidate_infos = select_endpoint(
            method=method,
            dyn_model=dyn_model,
            safe_model=safe_model,
            env=env,
            history=history,
            endpoints=endpoints,
            planner_cfg=planner_cfg,
            rng=rng,
        )
        candidate_metrics = evaluate_candidate_set_metrics(env, candidate_infos)
        ramp = env.create_ramp(history, endpoint)
        pressure = env.pressure(ramp, noise=True)
        safety = env.safety_value_from_pressure(pressure, noise=True)
        true_pressure = env.pressure(ramp, noise=False)
        true_failure = float(np.any(true_pressure > args.psi_max))
        true_dead_end_failure = float(true_pressure[0] <= args.psi_max and np.any(true_pressure > args.psi_max))

        X_train = np.vstack([X_train, ramp])
        X_train_gp = np.vstack([X_train_gp, ramp_gp])
        y_train = np.concatenate([y_train, pressure])
        z_train = np.concatenate([z_train, safety])
        history = np.vstack([history, ramp])
        failures += true_failure
        dead_end_failures += true_dead_end_failure
        executed += ramp.shape[0]

        logs["rmse"].append(rmse)
        logs["dynamics_rmse"].append(rmse)
        logs["safety_violation_rate"].append(float(failures / (round_idx + 1)))
        logs["trajectory_failure"].append(true_failure)
        logs["dead_end_safety_violation_rate"].append(float(dead_end_failures / (round_idx + 1)))
        logs["safe_coverage_iou"].append(iou)
        logs["safety_set_recovery_iou"].append(iou)
        logs["dead_end_free_recovery_iou"].append(float(candidate_metrics["dead_end_free_recovery_iou"]))
        logs["false_certification_rate"].append(float(candidate_metrics["false_certification_rate"]))
        logs["feasible_sizes"].append(float(candidate_metrics["feasible_sizes"]))
        logs["selected_safety_probability"].append(prob)
        logs["selected_score"].append(score)
        logs["executed_points"].append(executed)
        logs["estimated_l_ell"].append(float(estimated_l_ell))
        logs["effective_l_ell"].append(float(method_safety_margin))
        logs["beta_f"].append(float(beta_f))
        logs["beta_g"].append(float(beta_g))
    return {key: [float(v) for v in values] for key, values in logs.items()}


def _run_trial_task(task):
    args, method, seed = task
    return run_trial(args, method, seed)


def summarize_trials(trials: Sequence[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    keys = list(trials[0].keys())
    summary = {}
    for key in keys:
        arr = np.asarray([trial[key] for trial in trials], dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        valid = np.isfinite(arr)
        counts = np.sum(valid, axis=0)
        filled = np.where(valid, arr, 0.0)
        mean = np.divide(
            np.sum(filled, axis=0),
            counts,
            out=np.full(arr.shape[1], np.nan, dtype=float),
            where=counts > 0,
        )
        centered = np.where(valid, arr - mean.reshape(1, -1), 0.0)
        var = np.divide(
            np.sum(centered**2, axis=0),
            counts,
            out=np.full(arr.shape[1], np.nan, dtype=float),
            where=counts > 0,
        )
        summary[f"{key}_mean"] = mean.tolist()
        summary[f"{key}_std"] = np.sqrt(var).tolist()

    alias_pairs = {
        "dynamics_rmse": "rmse",
        "safety_set_recovery_iou_active": "safety_set_recovery_iou",
        "dead_end_free_recovery_iou_active": "dead_end_free_recovery_iou",
    }
    for alias, source in alias_pairs.items():
        for suffix in ("mean", "std"):
            source_key = f"{source}_{suffix}"
            if source_key in summary:
                summary[f"{alias}_{suffix}"] = list(summary[source_key])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rail-pressure high-pressure-fluid safe active-learning experiment.")
    parser.add_argument("--source-dir", type=Path, default=_config_value("data", "source_dir", DEFAULT_SOURCE_DIR))
    parser.add_argument("--methods", nargs="+", default=list(_config_value("general", "methods", ("salnx", "fa_random", "tebbe_abm"))))
    parser.add_argument("--rounds", type=int, default=_config_value("general", "rounds", 250))
    parser.add_argument("--trials", type=int, default=_config_value("general", "trials", 5))
    parser.add_argument("--parallel-workers", type=int, default=_config_value("general", "parallel_workers", 1), help="Number of worker processes for trial-level parallelism. Use 0 to auto-select, 1 for serial execution.")
    parser.add_argument("--m", type=int, default=_config_value("environment", "m", 5), help="Ramp discretization points per trajectory.")
    parser.add_argument("--n-init-trajectories", type=int, default=_config_value("environment", "n_init_trajectories", 25))
    parser.add_argument("--n-eval", type=int, default=_config_value("environment", "n_eval", 1000))
    parser.add_argument("--speed-grid-points", type=int, default=_config_value("environment", "speed_grid_points", 21))
    parser.add_argument("--actuation-grid-points", type=int, default=_config_value("environment", "actuation_grid_points", 21))
    parser.add_argument("--psi-max", type=float, default=_config_value("environment", "psi_max", 18.0))
    parser.add_argument("--lambda-p", type=float, default=_config_value("environment", "lambda_p", 10.0))
    parser.add_argument("--delta-f", type=float, default=_config_value("fa_sal", "delta_f", 0.05))
    parser.add_argument("--delta-g", type=float, default=_config_value("fa_sal", "delta_g", 0.05))
    parser.add_argument("--beta-domain-size", type=int, default=_config_value("fa_sal", "beta_domain_size", 0), help="Finite certificate domain size |D|. Use 0 for endpoint_count * m.")
    parser.add_argument("--fa-sal-beta-schedule", choices=["fixed", "finite_domain_time_uniform"], default="fixed")
    parser.add_argument("--fa-sal-beta-f", type=float, default=1.0)
    parser.add_argument("--fa-sal-beta-g", type=float, default=1.0)
    parser.add_argument("--fa-sal-l-ell", type=float, default=_config_value("fa_sal", "l_ell", 1.0))
    parser.add_argument("--fa-sal-l-ell-quantile", type=float, default=_config_value("fa_sal", "l_ell_quantile", 0.9))
    parser.add_argument("--fa-sal-l-ell-scale", type=float, default=_config_value("fa_sal", "l_ell_scale", 1.0))
    parser.add_argument("--fa-sal-l-ell-scale-sweep", type=float, nargs="*", default=list(_config_value("fa_sal", "l_ell_scale_sweep", ())))
    parser.add_argument("--fa-sal-lf-multiplier", type=float, default=1.0, help="Multiplier L_f / L_f^nom for FA-SAL; the existing nominal rail-pressure value is L_f^nom=1.")
    parser.add_argument("--fa-sal-uncertainty-criterion", choices=["logdet", "trace", "maxeig"], default=_config_value("fa_sal", "uncertainty_criterion", "logdet"))
    parser.add_argument("--fa-random-l-ell", type=float, default=_config_value("fa_random", "l_ell", 1.0))
    parser.add_argument("--fa-random-l-ell-quantile", type=float, default=_config_value("fa_random", "l_ell_quantile", 0.9))
    parser.add_argument("--fa-random-l-ell-scale", type=float, default=_config_value("fa_random", "l_ell_scale", 1.0))
    parser.add_argument("--fa-random-l-ell-scale-sweep", type=float, nargs="*", default=list(_config_value("fa_random", "l_ell_scale_sweep", ())))
    parser.add_argument("--myopic-fa-sal-l-ell", type=float, default=_config_value("myopic_fa_sal", "l_ell", 1.0))
    parser.add_argument("--myopic-fa-sal-l-ell-quantile", type=float, default=_config_value("myopic_fa_sal", "l_ell_quantile", 0.9))
    parser.add_argument("--myopic-fa-sal-l-ell-scale", type=float, default=_config_value("myopic_fa_sal", "l_ell_scale", 1.0))
    parser.add_argument("--myopic-fa-sal-l-ell-scale-sweep", type=float, nargs="*", default=list(_config_value("myopic_fa_sal", "l_ell_scale_sweep", ())))
    parser.add_argument("--safe-exploration-alpha", type=float, default=_config_value("safe_exploration", "alpha", 0.2))
    parser.add_argument("--salnx-alpha", type=float, default=_config_value("salnx", "alpha", 0.2))
    parser.add_argument("--salnx-alpha-sweep", type=float, nargs="*", default=list(_config_value("salnx", "alpha_sweep", ())))
    parser.add_argument("--salnx-mc-samples", type=int, default=_config_value("salnx", "mc_samples", 128))
    parser.add_argument("--salnx-uncertainty-criterion", choices=["logdet", "trace", "maxeig"], default=_config_value("salnx", "uncertainty_criterion", "logdet"))
    parser.add_argument("--tebbe-alpha", type=float, default=_config_value("tebbe", "alpha", 0.2))
    parser.add_argument("--tebbe-uncertainty-criterion", choices=["logdet", "trace", "maxeig"], default=_config_value("tebbe", "uncertainty_criterion", "logdet"))
    parser.add_argument("--tebbe-confidence-delta", type=float, default=_config_value("tebbe", "confidence_delta", 1e-2))
    parser.add_argument("--tebbe-sample-start", type=int, default=_config_value("tebbe", "sample_start", 32))
    parser.add_argument("--tebbe-sample-stages", type=int, default=_config_value("tebbe", "sample_stages", 6))
    parser.add_argument("--kernel", choices=["se", "matern52"], default=_config_value("learning", "kernel", "se"))
    parser.add_argument("--length-scale", type=float, default=_config_value("learning", "length_scale", 1.0))
    parser.add_argument("--pressure-target", type=float, default=13.5)
    parser.add_argument("--control-effort-weight", type=float, default=1e-6)
    parser.add_argument("--pressure-noise-std", type=float, default=_config_value("environment", "pressure_noise_std", 1.0))
    parser.add_argument("--safety-noise-std", type=float, default=_config_value("environment", "safety_noise_std", 0.01))
    parser.add_argument("--seed", type=int, default=_config_value("general", "seed", 0))
    parser.add_argument("--save-dir", type=Path, default=_config_value("general", "save_dir", Path("result_rail_pressure")))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.perf_counter()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    all_trials = {}
    variants = _build_method_variants(args)
    total_trial_count = int(args.trials) * max(1, len(variants))
    use_parallel = int(args.parallel_workers) != 1 and total_trial_count > 1
    worker_count = 1
    if use_parallel:
        requested_workers = int(args.parallel_workers)
        worker_count = max(1, requested_workers) if requested_workers > 0 else min(os.cpu_count() or 1, total_trial_count)
        use_parallel = worker_count > 1
    if use_parallel:
        print(f"Parallel trial mode: using {worker_count} worker processes.")
        print(f"BLAS thread limits per worker: {BLAS_THREAD_LIMITS}")
    variant_iter = _iter_progress(variants, desc="Rail-pressure variants", dynamic_ncols=True)
    for variant in variant_iter:
        label = str(variant["label"])
        method = str(variant["base_method"])
        variant_args = _namespace_with_overrides(args, dict(variant.get("overrides", {})))
        if use_parallel:
            tasks = [(variant_args, method, variant_args.seed + idx) for idx in range(int(variant_args.trials))]
            trials = [None] * len(tasks)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_idx = {executor.submit(_run_trial_task, task): idx for idx, task in enumerate(tasks)}
                for future in as_completed(future_to_idx):
                    trials[future_to_idx[future]] = future.result()
        else:
            trials = [run_trial(variant_args, method, variant_args.seed + idx) for idx in range(int(variant_args.trials))]
        summary = summarize_trials(trials)
        all_trials[label] = trials
        all_summaries[label] = summary
        print(
            f"{label}: final RMSE={summary['rmse_mean'][-1]:.3f} MPa | "
            f"SVR={summary['safety_violation_rate_mean'][-1]:.3f} | "
            f"safe IoU={summary['safe_coverage_iou_mean'][-1]:.3f}"
        )
    payload = {
        "config": {
            "source_dir": str(args.source_dir),
            "normalized_inputs": True,
            "input_transform": "matlab_offsets_scales",
            "metric_schema": "synthetic_v1",
            "methods": list(args.methods),
            "variant_labels": [str(variant["label"]) for variant in variants],
            "variants": [
                {
                    "label": str(variant["label"]),
                    "base_method": str(variant["base_method"]),
                    "result_id": str(variant["result_id"]),
                    "overrides": dict(variant.get("overrides", {})),
                }
                for variant in variants
            ],
            "rounds": args.rounds,
            "trials": args.trials,
            "parallel_workers": worker_count if use_parallel else 1,
            "m": args.m,
            "n_init_trajectories": args.n_init_trajectories,
            "psi_max": args.psi_max,
            "lambda_p": args.lambda_p,
            "beta_schedule": args.fa_sal_beta_schedule,
            "beta_formula": (
                f"beta_f={args.fa_sal_beta_f:g}, beta_g={args.fa_sal_beta_g:g}"
                if args.fa_sal_beta_schedule == "fixed"
                else "2*log(|D|*(pi^2*t^2/6)/delta_q)"
            ),
            "beta_f_fixed": args.fa_sal_beta_f,
            "beta_g_fixed": args.fa_sal_beta_g,
            "beta_domain_size": (
                args.beta_domain_size
                if args.beta_domain_size > 0
                else args.speed_grid_points * args.actuation_grid_points * args.m
            ),
            "delta_f": args.delta_f,
            "delta_g": args.delta_g,
            "fa_sal_l_ell": args.fa_sal_l_ell,
            "fa_sal_l_ell_quantile": args.fa_sal_l_ell_quantile,
            "fa_sal_l_ell_scale": args.fa_sal_l_ell_scale,
            "fa_sal_l_ell_scale_sweep": list(args.fa_sal_l_ell_scale_sweep),
            "fa_sal_lf_nominal": 1.0,
            "fa_sal_lf_multiplier": args.fa_sal_lf_multiplier,
            "fa_sal_lf_effective": args.fa_sal_lf_multiplier,
            "fa_sal_uncertainty_criterion": args.fa_sal_uncertainty_criterion,
            "fa_random_l_ell": args.fa_random_l_ell,
            "fa_random_l_ell_quantile": args.fa_random_l_ell_quantile,
            "fa_random_l_ell_scale": args.fa_random_l_ell_scale,
            "fa_random_l_ell_scale_sweep": list(args.fa_random_l_ell_scale_sweep),
            "myopic_fa_sal_l_ell": args.myopic_fa_sal_l_ell,
            "myopic_fa_sal_l_ell_quantile": args.myopic_fa_sal_l_ell_quantile,
            "myopic_fa_sal_l_ell_scale": args.myopic_fa_sal_l_ell_scale,
            "myopic_fa_sal_l_ell_scale_sweep": list(args.myopic_fa_sal_l_ell_scale_sweep),
            "safe_exploration_alpha": args.safe_exploration_alpha,
            "salnx_alpha": args.salnx_alpha,
            "salnx_alpha_sweep": list(args.salnx_alpha_sweep),
            "salnx_mc_samples": args.salnx_mc_samples,
            "salnx_uncertainty_criterion": args.salnx_uncertainty_criterion,
            "tebbe_alpha": args.tebbe_alpha,
            "tebbe_uncertainty_criterion": args.tebbe_uncertainty_criterion,
            "tebbe_confidence_delta": args.tebbe_confidence_delta,
            "tebbe_sample_start": args.tebbe_sample_start,
            "tebbe_sample_stages": args.tebbe_sample_stages,
            "kernel": args.kernel,
            "length_scale": args.length_scale,
            "pressure_noise_std": args.pressure_noise_std,
            "safety_noise_std": args.safety_noise_std,
            "seed": args.seed,
        },
        "summaries": all_summaries,
        "trials": all_trials,
        "elapsed_seconds": time.perf_counter() - start,
    }
    result_path = args.save_dir / "rail_pressure_results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_path}")
    for variant in variants:
        label = str(variant["label"])
        variant_payload = {
            **payload,
            "config": {
                **payload["config"],
                "methods": [str(variant["base_method"])],
                "variant_labels": [label],
                "variants": [
                    {
                        "label": label,
                        "base_method": str(variant["base_method"]),
                        "result_id": str(variant["result_id"]),
                        "overrides": dict(variant.get("overrides", {})),
                    }
                ],
            },
            "summaries": {label: all_summaries[label]},
            "trials": {label: all_trials[label]},
        }
        variant_path = _variant_result_path(args, variant)
        variant_path.write_text(json.dumps(variant_payload, indent=2), encoding="utf-8")
        print(f"Saved {variant_path}")


if __name__ == "__main__":
    main()
