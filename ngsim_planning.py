from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gp_models import DynamicsGP, SafetyGP

@dataclass
class PlannerConfig:
    horizon: int = 2
    beta_f: float = 1.0
    beta_g: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    Lf: float = 0.6
    L_ell: float = 0.25
    alpha_safety: float = 0.5
    safe_exploration_alpha: float = 0.05
    salnx_criterion: str = "logdet"
    salnx_joint_mc_samples: int = 64
    tebbe_alpha: float = 0.2
    tebbe_criterion: str = "trace"
    tebbe_confidence_delta: float = 1e-3
    tebbe_sample_start: int = 32
    tebbe_sample_stages: int = 6
    tebbe_endpoint_candidates: int = 81
    fa_safety_buffer_weight: float = 0.0
    fa_beam_width: int = 5


@dataclass
class TrajectoryCandidateInfo:
    endpoint: np.ndarray
    trajectory: np.ndarray
    feasible_full_horizon: bool
    acquisition: float
    pointwise_lcb: float
    trajectory_safety_prob: float
    min_safety_slack: float
    unsafe_upper_bound: float = 1.0


class FutureAwareTrajectoryPlanner:
    def __init__(self, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        self.dyn_model = dyn_model
        self.safe_model = safe_model
        self.cfg = cfg

    def evaluate_trajectory(self, endpoint: np.ndarray, trajectory: np.ndarray) -> TrajectoryCandidateInfo:
        X_plan = np.asarray(trajectory, dtype=float)
        lcbs = np.asarray(self.safe_model.lcb_batch(X_plan, self.cfg.beta_g), dtype=float)
        _, dyn_stds = self.dyn_model.predict_batch(X_plan)
        dyn_stds = np.asarray(dyn_stds, dtype=float)

        A = float(self.cfg.Lx + self.cfg.Ly * self.cfg.Lf)
        e_j = 0.0
        slacks: List[float] = []
        feasible = True
        for idx, lcb in enumerate(lcbs[: int(self.cfg.horizon)]):
            required = 0.0 if idx == 0 else float(self.cfg.L_ell) * e_j
            slack = float(lcb - required)
            slacks.append(slack)
            if slack < 0.0:
                feasible = False
                break
            e_j = A * e_j + float(self.cfg.Ly) * np.sqrt(float(self.cfg.beta_f)) * float(dyn_stds[idx])

        if X_plan.shape[0] < int(self.cfg.horizon):
            feasible = False

        acquisition = (
            self.dyn_model.rollout_information_gain(X_plan[: int(self.cfg.horizon)])
            if feasible
            else float("-inf")
        )
        return TrajectoryCandidateInfo(
            endpoint=np.asarray(endpoint, dtype=float).copy(),
            trajectory=X_plan.copy(),
            feasible_full_horizon=bool(feasible),
            acquisition=float(acquisition),
            pointwise_lcb=float(lcbs[0]) if lcbs.size else float("-inf"),
            trajectory_safety_prob=1.0 if feasible else 0.0,
            min_safety_slack=float(min(slacks)) if slacks else float("-inf"),
        )

    def select_trajectory(
        self,
        candidates: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TrajectoryCandidateInfo], List[TrajectoryCandidateInfo]]:
        infos = [self.evaluate_trajectory(endpoint, trajectory) for endpoint, trajectory in candidates]
        feasible = [info for info in infos if info.feasible_full_horizon]
        if not feasible:
            return None, None, None, infos
        best = max(feasible, key=lambda info: float(info.acquisition))
        return best.endpoint.copy(), best.trajectory.copy(), best, infos


class FARandomTrajectoryPlanner(FutureAwareTrajectoryPlanner):
    def __init__(
        self,
        dyn_model: DynamicsGP,
        safe_model: SafetyGP,
        cfg: PlannerConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(dyn_model, safe_model, cfg)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def select_trajectory(
        self,
        candidates: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TrajectoryCandidateInfo], List[TrajectoryCandidateInfo]]:
        infos = [self.evaluate_trajectory(endpoint, trajectory) for endpoint, trajectory in candidates]
        feasible = [info for info in infos if info.feasible_full_horizon]
        if not feasible:
            return None, None, None, infos
        best = feasible[int(self.rng.integers(len(feasible)))]
        return best.endpoint.copy(), best.trajectory.copy(), best, infos


class SALNXTrajectoryPlanner:
    def __init__(self, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        self.dyn_model = dyn_model
        self.safe_model = safe_model
        self.cfg = cfg

    def evaluate_trajectory(self, endpoint: np.ndarray, trajectory: np.ndarray) -> TrajectoryCandidateInfo:
        X_plan = np.asarray(trajectory, dtype=float)[: int(self.cfg.horizon)]
        prob = self.safe_model.trajectory_safe_probability(
            X_plan,
            mc_samples=self.cfg.salnx_joint_mc_samples,
        )
        feasible = float(prob) >= 1.0 - float(self.cfg.alpha_safety)
        acquisition = (
            self.dyn_model.rollout_uncertainty_score(X_plan, criterion=self.cfg.salnx_criterion)
            if feasible
            else float("-inf")
        )
        lcb = float(self.safe_model.lcb(X_plan[0], self.cfg.beta_g)) if X_plan.size else float("-inf")
        return TrajectoryCandidateInfo(
            endpoint=np.asarray(endpoint, dtype=float).copy(),
            trajectory=np.asarray(trajectory, dtype=float).copy(),
            feasible_full_horizon=bool(feasible and X_plan.shape[0] >= int(self.cfg.horizon)),
            acquisition=float(acquisition),
            pointwise_lcb=lcb,
            trajectory_safety_prob=float(prob),
            min_safety_slack=lcb,
            unsafe_upper_bound=float(1.0 - prob),
        )

    def select_trajectory(
        self,
        candidates: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TrajectoryCandidateInfo], List[TrajectoryCandidateInfo]]:
        infos = [self.evaluate_trajectory(endpoint, trajectory) for endpoint, trajectory in candidates]
        feasible = [info for info in infos if info.feasible_full_horizon]
        if not feasible:
            return None, None, None, infos
        best = max(feasible, key=lambda info: float(info.acquisition))
        return best.endpoint.copy(), best.trajectory.copy(), best, infos


class TebbeABMTrajectoryPlanner(SALNXTrajectoryPlanner):
    def __init__(self, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        super().__init__(dyn_model, safe_model, cfg)
        self._normal = NormalDist()
        self._sample_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _sample_schedule(self) -> List[int]:
        return [int(self.cfg.tebbe_sample_start * (2 ** r)) for r in range(max(1, int(self.cfg.tebbe_sample_stages)))]

    def _standard_normals(self, dim: int, total_samples: int) -> np.ndarray:
        key = (int(dim), int(total_samples))
        cached = self._sample_cache.get(key)
        if cached is None:
            rng = np.random.default_rng(0)
            cached = rng.standard_normal((total_samples, dim))
            self._sample_cache[key] = cached
        return cached

    def _abm_unsafe_upper_bound(self, mean_g: np.ndarray, cov_g: np.ndarray) -> Tuple[str, float]:
        mean_g = np.asarray(mean_g, dtype=float).reshape(-1)
        cov_g = np.asarray(cov_g, dtype=float)
        if not np.all(mean_g > 0.0):
            return "unsafe", 1.0

        schedule = self._sample_schedule()
        max_samples = schedule[-1]
        alpha = float(self.cfg.tebbe_alpha)
        delta = float(self.cfg.tebbe_confidence_delta)
        sigma_tilde = float(np.sqrt(np.max(np.diag(cov_g) / np.maximum(mean_g**2, 1e-12))))
        if sigma_tilde <= 1e-12:
            return "safe", 0.0

        cov_x = cov_g / np.outer(mean_g, mean_g)
        cov_x = 0.5 * (cov_x + cov_x.T)
        chol = np.linalg.cholesky(cov_x + 1e-10 * np.eye(cov_x.shape[0]))
        base = self._standard_normals(cov_x.shape[0], max_samples)
        borell_samples = np.max(base @ chol.T, axis=1)
        last_upper = 1.0

        for stage_idx, sample_count in enumerate(schedule, start=1):
            samples_now = borell_samples[:sample_count]
            p_hat = float(np.mean(samples_now > 1.0))
            half_delta_factor = max(3.0 * delta / (np.pi**2 * stage_idx**2), 1e-12)
            full_delta_factor = max(6.0 * delta / (np.pi**2 * stage_idx**2), 1e-12)
            c_upper = float(np.sqrt(max(0.0, 2.0 * abs(np.log(half_delta_factor)) / sample_count)))
            c_lower = float(np.sqrt(max(0.0, 2.0 * abs(np.log(full_delta_factor)) / sample_count)))
            p_mc_plus = min(1.0, p_hat + np.sqrt(alpha * (1.0 - alpha)) * c_upper)
            p_mc_minus = max(0.0, p_hat - 0.25 * c_lower**2 - np.sqrt(alpha) * c_lower)
            chi = min(1.0 - 1e-12, 1.0 - half_delta_factor)
            beta_plus = min(1.0, 0.5 + self._normal.inv_cdf(chi) / np.sqrt(4.0 * sample_count))
            q_plus = float(np.quantile(samples_now, beta_plus, method="linear"))
            p_borell_plus = 1.0 - self._normal.cdf((1.0 - q_plus) / sigma_tilde)
            safe_upper = min(p_borell_plus, p_mc_plus)
            last_upper = safe_upper
            if safe_upper <= alpha:
                return "safe", float(safe_upper)
            if p_mc_minus >= alpha:
                return "unsafe", float(p_mc_minus)
        return "undecided", float(last_upper)

    def evaluate_trajectory(self, endpoint: np.ndarray, trajectory: np.ndarray) -> TrajectoryCandidateInfo:
        X_plan = np.asarray(trajectory, dtype=float)[: int(self.cfg.horizon)]
        mean_g, cov_g = self.safe_model.gp.posterior(X_plan, return_cov=True)
        status, unsafe_upper = self._abm_unsafe_upper_bound(mean_g, cov_g)
        feasible = status == "safe" and X_plan.shape[0] >= int(self.cfg.horizon)
        acquisition = (
            self.dyn_model.rollout_uncertainty_score(X_plan, criterion=self.cfg.tebbe_criterion)
            if feasible
            else float("-inf")
        )
        lcb = float(self.safe_model.lcb(X_plan[0], self.cfg.beta_g)) if X_plan.size else float("-inf")
        return TrajectoryCandidateInfo(
            endpoint=np.asarray(endpoint, dtype=float).copy(),
            trajectory=np.asarray(trajectory, dtype=float).copy(),
            feasible_full_horizon=bool(feasible),
            acquisition=float(acquisition),
            pointwise_lcb=lcb,
            trajectory_safety_prob=float(max(0.0, 1.0 - unsafe_upper)),
            min_safety_slack=lcb,
            unsafe_upper_bound=float(unsafe_upper),
        )
