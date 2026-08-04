from dataclasses import dataclass
from itertools import product
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment import NARXDoubleIntegratorEnv
from gp_models import DynamicsGP, SafetyGP, normal_cdf_array


def evaluate_planned_rollout(
    env: NARXDoubleIntegratorEnv,
    y_t: float,
    y_tm1: float,
    planned_actions: Sequence[float],
    horizon: int,
    epsilon: float,
) -> Dict[str, float]:
    if len(planned_actions) == 0:
        return {
            "immediate_margin": float("-inf"),
            "best_margin": float("-inf"),
            "immediate_safe": 0.0,
            "future_safe": 0.0,
            "interior_future_safe": 0.0,
            "dead_end": 0.0,
            "epsilon_dead_end": 0.0,
            "plan_complete": 0.0,
        }

    current_y_t = float(y_t)
    current_y_tm1 = float(y_tm1)
    margins: List[float] = []
    for action in planned_actions[:horizon]:
        z = env.make_regressor(current_y_t, current_y_tm1, float(action))
        margins.append(env.safety_value(z))
        y_next = env.step(z, noise=False)
        current_y_tm1, current_y_t = current_y_t, y_next

    plan_complete = len(planned_actions) >= horizon
    best_margin = float(min(margins)) if plan_complete else float("-inf")
    immediate_margin = float(margins[0])
    return {
        "immediate_margin": immediate_margin,
        "best_margin": best_margin,
        "immediate_safe": float(immediate_margin >= 0.0),
        "future_safe": float(best_margin >= 0.0),
        "interior_future_safe": float(best_margin >= epsilon),
        "dead_end": float(immediate_margin >= 0.0 and best_margin < 0.0),
        "epsilon_dead_end": float(immediate_margin >= epsilon and best_margin <= -epsilon),
        "plan_complete": float(plan_complete),
    }


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
    tebbe_sample_start: int = 100
    tebbe_sample_stages: int = 17
    tebbe_endpoint_candidates: int = 11
    tebbe_legacy_mc_prefix: bool = False
    tebbe_local_refinement: bool = False
    fa_safety_buffer_weight: float = 0.0
    fa_beam_width: int = 5
    fa_continuation_policy: str = "uncertainty-max"
    fa_continuation_seed: int = 0
    control_target: float = 4.45
    control_q_y: float = 1.0
    control_q_velocity: float = 0.25
    control_r_u: float = 0.05
    control_r_delta_u: float = 0.05
    control_terminal_weight: float = 2.0
    control_beam_width: int = 1
    active_mpc_information_weight: float = 1.0
    nominal_mpc_safety_margin: float = 0.0
    control_safety_limit: float = 0.0
    control_safety_direction: str = "upper"
    output_target: float = 0.0


@dataclass
class ConfidenceSchedule:
    delta_f: float = 0.05
    delta_g: float = 0.05
    y_min: float = -2.0
    y_max: float = 6.0
    y_grid_points: int = 121
    beta_multiplier: float = 1.0

    def pi_t(self, round_idx: int) -> float:
        return (np.pi**2 * float(round_idx) ** 2) / 6.0

    def state_domain_size(self, env: NARXDoubleIntegratorEnv) -> int:
        action_count = len(env.candidate_actions())
        return int(self.y_grid_points * self.y_grid_points * action_count)

    def beta(self, round_idx: int, domain_size: int, delta: float) -> float:
        return float(
            self.beta_multiplier
            * 2.0
            * np.log(domain_size * self.pi_t(round_idx) / delta)
        )

    def beta_f(self, round_idx: int, env: NARXDoubleIntegratorEnv) -> float:
        return self.beta(round_idx, self.state_domain_size(env), self.delta_f)

    def beta_g(self, round_idx: int, env: NARXDoubleIntegratorEnv) -> float:
        return self.beta(round_idx, self.state_domain_size(env), self.delta_g)


@dataclass
class CandidateInfo:
    u0: float
    certified_depth: int
    feasible_full_horizon: bool
    acquisition: float
    pointwise_lcb: float
    planned_actions: List[float]
    planned_states: List[np.ndarray]
    planned_outputs: List[float]
    rollout_stds: List[float]
    max_rollout_std: float
    trajectory_safety_prob: float = 0.0
    min_safety_slack: float = float("-inf")
    trajectory_param: Optional[float] = None


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


class ControlTrajectoryPlanner:


    def __init__(self, dyn_model, safe_model, cfg, safety_limit, safety_direction, output_target):
        if safety_direction not in {"upper", "lower"}:
            raise ValueError("safety_direction must be upper or lower")
        self.dyn_model, self.safe_model, self.cfg = dyn_model, safe_model, cfg
        self.safety_limit = float(safety_limit)
        self.safety_direction = safety_direction
        self.output_target = float(output_target)

    def _margin(self, outputs):
        outputs = np.asarray(outputs, dtype=float)
        return self.safety_limit - outputs if self.safety_direction == "upper" else outputs - self.safety_limit

    def _control_cost(self, endpoint, means):
        endpoint = np.asarray(endpoint, dtype=float).reshape(-1)
        return float(np.sum((means - self.output_target) ** 2) + self.cfg.control_r_u * np.sum(endpoint ** 2))

    def _constraint_margins(self, trajectory, means, stds):
        return self._margin(means)

    def evaluate_trajectory(self, endpoint, trajectory):
        X_plan = np.asarray(trajectory, dtype=float)[: int(self.cfg.horizon)]
        means, stds = self.dyn_model.predict_batch(X_plan)
        means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
        margins = self._constraint_margins(X_plan, means, stds)
        feasible = bool(X_plan.shape[0] >= int(self.cfg.horizon) and np.all(margins >= 0.0))
        return TrajectoryCandidateInfo(
            endpoint=np.asarray(endpoint, dtype=float).copy(), trajectory=np.asarray(trajectory, dtype=float).copy(),
            feasible_full_horizon=feasible, acquisition=-self._control_cost(endpoint, means),
            pointwise_lcb=float(margins[0]) if margins.size else float("-inf"),
            trajectory_safety_prob=1.0 if feasible else 0.0,
            min_safety_slack=float(np.min(margins)) if margins.size else float("-inf"),
            unsafe_upper_bound=0.0 if feasible else 1.0)

    def select_trajectory(self, candidates):
        infos = [self.evaluate_trajectory(endpoint, trajectory) for endpoint, trajectory in candidates]
        feasible = [info for info in infos if info.feasible_full_horizon]
        if not feasible:
            return None, None, None, infos
        best = max(feasible, key=lambda info: float(info.acquisition))
        return best.endpoint.copy(), best.trajectory.copy(), best, infos


class NominalMPCTrajectoryPlanner(ControlTrajectoryPlanner):
    pass


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
        self._trajectory_abm_sample_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._trajectory_abm_prefix_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _sample_schedule(self) -> List[int]:
        return [int(self.cfg.tebbe_sample_start * (2 ** r)) for r in range(max(1, int(self.cfg.tebbe_sample_stages)))]

    def _standard_normal_batch(self, dim: int, stage_idx: int, sample_count: int) -> np.ndarray:
        key = (int(dim), int(stage_idx), int(sample_count))
        cached = self._trajectory_abm_sample_cache.get(key)
        if cached is None:
            seed = np.random.SeedSequence([0, int(dim), int(stage_idx)])
            cached = np.random.default_rng(seed).standard_normal((int(sample_count), int(dim)))
            self._trajectory_abm_sample_cache[key] = cached
        return cached

    def _legacy_standard_normals(self, dim: int, total_samples: int) -> np.ndarray:
        key = (int(dim), int(total_samples))
        cached = self._trajectory_abm_prefix_cache.get(key)
        if cached is None:
            cached = np.random.default_rng(0).standard_normal((int(total_samples), int(dim)))
            self._trajectory_abm_prefix_cache[key] = cached
        return cached

    def _abm_unsafe_upper_bound(self, mean_g: np.ndarray, cov_g: np.ndarray) -> Tuple[str, float]:
        mean_g = np.asarray(mean_g, dtype=float).reshape(-1)
        cov_g = np.asarray(cov_g, dtype=float)
        if not np.all(mean_g > 0.0):
            return "unsafe", 1.0

        schedule = self._sample_schedule()
        alpha = float(self.cfg.tebbe_alpha)
        delta = float(self.cfg.tebbe_confidence_delta)
        sigma_tilde = float(np.sqrt(np.max(np.diag(cov_g) / np.maximum(mean_g**2, 1e-12))))
        if sigma_tilde <= 1e-12:
            return "safe", 0.0

        cov_x = cov_g / np.outer(mean_g, mean_g)
        cov_x = 0.5 * (cov_x + cov_x.T)
        eye = np.eye(cov_x.shape[0])
        chol = None
        for jitter in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
            try:
                chol = np.linalg.cholesky(cov_x + jitter * eye)
                break
            except np.linalg.LinAlgError:
                continue
        if chol is None:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_x)
            cov_x = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            cov_x = 0.5 * (cov_x + cov_x.T)
            chol = np.linalg.cholesky(cov_x + 1e-10 * eye)
        last_upper = 1.0
        legacy_maxima = None
        sampled_maxima: List[np.ndarray] = []
        previous_count = 0
        if self.cfg.tebbe_legacy_mc_prefix:
            base = self._legacy_standard_normals(cov_x.shape[0], schedule[-1])
            legacy_maxima = np.max(base @ chol.T, axis=1)

        for stage_idx, sample_count in enumerate(schedule, start=1):
            if legacy_maxima is not None:
                samples_now = legacy_maxima[:sample_count]
            else:
                batch_count = int(sample_count - previous_count)
                base_batch = self._standard_normal_batch(cov_x.shape[0], stage_idx, batch_count)
                sampled_maxima.append(np.max(base_batch @ chol.T, axis=1))
                samples_now = np.concatenate(sampled_maxima)
                previous_count = int(sample_count)
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


class FutureAwarePlanner:
    def __init__(self, env: NARXDoubleIntegratorEnv, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        self.env = env
        self.dyn_model = dyn_model
        self.safe_model = safe_model
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.fa_continuation_seed))
        if self.cfg.fa_continuation_policy not in {
            "uncertainty-max",
            "random-safe",
            "greedy-margin",
        }:
            raise ValueError(
                "fa_continuation_policy must be one of "
                "{'uncertainty-max', 'random-safe', 'greedy-margin'}."
            )

    def evaluate_candidate(self, y_t: float, y_tm1: float, u0: float, actions: np.ndarray) -> CandidateInfo:
        beam_width = max(1, int(self.cfg.fa_beam_width))
        A = self.cfg.Lx + self.cfg.Ly * self.cfg.Lf
        z0 = self.env.make_regressor(y_t, y_tm1, float(u0))
        lcb_1 = self.safe_model.lcb(z0, self.cfg.beta_g)

        if lcb_1 < 0.0:
            return CandidateInfo(
                u0=float(u0),
                certified_depth=0,
                feasible_full_horizon=False,
                acquisition=float("-inf"),
                pointwise_lcb=float(lcb_1),
                planned_actions=[float(u0)],
                planned_states=[z0.copy()],
                planned_outputs=[],
                rollout_stds=[],
                max_rollout_std=0.0,
                min_safety_slack=float(lcb_1),
            )

        beams = [
            (
                z0.copy(),
                [float(u0)],
                [z0.copy()],
                [],
                [],
                0.0,
                float(lcb_1),
            )
        ]
        best_partial = beams[0]

        for _ in range(1, self.cfg.horizon + 1):
            next_beams = []

            for current_state, act_plan, state_plan, outputs, stds, e_j, min_slack in beams:
                y_hat, sigma_f = self.dyn_model.predict(current_state)
                outputs_new = outputs + [float(y_hat)]
                stds_new = stds + [float(sigma_f)]

                if len(act_plan) >= self.cfg.horizon:
                    next_beams.append(
                        (
                            current_state,
                            act_plan,
                            state_plan,
                            outputs_new,
                            stds_new,
                            e_j,
                            min_slack,
                        )
                    )
                    continue

                e_next = A * e_j + self.cfg.Ly * np.sqrt(self.cfg.beta_f) * sigma_f
                z_next_all = np.asarray(
                    [self.env.shift(current_state, y_hat, float(v)) for v in actions],
                    dtype=float,
                )
                next_lcbs = self.safe_model.lcb_batch(z_next_all, self.cfg.beta_g)
                safe_mask = next_lcbs >= self.cfg.L_ell * e_next

                if not np.any(safe_mask):
                    next_beams.append(
                        (
                            current_state,
                            act_plan,
                            state_plan,
                            outputs_new,
                            stds_new,
                            e_j,
                            min_slack,
                        )
                    )
                    continue

                safe_states = z_next_all[safe_mask]
                safe_actions = actions[safe_mask]
                safe_lcbs = next_lcbs[safe_mask]
                safe_sigmas = self.dyn_model.predict_batch(safe_states)[1]

                successors = []
                for sigma_next, next_lcb, next_action, next_state in zip(
                    safe_sigmas,
                    safe_lcbs,
                    safe_actions,
                    safe_states,
                ):
                    slack = float(next_lcb - self.cfg.L_ell * e_next)
                    successors.append(
                        (
                            float(sigma_next),
                            (
                                next_state.copy(),
                                act_plan + [float(next_action)],
                                state_plan + [next_state.copy()],
                                outputs_new,
                                stds_new,
                                float(e_next),
                                min(float(min_slack), slack),
                            ),
                        )
                    )

                if self.cfg.fa_continuation_policy == "uncertainty-max":
                    successors.sort(key=lambda item: item[0], reverse=True)
                elif self.cfg.fa_continuation_policy == "greedy-margin":
                    successors.sort(key=lambda item: item[1][6], reverse=True)
                else:
                    self.rng.shuffle(successors)
                next_beams.extend([item[1] for item in successors[:beam_width]])

            next_beams.sort(
                key=lambda item: (
                    len(item[1]),
                    float(np.sum(item[4])) if item[4] else 0.0,
                    item[6],
                ),
                reverse=True,
            )
            beams = next_beams[:beam_width] if next_beams else beams

            if len(beams[0][1]) > len(best_partial[1]):
                best_partial = beams[0]

            if len(beams[0][1]) >= self.cfg.horizon:
                break

        full_beams = [beam for beam in beams if len(beam[1]) >= self.cfg.horizon]
        if full_beams:
            scored = []
            for beam in full_beams:
                _, _, state_plan, _, _, _, _ = beam
                X_plan = np.asarray(state_plan[: self.cfg.horizon], dtype=float)
                acquisition = self.dyn_model.rollout_information_gain(X_plan)
                scored.append((float(acquisition), beam))
            scored.sort(key=lambda item: item[0], reverse=True)
            acquisition, chosen = scored[0]
            feasible_full_horizon = True
        else:
            chosen = best_partial
            _, _, state_plan, _, _, _, _ = chosen
            if len(state_plan) >= 1:
                X_plan = np.asarray(state_plan, dtype=float)
                acquisition = self.dyn_model.rollout_information_gain(X_plan)
            else:
                acquisition = float("-inf")
            feasible_full_horizon = False

        current_state, act_plan, state_plan, outputs, stds, _, min_slack = chosen
        return CandidateInfo(
            u0=float(u0),
            certified_depth=len(act_plan),
            feasible_full_horizon=bool(feasible_full_horizon),
            acquisition=float(acquisition),
            pointwise_lcb=float(lcb_1),
            planned_actions=list(act_plan),
            planned_states=[np.asarray(s, dtype=float).copy() for s in state_plan],
            planned_outputs=list(outputs),
            rollout_stds=list(stds),
            max_rollout_std=float(max(stds) if stds else 0.0),
            min_safety_slack=float(min_slack),
        )

    def candidate_infos(self, y_t: float, y_tm1: float, actions: np.ndarray) -> List[CandidateInfo]:
        return [self.evaluate_candidate(y_t, y_tm1, float(u), actions) for u in actions]

    def select_action(
        self,
        y_t: float,
        y_tm1: float,
        actions: np.ndarray,
    ) -> Tuple[Optional[float], Optional[CandidateInfo], List[CandidateInfo]]:
        infos = self.candidate_infos(y_t, y_tm1, actions)
        feasible = [info for info in infos if info.feasible_full_horizon]
        if feasible:
            best = max(feasible, key=lambda info: float(info.acquisition))
            return best.u0, best, infos
        return None, None, infos


class FARandomPlanner(FutureAwarePlanner):
    def __init__(
        self,
        env: NARXDoubleIntegratorEnv,
        dyn_model: DynamicsGP,
        safe_model: SafetyGP,
        cfg: PlannerConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(env, dyn_model, safe_model, cfg)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def select_action(
        self,
        y_t: float,
        y_tm1: float,
        actions: np.ndarray,
    ) -> Tuple[Optional[float], Optional[CandidateInfo], List[CandidateInfo]]:
        infos = self.candidate_infos(y_t, y_tm1, actions)
        feasible = [info for info in infos if info.feasible_full_horizon]
        if feasible:
            chosen_idx = int(self.rng.integers(len(feasible)))
            chosen = feasible[chosen_idx]
            return chosen.u0, chosen, infos
        return None, None, infos


class NominalMPCPlanner:

    def __init__(self, env, dyn_model, safe_model, cfg):
        self.env, self.dyn_model, self.safe_model, self.cfg = env, dyn_model, safe_model, cfg
        self.previous_action = 0.0

    def set_trajectory_anchor(self, action):
        self.previous_action = float(action)

    def _stage_cost(self, y, y_prev, action, action_prev):
        velocity = (float(y)-float(y_prev))/max(float(self.env.dt), 1e-12)
        return float(self.cfg.control_q_y*(float(y)-self.cfg.control_target)**2
                     + self.cfg.control_q_velocity*velocity**2
                     + self.cfg.control_r_u*float(action)**2
                     + self.cfg.control_r_delta_u*(float(action)-float(action_prev))**2)

    def _evaluate_plan(self, y_t, y_tm1, plan):
        y, y_prev, u_prev = float(y_t), float(y_tm1), self.previous_action
        states, outputs, stds, cost = [], [], [], 0.0
        min_slack, feasible = float('inf'), True
        for action in plan:
            z = self.env.make_regressor(y, y_prev, float(action)); states.append(z.copy())
            cost += self._stage_cost(y, y_prev, action, u_prev)
            y_next = self.env.nominal_transition_mean(z)
            std = 0.0
            outputs.append(float(y_next)); stds.append(float(std))
            slack = float(self.env.y_max-self.cfg.nominal_mpc_safety_margin-abs(y_next))
            min_slack, feasible = min(min_slack, slack), bool(feasible and slack >= 0.0)
            y_prev, y, u_prev = y, float(y_next), float(action)
        cost += self.cfg.control_terminal_weight*self.cfg.control_q_y*(y-self.cfg.control_target)**2
        return CandidateInfo(u0=float(plan[0]), certified_depth=len(plan) if feasible else 0,
            feasible_full_horizon=feasible, acquisition=-float(cost), pointwise_lcb=float(min_slack),
            planned_actions=list(map(float, plan)), planned_states=states, planned_outputs=outputs,
            rollout_stds=stds, max_rollout_std=float(max(stds) if stds else 0.0),
            min_safety_slack=float(min_slack))

    def _evaluate_plans(self, y_t, y_tm1, plans):
        return [self._evaluate_plan(y_t, y_tm1, plan) for plan in plans]

    def candidate_infos(self, y_t, y_tm1, actions):
        action_values = np.asarray(actions, dtype=float)
        horizon = max(1, int(self.cfg.horizon))
        beam_width = max(1, int(self.cfg.control_beam_width))
        best = {}
        for first_action in action_values:
            beams = [(float(first_action),)]
            for _ in range(1, horizon):
                expanded = [plan + (float(action),) for plan in beams for action in action_values]
                scored = list(zip(self._evaluate_plans(y_t, y_tm1, expanded), expanded))
                feasible = [item for item in scored if item[0].feasible_full_horizon]
                pool = feasible if feasible else scored
                pool.sort(key=lambda item: float(item[0].acquisition), reverse=True)
                beams = [plan for _, plan in pool[:beam_width]]
            candidates = self._evaluate_plans(y_t, y_tm1, beams)
            best[float(first_action)] = max(
                candidates, key=lambda info: (info.feasible_full_horizon, info.acquisition)
            )
        return [best[float(action)] for action in action_values]

    def select_action(self, y_t, y_tm1, actions):
        infos = self.candidate_infos(y_t, y_tm1, actions)
        feasible = [i for i in infos if i.feasible_full_horizon]
        if not feasible: return None, None, infos
        chosen = max(feasible, key=lambda i: float(i.acquisition))
        return chosen.u0, chosen, infos


class ActiveLearningMPCPlanner(NominalMPCPlanner):


    def _evaluate_plan(self, y_t, y_tm1, plan):
        info = super()._evaluate_plan(y_t, y_tm1, plan)
        X_plan = np.asarray(info.planned_states, dtype=float)
        if X_plan.size:
            information_gain = self.dyn_model.rollout_information_gain(X_plan)
            info.acquisition += float(self.cfg.active_mpc_information_weight) * float(information_gain)
        return info

    def _evaluate_plans(self, y_t, y_tm1, plans):
        infos = [
            NominalMPCPlanner._evaluate_plan(self, y_t, y_tm1, plan)
            for plan in plans
        ]
        if not infos:
            return infos
        blocks = np.asarray([info.planned_states for info in infos], dtype=float)
        gains = self.dyn_model.rollout_information_gain_batch(blocks)
        for info, gain in zip(infos, gains):
            info.acquisition += float(self.cfg.active_mpc_information_weight) * float(gain)
        return infos


class PointwiseSafePlanner:
    def __init__(self, env: NARXDoubleIntegratorEnv, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        self.env = env
        self.dyn_model = dyn_model
        self.safe_model = safe_model
        self.cfg = cfg

    def evaluate_candidate(self, y_t: float, y_tm1: float, u0: float, actions: np.ndarray) -> CandidateInfo:
        z = self.env.make_regressor(y_t, y_tm1, float(u0))
        lcb_1 = self.safe_model.lcb(z, self.cfg.beta_g)
        planned_actions = [float(u0)]
        planned_states = [z.copy()]
        planned_outputs: List[float] = []
        rollout_stds: List[float] = []
        current_state = z.copy()

        while True:
            y_hat, sigma_f = self.dyn_model.predict(current_state)
            planned_outputs.append(y_hat)
            rollout_stds.append(sigma_f)
            if len(planned_actions) >= self.cfg.horizon or lcb_1 < 0.0:
                break

            z_next_all = np.asarray(
                [self.env.shift(current_state, y_hat, float(v)) for v in actions],
                dtype=float,
            )
            next_lcbs = self.safe_model.lcb_batch(z_next_all, self.cfg.beta_g)
            safe_mask = next_lcbs >= 0.0
            safe_successors = []
            if np.any(safe_mask):
                sigma_next_all = self.dyn_model.predict_batch(z_next_all[safe_mask])[1]
                for sigma_next, next_action, next_state in zip(
                    sigma_next_all,
                    actions[safe_mask],
                    z_next_all[safe_mask],
                ):
                    safe_successors.append((float(sigma_next), float(next_action), next_state))

            if not safe_successors:
                break

            safe_successors.sort(key=lambda item: item[0], reverse=True)
            _, next_action, next_state = safe_successors[0]
            planned_actions.append(next_action)
            planned_states.append(next_state.copy())
            current_state = next_state

        acquisition = self.dyn_model.rollout_information_gain(z.reshape(1, -1)) if lcb_1 >= 0.0 else float("-inf")
        return CandidateInfo(
            u0=float(u0),
            certified_depth=1 if lcb_1 >= 0.0 else 0,
            feasible_full_horizon=bool(lcb_1 >= 0.0),
            acquisition=acquisition,
            pointwise_lcb=lcb_1,
            planned_actions=planned_actions,
            planned_states=planned_states,
            planned_outputs=planned_outputs,
            rollout_stds=rollout_stds,
            max_rollout_std=float(max(rollout_stds) if rollout_stds else 0.0),
        )

    def candidate_infos(self, y_t: float, y_tm1: float, actions: np.ndarray) -> List[CandidateInfo]:
        return [self.evaluate_candidate(y_t, y_tm1, float(u), actions) for u in actions]

    def select_action(
        self,
        y_t: float,
        y_tm1: float,
        actions: np.ndarray,
    ) -> Tuple[Optional[float], Optional[CandidateInfo], List[CandidateInfo]]:
        infos = self.candidate_infos(y_t, y_tm1, actions)
        feasible = [info for info in infos if info.feasible_full_horizon]
        if feasible:
            best = max(feasible, key=lambda item: item.acquisition)
            return best.u0, best, infos
        fallback = min(infos, key=lambda item: item.u0)
        return float(np.min(actions)), fallback, infos


class SafeExplorationPlanner:
    def __init__(self, env: NARXDoubleIntegratorEnv, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        self.env = env
        self.dyn_model = dyn_model
        self.safe_model = safe_model
        self.cfg = cfg

    def candidate_infos(self, y_t: float, y_tm1: float, actions: np.ndarray) -> List[CandidateInfo]:
        states = np.asarray(
            [self.env.make_regressor(y_t, y_tm1, float(u)) for u in actions],
            dtype=float,
        )
        mean_g, std_g = self.safe_model.predict_batch(states)
        safety_probs = normal_cdf_array(mean_g / np.maximum(std_g, 1e-8))
        _, dyn_stds = self.dyn_model.predict_batch(states)
        lcb_values = self.safe_model.lcb_batch(states, self.cfg.beta_g)
        required_prob = 1.0 - float(self.cfg.safe_exploration_alpha)

        infos = []
        for action, state, prob_safe, dyn_std, lcb in zip(actions, states, safety_probs, dyn_stds, lcb_values):
            feasible = float(prob_safe) >= required_prob
            acquisition = float(np.log(max(float(dyn_std) ** 2, 1e-12))) if feasible else float("-inf")
            infos.append(
                CandidateInfo(
                    u0=float(action),
                    certified_depth=1 if feasible else 0,
                    feasible_full_horizon=bool(feasible),
                    acquisition=acquisition,
                    pointwise_lcb=float(lcb),
                    planned_actions=[float(action)],
                    planned_states=[np.asarray(state, dtype=float).copy()],
                    planned_outputs=[],
                    rollout_stds=[float(dyn_std)],
                    max_rollout_std=float(dyn_std),
                    trajectory_safety_prob=float(prob_safe),
                )
            )
        return infos

    def evaluate_candidate(self, y_t: float, y_tm1: float, u0: float, actions: np.ndarray) -> CandidateInfo:
        infos = self.candidate_infos(y_t, y_tm1, np.asarray([float(u0)], dtype=float))
        return infos[0]

    def select_action(
        self,
        y_t: float,
        y_tm1: float,
        actions: np.ndarray,
    ) -> Tuple[Optional[float], Optional[CandidateInfo], List[CandidateInfo]]:
        infos = self.candidate_infos(y_t, y_tm1, actions)
        feasible = [info for info in infos if info.feasible_full_horizon]
        if feasible:
            best = max(feasible, key=lambda item: item.acquisition)
            return best.u0, best, infos
        return None, None, infos


class SALNXPlanner:
    def __init__(self, env: NARXDoubleIntegratorEnv, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        self.env = env
        self.dyn_model = dyn_model
        self.safe_model = safe_model
        self.cfg = cfg
        self._trajectory_anchor = 0.0

    def set_trajectory_anchor(self, u_anchor: float) -> None:
        self._trajectory_anchor = float(np.clip(u_anchor, -self.env.u_max, self.env.u_max))

    def _trajectory_safe_probability(self, planned_states: Sequence[np.ndarray]) -> float:
        X_plan = np.asarray(planned_states, dtype=float)
        return self.safe_model.trajectory_safe_probability(
            X_plan,
            mc_samples=self.cfg.salnx_joint_mc_samples,
        )

    def _trajectory_actions_from_eta(self, eta: float, actions: np.ndarray) -> List[float]:
        eta = float(np.clip(float(eta), -self.env.u_max, self.env.u_max))

        if self.cfg.horizon <= 1:
            raw = np.array([eta], dtype=float)
        else:
            raw = np.linspace(
                float(self._trajectory_anchor),
                eta,
                self.cfg.horizon + 1,
            )[1:]

        snapped = []

        for value in raw:
            idx = int(np.argmin(np.abs(actions - value)))
            snapped.append(float(actions[idx]))
        return snapped

    def _simulate_mean_trajectory(
        self,
        y_t: float,
        y_tm1: float,
        action_plan: Sequence[float],
    ) -> Tuple[List[np.ndarray], List[float], List[float]]:
        states: List[np.ndarray] = []
        outputs: List[float] = []
        stds: List[float] = []
        current_state = self.env.make_regressor(y_t, y_tm1, float(action_plan[0]))
        states.append(current_state.copy())

        for step_idx in range(len(action_plan)):
            y_hat, sigma_f = self.dyn_model.predict(current_state)
            outputs.append(float(y_hat))
            stds.append(float(sigma_f))
            if step_idx + 1 >= len(action_plan):
                break
            next_action = float(action_plan[step_idx + 1])
            current_state = self.env.shift(current_state, y_hat, next_action)
            states.append(current_state.copy())
        return states, outputs, stds

    def candidate_infos(self, y_t: float, y_tm1: float, actions: np.ndarray) -> List[CandidateInfo]:
        return [self.evaluate_candidate(y_t, y_tm1, float(eta), actions) for eta in actions]

    def _batch_endpoint_candidates(self, actions: np.ndarray) -> np.ndarray:
        return np.asarray(actions, dtype=float)

    def _batch_action_plans(self, anchors: np.ndarray, endpoints: np.ndarray, actions: np.ndarray) -> np.ndarray:
        horizon = int(self.cfg.horizon)
        raw = np.asarray(
            [
                [np.linspace(float(anchor), float(endpoint), horizon + 1)[1:]
                 for endpoint in endpoints]
                for anchor in anchors
            ],
            dtype=float,
        )
        distances = np.abs(raw[..., None] - np.asarray(actions, dtype=float))
        return np.asarray(actions, dtype=float)[np.argmin(distances, axis=-1)]

    def _batch_feasible_from_posterior(self, mean_g: np.ndarray, cov_g: np.ndarray) -> np.ndarray:
        _, horizon = mean_g.shape
        if horizon == 1:
            std = np.sqrt(np.maximum(cov_g[:, 0, 0], 0.0))
            probability = normal_cdf_array(mean_g[:, 0] / np.maximum(std, 1e-8))
        else:
            chol = np.linalg.cholesky(cov_g + 1e-10 * np.eye(horizon)[None, :, :])
            base = self.safe_model._orthant_standard_normals(horizon, self.cfg.salnx_joint_mc_samples)
            samples = mean_g[:, None, :] + np.einsum("sh,bkh->bsk", base, chol)
            probability = np.mean(np.all(samples >= 0.0, axis=2), axis=1)
        return probability >= (1.0 - float(self.cfg.alpha_safety))

    def batch_any_feasible(self, X_eval: np.ndarray, actions: np.ndarray, chunk_size: int = 32) -> np.ndarray:

        X_eval = np.asarray(X_eval, dtype=float)
        actions = np.asarray(actions, dtype=float)
        endpoints = self._batch_endpoint_candidates(actions)
        candidate_count = len(endpoints)
        results: List[np.ndarray] = []
        for start in range(0, len(X_eval), max(1, int(chunk_size))):
            states = X_eval[start:start + max(1, int(chunk_size))]
            action_plans = self._batch_action_plans(states[:, 2], endpoints, actions)
            state_count, _, horizon = action_plans.shape
            flat_count = state_count * candidate_count
            flat_plans = action_plans.reshape(flat_count, horizon)
            current = np.column_stack((np.repeat(states[:, 0], candidate_count), np.repeat(states[:, 1], candidate_count), flat_plans[:, 0]))
            trajectories = np.empty((flat_count, horizon, 3), dtype=float)
            for step_idx in range(horizon):
                trajectories[:, step_idx, :] = current
                if step_idx + 1 >= horizon:
                    break
                y_next, _ = self.dyn_model.predict_batch(current)
                current = np.column_stack((y_next, current[:, 0], flat_plans[:, step_idx + 1]))
            mean_g, cov_g = self.safe_model.gp.posterior_blocks(trajectories)
            feasible = self._batch_feasible_from_posterior(mean_g, cov_g)
            results.append(feasible.reshape(state_count, candidate_count).any(axis=1))
        return np.concatenate(results) if results else np.empty(0, dtype=bool)

    def evaluate_candidate(self, y_t: float, y_tm1: float, u0: float, actions: np.ndarray) -> CandidateInfo:
        eta = float(u0)
        action_plan = self._trajectory_actions_from_eta(eta, actions)
        planned_states, planned_outputs, rollout_stds = self._simulate_mean_trajectory(
            y_t=y_t,
            y_tm1=y_tm1,
            action_plan=action_plan,
        )
        X_plan = np.asarray(planned_states, dtype=float)
        z0 = planned_states[0]
        lcb_1 = self.safe_model.lcb(z0, self.cfg.beta_g)
        trajectory_safety_prob = self._trajectory_safe_probability(planned_states)
        feasible = trajectory_safety_prob >= (1.0 - self.cfg.alpha_safety)
        acquisition = (
            self.dyn_model.rollout_uncertainty_score(
                X_plan,
                criterion=self.cfg.salnx_criterion,
            )
            if feasible
            else float("-inf")
        )

        return CandidateInfo(
            u0=float(action_plan[0]),
            certified_depth=len(action_plan) if feasible else 0,
            feasible_full_horizon=bool(len(action_plan) >= self.cfg.horizon and feasible),
            acquisition=float(acquisition),
            pointwise_lcb=lcb_1,
            planned_actions=list(action_plan),
            planned_states=[np.asarray(state, dtype=float).copy() for state in planned_states],
            planned_outputs=list(planned_outputs),
            rollout_stds=list(rollout_stds),
            max_rollout_std=float(max(rollout_stds) if rollout_stds else 0.0),
            trajectory_safety_prob=float(trajectory_safety_prob),
            trajectory_param=float(eta),
        )

    def select_action(
        self,
        y_t: float,
        y_tm1: float,
        actions: np.ndarray,
    ) -> Tuple[Optional[float], Optional[CandidateInfo], List[CandidateInfo]]:
        infos = self.candidate_infos(y_t, y_tm1, actions)
        feasible = [info for info in infos if info.feasible_full_horizon]
        if feasible:
            best = max(feasible, key=lambda item: item.acquisition)
            return best.u0, best, infos
        return None, None, infos


class TebbeABMPlanner(SALNXPlanner):
    def __init__(self, env: NARXDoubleIntegratorEnv, dyn_model: DynamicsGP, safe_model: SafetyGP, cfg: PlannerConfig):
        super().__init__(env, dyn_model, safe_model, cfg)
        self._normal = NormalDist()
        self._trajectory_anchor = 0.0
        self._abm_base_sample_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._abm_prefix_sample_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def set_trajectory_anchor(self, u_anchor: float) -> None:
        self._trajectory_anchor = float(np.clip(u_anchor, -self.env.u_max, self.env.u_max))

    def _sample_schedule(self) -> List[int]:
        return [int(self.cfg.tebbe_sample_start * (2 ** r)) for r in range(max(1, int(self.cfg.tebbe_sample_stages)))]

    def _standard_normal_batch(self, dim: int, stage_idx: int, sample_count: int) -> np.ndarray:
        key = (int(dim), int(stage_idx), int(sample_count))
        cached = self._abm_base_sample_cache.get(key)
        if cached is None:
            seed = np.random.SeedSequence([0, int(dim), int(stage_idx)])
            cached = np.random.default_rng(seed).standard_normal(
                (int(sample_count), int(dim))
            )
            self._abm_base_sample_cache[key] = cached
        return cached

    def _legacy_standard_normals(self, dim: int, total_samples: int) -> np.ndarray:
        key = (int(dim), int(total_samples))
        cached = self._abm_prefix_sample_cache.get(key)
        if cached is None:
            cached = np.random.default_rng(0).standard_normal((int(total_samples), int(dim)))
            self._abm_prefix_sample_cache[key] = cached
        return cached

    def _abm_unsafe_upper_bound(self, mean_g: np.ndarray, cov_g: np.ndarray) -> Tuple[str, float]:
        mean_g = np.asarray(mean_g, dtype=float).reshape(-1)
        cov_g = np.asarray(cov_g, dtype=float)

        schedule = self._sample_schedule()
        alpha = float(self.cfg.tebbe_alpha)
        delta = float(self.cfg.tebbe_confidence_delta)
        last_safe_upper = 1.0

        jitter = 1e-10 * np.eye(cov_g.shape[0])
        if not np.all(mean_g > 0.0):
            return "unsafe", 1.0

        sigma_tilde = float(np.sqrt(np.max(np.diag(cov_g) / np.maximum(mean_g**2, 1e-12))))
        if sigma_tilde <= 1e-12:
            return "safe", 0.0
        cov_x = cov_g / np.outer(mean_g, mean_g)
        cov_x = 0.5 * (cov_x + cov_x.T)
        chol_x = np.linalg.cholesky(cov_x + jitter)
        legacy_maxima = None
        sampled_maxima: List[np.ndarray] = []
        previous_count = 0
        if self.cfg.tebbe_legacy_mc_prefix:
            base = self._legacy_standard_normals(cov_g.shape[0], schedule[-1])
            legacy_maxima = np.max(base @ chol_x.T, axis=1)

        for stage_idx, sample_count in enumerate(schedule, start=1):
            if legacy_maxima is not None:
                samples_now = legacy_maxima[:sample_count]
            else:
                batch_count = int(sample_count - previous_count)
                base_batch = self._standard_normal_batch(cov_g.shape[0], stage_idx, batch_count)
                sampled_maxima.append(np.max(base_batch @ chol_x.T, axis=1))
                samples_now = np.concatenate(sampled_maxima)
                previous_count = int(sample_count)
            p_hat = float(np.mean(samples_now > 1.0))
            half_delta_factor = max(3.0 * delta / (np.pi**2 * stage_idx**2), 1e-12)
            full_delta_factor = max(6.0 * delta / (np.pi**2 * stage_idx**2), 1e-12)
            c_upper = float(np.sqrt(max(0.0, 2.0 * abs(np.log(half_delta_factor)) / sample_count)))
            c_lower = float(np.sqrt(max(0.0, 2.0 * abs(np.log(full_delta_factor)) / sample_count)))
            p_mc_plus = min(1.0, p_hat + np.sqrt(alpha * (1.0 - alpha)) * c_upper)
            p_mc_minus = max(0.0, p_hat - 0.25 * c_lower**2 - np.sqrt(alpha) * c_lower)
            last_safe_upper = p_mc_plus

            chi = min(1.0 - 1e-12, 1.0 - half_delta_factor)
            beta_margin = self._normal.inv_cdf(chi) / np.sqrt(4.0 * sample_count)
            beta_plus = min(1.0, 0.5 + beta_margin)
            q_plus = float(np.quantile(samples_now, beta_plus, method="linear"))
            p_borell_plus = 1.0 - self._normal.cdf((1.0 - q_plus) / sigma_tilde)
            safe_upper = min(p_borell_plus, p_mc_plus)
            last_safe_upper = safe_upper
            if safe_upper <= alpha:
                return "safe", float(safe_upper)
            if p_mc_minus >= alpha:
                return "unsafe", float(p_mc_minus)

        return "undecided", float(last_safe_upper)

    def _trajectory_actions_from_eta(self, eta: float, actions: np.ndarray) -> List[float]:
        eta = float(np.clip(float(eta), -self.env.u_max, self.env.u_max))
        return [
            float(value)
            for value in np.linspace(
                float(self._trajectory_anchor),
                eta,
                int(self.cfg.horizon) + 1,
            )[1:]
        ]

    def _evaluate_trajectory(self, y_t: float, y_tm1: float, eta: float, actions: np.ndarray) -> CandidateInfo:
        action_plan = self._trajectory_actions_from_eta(float(eta), actions)
        z0 = self.env.make_regressor(y_t, y_tm1, float(action_plan[0]))
        lcb_1 = self.safe_model.lcb(z0, self.cfg.beta_g)
        planned_states, planned_outputs, rollout_stds = self._simulate_mean_trajectory(
            y_t=y_t,
            y_tm1=y_tm1,
            action_plan=action_plan,
        )
        X_plan = np.asarray(planned_states, dtype=float)
        mean_g, cov_g = self.safe_model.gp.posterior(X_plan, return_cov=True)
        abm_status, unsafe_upper = self._abm_unsafe_upper_bound(mean_g, cov_g)
        if abm_status != "safe":
            return CandidateInfo(
                u0=float(action_plan[0]),
                certified_depth=0,
                feasible_full_horizon=False,
                acquisition=float("-inf"),
                pointwise_lcb=lcb_1,
                planned_actions=list(action_plan),
                planned_states=[np.asarray(state, dtype=float).copy() for state in planned_states],
                planned_outputs=list(planned_outputs),
                rollout_stds=list(rollout_stds),
                max_rollout_std=float(max(rollout_stds) if rollout_stds else 0.0),
                trajectory_safety_prob=float(max(0.0, 1.0 - unsafe_upper)),
                trajectory_param=float(eta),
            )

        acquisition = self.dyn_model.rollout_uncertainty_score(
            X_plan,
            criterion=self.cfg.tebbe_criterion,
        )
        return CandidateInfo(
            u0=float(action_plan[0]),
            certified_depth=len(action_plan),
            feasible_full_horizon=bool(len(action_plan) >= self.cfg.horizon),
            acquisition=acquisition,
            pointwise_lcb=lcb_1,
            planned_actions=list(action_plan),
            planned_states=[np.asarray(state, dtype=float).copy() for state in planned_states],
            planned_outputs=list(planned_outputs),
            rollout_stds=list(rollout_stds),
            max_rollout_std=float(max(rollout_stds) if rollout_stds else 0.0),
            trajectory_safety_prob=float(max(0.0, 1.0 - unsafe_upper)),
            trajectory_param=float(eta),
        )

    def evaluate_candidate(self, y_t: float, y_tm1: float, u0: float, actions: np.ndarray) -> CandidateInfo:
        return self._evaluate_trajectory(y_t, y_tm1, float(u0), actions)

    def _endpoint_candidates(self) -> np.ndarray:
        candidate_count = max(3, int(self.cfg.tebbe_endpoint_candidates))
        return np.linspace(-self.env.u_max, self.env.u_max, candidate_count)

    def _batch_endpoint_candidates(self, actions: np.ndarray) -> np.ndarray:
        return self._endpoint_candidates()

    def _batch_action_plans(self, anchors: np.ndarray, endpoints: np.ndarray, actions: np.ndarray) -> np.ndarray:
        horizon = int(self.cfg.horizon)
        return np.asarray(
            [
                [np.linspace(float(anchor), float(endpoint), horizon + 1)[1:]
                 for endpoint in endpoints]
                for anchor in anchors
            ],
            dtype=float,
        )

    def _batch_feasible_from_posterior(self, mean_g: np.ndarray, cov_g: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                self._abm_unsafe_upper_bound(mean, cov)[0] == "safe"
                for mean, cov in zip(mean_g, cov_g)
            ],
            dtype=bool,
        )

    def batch_any_feasible(self, X_eval: np.ndarray, actions: np.ndarray, chunk_size: int = 8) -> np.ndarray:
        return super().batch_any_feasible(X_eval, actions, chunk_size=chunk_size)

    def candidate_infos(self, y_t: float, y_tm1: float, actions: np.ndarray) -> List[CandidateInfo]:
        return [self._evaluate_trajectory(y_t, y_tm1, float(eta), actions) for eta in self._endpoint_candidates()]

    def select_action(
        self,
        y_t: float,
        y_tm1: float,
        actions: np.ndarray,
    ) -> Tuple[Optional[float], Optional[CandidateInfo], List[CandidateInfo]]:
        endpoints = self._endpoint_candidates()
        infos = [self._evaluate_trajectory(y_t, y_tm1, float(eta), actions) for eta in endpoints]
        feasible = [info for info in infos if info.feasible_full_horizon]
        if not feasible:
            return None, None, infos

        best = max(feasible, key=lambda item: item.acquisition)
        if self.cfg.tebbe_local_refinement:
            best_idx = int(np.argmin(np.abs(endpoints - float(best.trajectory_param))))
            left = float(endpoints[max(0, best_idx - 1)])
            right = float(endpoints[min(len(endpoints) - 1, best_idx + 1)])
            if right > left:
                phi = 0.5 * (np.sqrt(5.0) - 1.0)
                a, b = left, right
                c = b - phi * (b - a)
                d = a + phi * (b - a)

                def score(eta: float) -> Tuple[float, CandidateInfo]:
                    info = self._evaluate_trajectory(y_t, y_tm1, float(eta), actions)
                    if not info.feasible_full_horizon or not np.isfinite(info.acquisition):
                        return float("-inf"), info
                    return float(info.acquisition), info

                c_score, c_info = score(c)
                d_score, d_info = score(d)
                for _ in range(32):
                    if abs(b - a) <= 1e-3:
                        break
                    if c_score < d_score:
                        a, c, c_score, c_info = c, d, d_score, d_info
                        d = a + phi * (b - a)
                        d_score, d_info = score(d)
                    else:
                        b, d, d_score, d_info = d, c, c_score, c_info
                        c = b - phi * (b - a)
                        c_score, c_info = score(c)
                refined = c_info if c_score >= d_score else d_info
                infos.append(refined)
                if refined.feasible_full_horizon and refined.acquisition > best.acquisition:
                    best = refined
        return best.u0, best, infos
