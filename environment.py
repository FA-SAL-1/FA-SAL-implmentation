

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NARXDoubleIntegratorEnv:
    kind: str = "double_integrator"
    dt: float = 1.0
    a: float = 0.8
    b: float = 0.2
    c_d: float = 0.03
    u_max: float = 1.0
    y_max: float = 5.0
    sigma_eps: float = 0.03
    sigma_g: float = 0.03

    runaway_center: float = 3.0
    runaway_halfwidth: float = 0.8
    runaway_strength: float = 0.18
    runaway_velocity_threshold: float = 0.25

    funnel_center: float = 3.4
    funnel_halfwidth: float = 0.7
    funnel_runaway_strength: float = 0.28
    funnel_velocity_threshold: float = 0.15
    funnel_brake_fade: float = 0.72

    lattice_start: float = 2.2
    lattice_period: float = 1.15
    lattice_count: int = 3
    lattice_halfwidth: float = 0.32
    lattice_tailwidth: float = 0.9
    lattice_runaway_strength: float = 0.52
    lattice_tail_runaway_multiplier: float = 1.8
    lattice_velocity_threshold: float = 0.03
    lattice_brake_fade: float = 0.9
    lattice_tail_brake_fade: float = 0.96
    smooth_switches: bool = False
    switch_sharpness: float = 8.0

    goal_y_min: float = 4.2
    goal_y_max: float = 4.7

    fail_y: float = 5.2
    failure_absorbing: bool = True

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def candidate_actions(self) -> np.ndarray:
        return np.round(np.arange(-self.u_max, self.u_max + 1e-9, 0.2), 1)

    def make_regressor(self, y_t: float, y_tm1: float, u_t: float) -> np.ndarray:
        return np.array([y_t, y_tm1, u_t], dtype=float)

    def velocity(self, z: np.ndarray) -> float:
        return float((z[0] - z[1]) / self.dt)

    def nominal_transition_mean(self, z: np.ndarray) -> float:

        y_t, y_tm1, u_t = map(float, z)
        velocity = (y_t - y_tm1) / self.dt
        drag = self.c_d * self.dt * velocity * abs(velocity)
        return float(y_t + self.a * self.dt * velocity + self.b * self.dt * u_t - drag)

    def transition_mean(self, z: np.ndarray) -> float:
        y_t, y_tm1, u_t = map(float, z)
        velocity = (y_t - y_tm1) / self.dt

        if self.kind == "diagnostic_deadend":
            return y_t + velocity + 0.4 * u_t - 0.05 * velocity * abs(velocity)

        drag = self.c_d * self.dt * velocity * abs(velocity)
        control_gain = self.effective_control_gain(z)

        base = (
            y_t
            + self.a * self.dt * velocity
            + control_gain * self.dt * u_t
            - drag
        )

        if self.kind == "sparse_deadend":
            return base + self.lattice_push(z)

        return (
            base
            + self.runaway_push(z)
            + self.funnel_push(z)
            + self.lattice_push(z)
            + self.diagnostic_deadend_push(z)
        )

    def runaway_push(self, z: np.ndarray) -> float:
        if self.kind != "runaway_zone":
            return 0.0

        y_t = float(z[0])
        velocity = self.velocity(z)

        if velocity <= self.runaway_velocity_threshold:
            return 0.0

        distance = abs(y_t - self.runaway_center)
        if distance >= self.runaway_halfwidth:
            return 0.0

        shape = 1.0 - distance / max(self.runaway_halfwidth, 1e-8)
        return self.runaway_strength * shape * (
            velocity - self.runaway_velocity_threshold
        ) ** 2

    def inside_funnel(self, z: np.ndarray) -> bool:
        if self.kind not in {"commitment_funnel", "goal_funnel"}:
            return False

        y_t = float(z[0])
        return abs(y_t - self.funnel_center) < self.funnel_halfwidth

    def lattice_centers(self) -> np.ndarray:
        if self.lattice_count <= 0:
            return np.array([], dtype=float)
        return self.lattice_start + self.lattice_period * np.arange(
            self.lattice_count, dtype=float
        )

    def nearest_lattice_center(self, y_t: float) -> float:
        centers = self.lattice_centers()
        if len(centers) == 0:
            return float("nan")
        return float(centers[int(np.argmin(np.abs(centers - y_t)))])

    def inside_lattice_trap(self, z: np.ndarray) -> bool:
        if self.kind not in {"funnel_lattice", "sparse_deadend"}:
            return False

        y_t = float(z[0])
        nearest = self.nearest_lattice_center(y_t)

        if not np.isfinite(nearest):
            return False

        return abs(y_t - nearest) < self.lattice_halfwidth

    def inside_lattice_commitment(self, z: np.ndarray) -> bool:
        if self.kind not in {"funnel_lattice", "sparse_deadend"}:
            return False

        y_t = float(z[0])
        velocity = self.velocity(z)
        nearest = self.nearest_lattice_center(y_t)

        if not np.isfinite(nearest):
            return False

        left_edge = nearest - self.lattice_halfwidth
        right_edge = nearest + self.lattice_tailwidth

        return (
            velocity > self.lattice_velocity_threshold
            and left_edge <= y_t <= right_edge
        )

    def lattice_shape(self, y_t: float) -> float:
        nearest = self.nearest_lattice_center(y_t)

        if not np.isfinite(nearest):
            return 0.0

        distance = abs(y_t - nearest)
        if distance >= self.lattice_halfwidth:
            return 0.0

        return max(0.0, 1.0 - distance / max(self.lattice_halfwidth, 1e-8))

    def lattice_commitment_shape(self, y_t: float) -> float:
        nearest = self.nearest_lattice_center(y_t)

        if not np.isfinite(nearest):
            return 0.0

        if y_t <= nearest:
            left_edge = nearest - self.lattice_halfwidth
            if y_t <= left_edge:
                return 0.0
            return max(
                0.0,
                min(1.0, (y_t - left_edge) / max(self.lattice_halfwidth, 1e-8)),
            )

        right_edge = nearest + self.lattice_tailwidth
        if y_t >= right_edge:
            return 0.0

        return max(
            0.0,
            min(1.0, 1.0 - (y_t - nearest) / max(self.lattice_tailwidth, 1e-8)),
        )

    def effective_control_gain(self, z: np.ndarray) -> float:
        if self.smooth_switches and self.kind in {"funnel_lattice", "sparse_deadend"}:
            y_t = float(z[0])
            velocity = self.velocity(z)
            velocity_gate = self._sigmoid(
                self.switch_sharpness
                * (velocity - self.lattice_velocity_threshold)
            )
            remaining_control = 1.0
            for center in self.lattice_centers():
                left_edge = float(center - self.lattice_halfwidth)
                right_edge = float(center + self.lattice_tailwidth)
                window = self._smooth_window(y_t, left_edge, right_edge)
                tail_gate = self._sigmoid(
                    self.switch_sharpness * (y_t - float(center))
                )
                fade = (
                    self.lattice_brake_fade
                    + (self.lattice_tail_brake_fade - self.lattice_brake_fade)
                    * tail_gate
                )
                activation = float(np.clip(velocity_gate * window, 0.0, 1.0))
                remaining_control *= 1.0 - activation * float(fade)
            return self.b * float(np.clip(remaining_control, 0.0, 1.0))

        if self.inside_lattice_commitment(z):
            y_t = float(z[0])
            nearest = self.nearest_lattice_center(y_t)
            fade = (
                self.lattice_tail_brake_fade
                if y_t >= nearest
                else self.lattice_brake_fade
            )
            return self.b * (1.0 - fade)

        if self.inside_lattice_trap(z):
            return self.b * (1.0 - self.lattice_brake_fade)

        if self.kind != "sparse_deadend" and self.inside_funnel(z):
            return self.b * (1.0 - self.funnel_brake_fade)

        return self.b

    def funnel_push(self, z: np.ndarray) -> float:
        if not self.inside_funnel(z):
            return 0.0

        velocity = self.velocity(z)
        if velocity <= self.funnel_velocity_threshold:
            return 0.0

        y_t = float(z[0])
        distance = abs(y_t - self.funnel_center)
        shape = max(0.0, 1.0 - distance / max(self.funnel_halfwidth, 1e-8))

        return self.funnel_runaway_strength * shape * (
            velocity - self.funnel_velocity_threshold
        ) ** 2

    def lattice_push(self, z: np.ndarray) -> float:
        if self.smooth_switches and self.kind in {"funnel_lattice", "sparse_deadend"}:
            y_t = float(z[0])
            velocity = self.velocity(z)
            kappa = max(float(self.switch_sharpness), 1e-6)
            positive_velocity = float(
                np.logaddexp(
                    0.0,
                    kappa * (velocity - self.lattice_velocity_threshold),
                )
                / kappa
            )
            total_push = 0.0
            for trap_idx, center in enumerate(self.lattice_centers()):
                center = float(center)
                left_edge = center - self.lattice_halfwidth
                right_edge = center + self.lattice_tailwidth
                window = self._smooth_window(y_t, left_edge, right_edge)
                tail_gate = self._sigmoid(kappa * (y_t - center))
                tail_multiplier = 1.0 + (
                    self.lattice_tail_runaway_multiplier - 1.0
                ) * tail_gate
                trap_scale = 1.0 + 0.10 * int(trap_idx)
                total_push += (
                    self.lattice_runaway_strength
                    * trap_scale
                    * tail_multiplier
                    * window
                    * positive_velocity**2
                )
            return float(total_push)

        if not self.inside_lattice_commitment(z):
            return 0.0

        velocity = self.velocity(z)
        if velocity <= self.lattice_velocity_threshold:
            return 0.0

        y_t = float(z[0])
        nearest = self.nearest_lattice_center(y_t)

        if not np.isfinite(nearest):
            return 0.0

        shape = self.lattice_commitment_shape(y_t)
        trap_idx = int(
            round((nearest - self.lattice_start) / max(self.lattice_period, 1e-8))
        )
        trap_scale = 1.0 + 0.10 * trap_idx
        tail_multiplier = (
            self.lattice_tail_runaway_multiplier if y_t >= nearest else 1.0
        )

        return (
            self.lattice_runaway_strength
            * trap_scale
            * tail_multiplier
            * shape
            * (velocity - self.lattice_velocity_threshold) ** 2
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = float(value)
        if value >= 0.0:
            return float(1.0 / (1.0 + np.exp(-value)))
        exp_value = float(np.exp(value))
        return float(exp_value / (1.0 + exp_value))

    def _smooth_window(self, value: float, left: float, right: float) -> float:
        kappa = max(float(self.switch_sharpness), 1e-6)
        return float(
            self._sigmoid(kappa * (value - left))
            * self._sigmoid(kappa * (right - value))
        )

    def diagnostic_deadend_push(self, z: np.ndarray) -> float:
        if self.kind != "diagnostic_deadend":
            return 0.0

        velocity = self.velocity(z)
        if velocity <= 0.0:
            return 0.0

        y_t = float(z[0])
        normalized_height = max(0.0, min(1.0, y_t / max(self.y_max, 1e-8)))

        return 0.08 * normalized_height * velocity**2

    def pointwise_safety_value(self, z: np.ndarray) -> float:

        y_t = float(z[0])
        upper_margin = self.y_max - y_t
        lower_margin = y_t + self.y_max
        return float(min(upper_margin, lower_margin))

    def stopping_distance_value(self, z: np.ndarray) -> float:

        y_t = float(z[0])
        velocity = self.velocity(z)
        max_braking = max(abs(self.b) * self.u_max, 1e-8)
        stopping_distance = max(velocity, 0.0) ** 2 / (2.0 * max_braking)
        upper_margin = self.y_max - y_t - stopping_distance
        lower_margin = y_t + self.y_max
        return float(min(upper_margin, lower_margin))

    def safety_value(self, z: np.ndarray) -> float:

        return self.pointwise_safety_value(z)

    def in_failure_mode(self, z: np.ndarray) -> bool:
        return self.pointwise_safety_value(z) < 0.0

    def failure_observation(self, z: np.ndarray, noise: bool = True) -> float:
        eps = self.rng.normal(0.0, self.sigma_eps) if noise else 0.0
        sign = 1.0 if float(z[0]) >= 0.0 else -1.0
        return sign * self.fail_y + eps

    def step(self, z: np.ndarray, noise: bool = True) -> float:
        if self.failure_absorbing and self.in_failure_mode(z):
            return self.failure_observation(z, noise=noise)

        eps = self.rng.normal(0.0, self.sigma_eps) if noise else 0.0
        return self.transition_mean(z) + eps

    def shift(self, z: np.ndarray, y_next: float, u_next: float) -> np.ndarray:
        return np.array([y_next, float(z[0]), u_next], dtype=float)

    def observe_safety(self, z: np.ndarray, noise: bool = True) -> float:
        eps = self.rng.normal(0.0, self.sigma_g) if noise else 0.0
        return self.safety_value(z) + eps

    def is_safe(self, z: np.ndarray) -> bool:
        return self.safety_value(z) >= 0.0

    def state_is_safe(self, y_t: float, y_tm1: float) -> bool:
        probe = self.make_regressor(y_t, y_tm1, 0.0)
        return self.is_safe(probe)

    def goal_reached(self, y_t: float) -> bool:
        if self.kind != "goal_funnel":
            return False
        return self.goal_y_min <= float(y_t) <= self.goal_y_max
