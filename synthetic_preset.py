from typing import Any, Dict


SYNTHETIC_ENVIRONMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {
        "kind": "sparse_deadend",
        "y_max": 5.0,

        "dynamics_noise_std": 0.002,
        "safety_noise_std": 0.001,

        "runaway_center": 3.0,
        "runaway_halfwidth": 0.8,
        "runaway_strength": 0.0,
        "runaway_velocity_threshold": 0.25,

        "funnel_center": 3.4,
        "funnel_halfwidth": 0.7,
        "funnel_runaway_strength": 0.0,
        "funnel_velocity_threshold": 0.15,
        "funnel_brake_fade": 0.0,

        "lattice_start": 3.2,
        "lattice_period": 1.6,
        "lattice_count": 5,
        "lattice_halfwidth": 0.15,
        "lattice_tailwidth": 0.30,
        "lattice_runaway_strength": 0.12,
        "lattice_tail_runaway_multiplier": 1.0,
        "lattice_velocity_threshold": 0.35,

        "lattice_brake_fade": 0.35,
        "lattice_tail_brake_fade": 0.55,

        "goal_y_min": 4.2,
        "goal_y_max": 4.8,
    },

    "hard": {
    "kind": "sparse_deadend",

    "y_max": 4.35,

    "dynamics_noise_std": 0.015,
    "safety_noise_std": 0.008,

    "runaway_center": 3.0,
    "runaway_halfwidth": 0.8,
    "runaway_strength": 0.0,
    "runaway_velocity_threshold": 0.25,

    "funnel_center": 3.4,
    "funnel_halfwidth": 0.7,
    "funnel_runaway_strength": 0.0,
    "funnel_velocity_threshold": 0.15,
    "funnel_brake_fade": 0.0,

    "lattice_start": 2.45,
    "lattice_period": 1.0,
    "lattice_count": 8,
    "lattice_halfwidth": 0.28,
    "lattice_tailwidth": 0.52,

    "lattice_runaway_strength": 0.34,
    "lattice_tail_runaway_multiplier": 1.8,
    "lattice_velocity_threshold": 0.18,

    "lattice_brake_fade": 0.72,
    "lattice_tail_brake_fade": 0.90,

    "goal_y_min": 3.85,
    "goal_y_max": 4.25,
},

"stress": {
    "kind": "sparse_deadend",

    "y_max": 4.15,

    "dynamics_noise_std": 0.025,
    "safety_noise_std": 0.012,

    "runaway_center": 3.0,
    "runaway_halfwidth": 0.8,
    "runaway_strength": 0.0,
    "runaway_velocity_threshold": 0.25,

    "funnel_center": 3.4,
    "funnel_halfwidth": 0.7,
    "funnel_runaway_strength": 0.0,
    "funnel_velocity_threshold": 0.15,
    "funnel_brake_fade": 0.0,

    "lattice_start": 2.15,
    "lattice_period": 0.85,
    "lattice_count": 10,
    "lattice_halfwidth": 0.34,
    "lattice_tailwidth": 0.62,

    "lattice_runaway_strength": 0.46,
    "lattice_tail_runaway_multiplier": 2.2,
    "lattice_velocity_threshold": 0.14,

    "lattice_brake_fade": 0.84,
    "lattice_tail_brake_fade": 0.96,

    "goal_y_min": 3.70,
    "goal_y_max": 4.05,
},


    "penalty_stress": {
    "kind": "sparse_deadend",

    "y_max": 3.95,

    "dynamics_noise_std": 0.018,
    "safety_noise_std": 0.008,

    "runaway_center": 3.0,
    "runaway_halfwidth": 0.8,
    "runaway_strength": 0.0,
    "runaway_velocity_threshold": 0.25,

    "funnel_center": 3.4,
    "funnel_halfwidth": 0.7,
    "funnel_runaway_strength": 0.0,
    "funnel_velocity_threshold": 0.15,
    "funnel_brake_fade": 0.0,

    "lattice_start": 1.85,
    "lattice_period": 0.72,
    "lattice_count": 12,

    "lattice_halfwidth": 0.38,
    "lattice_tailwidth": 0.85,

    "lattice_runaway_strength": 0.72,
    "lattice_tail_runaway_multiplier": 3.0,
    "lattice_velocity_threshold": 0.08,

    "lattice_brake_fade": 0.92,
    "lattice_tail_brake_fade": 0.99,

    "goal_y_min": 3.50,
    "goal_y_max": 3.85,

    "fail_y": 7.5,
    "failure_absorbing": True,
    },

    "smooth_penalty_stress": {
        "kind": "sparse_deadend",
        "y_max": 3.95,
        "dynamics_noise_std": 0.018,
        "safety_noise_std": 0.008,
        "runaway_center": 3.0,
        "runaway_halfwidth": 0.8,
        "runaway_strength": 0.0,
        "runaway_velocity_threshold": 0.25,
        "funnel_center": 3.4,
        "funnel_halfwidth": 0.7,
        "funnel_runaway_strength": 0.0,
        "funnel_velocity_threshold": 0.15,
        "funnel_brake_fade": 0.0,
        "lattice_start": 1.85,
        "lattice_period": 0.72,
        "lattice_count": 12,
        "lattice_halfwidth": 0.38,
        "lattice_tailwidth": 0.85,
        "lattice_runaway_strength": 0.72,
        "lattice_tail_runaway_multiplier": 3.0,
        "lattice_velocity_threshold": 0.08,
        "lattice_brake_fade": 0.92,
        "lattice_tail_brake_fade": 0.99,
        "goal_y_min": 3.50,
        "goal_y_max": 3.85,
        "fail_y": 7.5,
        "failure_absorbing": True,
        "smooth_switches": True,
        "switch_sharpness": 8.0,
    },


}








def get_synthetic_environment_preset(name: str) -> Dict[str, Any]:
    if name not in SYNTHETIC_ENVIRONMENT_PRESETS:
        available = ", ".join(sorted(SYNTHETIC_ENVIRONMENT_PRESETS.keys()))
        raise ValueError(
            f"Unknown synthetic environment preset: {name}. "
            f"Available presets: {available}"
        )

    return dict(SYNTHETIC_ENVIRONMENT_PRESETS[name])
