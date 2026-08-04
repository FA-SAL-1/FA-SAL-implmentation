from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple
import numpy as np
from synthetic_preset import get_synthetic_environment_preset

HORIZON=2

@dataclass
class GeneralConfig:
    rounds: int = 150
    trials: int = 10
    kernel: str = "se"
    eval_points: int = 100
    recovery_eval_interval: int = 10
    episodic_reset_period: int = 0
    evaluation_horizon: int = HORIZON

    methods: Tuple[str, ...] = ("fa_sal", "tebbe_abm", "salnx")



    parallel_workers: int = 4
    result_dirname: str = "result"



@dataclass
class InitialDataConfig:
    n_init: int = 150

    y_min: float = -0.8
    y_max: float = 0.8
    v_min: float = -0.12
    v_max: float = 0.18

    start_y_tm1: float = 0.05
    start_y_t: float = 0.18


@dataclass
class FASALConfig:
    horizon: int = HORIZON
    beam_width: int = 1
    continuation_policy: str = "uncertainty-max"
    continuation_policy_sweep: Tuple[str, ...] = ()
    epsilon_margin: float = 0
    delta_f: float = 0.05
    delta_g: float = 0.05
    state_grid_points: int = 31
    state_y_min: float = -2.0
    state_y_max: float = 6.0
    lx: float = 1.0
    ly: float = 1.0
    lf: float = -np.inf
    lf_quantile: float = 0.5
    lf_scale: float = 1.0
    lf_scale_sweep: Tuple[float, ...] = ()
    lf_cap: float = np.inf
    l_ell: float = 1
    l_ell_quantile: float = 0.9
    l_ell_scale: float = 1.0

    l_ell_scale_sweep: Tuple[float, ...] = (0.14,)


    buffer_weight: float = 0



@dataclass
class ControlBaselineConfig:
    horizon: int = HORIZON
    target: float = 4.45
    q_y: float = 1.0
    q_velocity: float = 0.25
    r_u: float = 0.05
    r_delta_u: float = 0.05
    terminal_weight: float = 2.0
    beam_width: int = 1


@dataclass
class NominalMPCBaselineConfig(ControlBaselineConfig):
    safety_margin: float = 0.0


@dataclass
class SALNXBaselineConfig:
    horizon: int = HORIZON
    alpha: float = 0.1
    alpha_sweep: Tuple[float, ...] = (0.2,)
    criterion: str = "logdet"
    joint_mc_samples: int = 64


@dataclass
class PointwiseBaselineConfig:
    name: str = "Pointwise"
    horizon: int = 4


@dataclass
class SafeExplorationBaselineConfig:
    horizon: int = 1
    alpha: float = 0.05


@dataclass
class FARandomBaselineConfig:
    name: str = "FA-SAL-Random"
    horizon: int = 4



@dataclass
class TebbeBaselineConfig:
    horizon: int = HORIZON
    alpha: float = 0.1
    alpha_sweep: Tuple[float, ...] = ()
    criterion: str = "trace"
    confidence_delta: float = 1e-3
    confidence_delta_sweep: Tuple[float, ...] = ()
    sample_start: int = 100
    sample_stages: int = 17
    endpoint_candidates: int = 11


@dataclass
class EnvironmentConfig:
    preset: str = "penalty_stress"

    def __post_init__(self) -> None:
        preset_values = get_synthetic_environment_preset(self.preset)

        for key, value in preset_values.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        data = {"preset": self.preset}
        data.update(get_synthetic_environment_preset(self.preset))
        return data



@dataclass
class SnapshotConfig:
    iters: Tuple[int, ...] = field(default_factory=lambda: (3, 13, 24))
    grid_size: int = 180
    u_slice: float = 0.0


@dataclass
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    general: GeneralConfig = field(default_factory=GeneralConfig)
    fa_sal: FASALConfig = field(default_factory=FASALConfig)
    fa_random: FARandomBaselineConfig = field(default_factory=FARandomBaselineConfig)
    pointwise: PointwiseBaselineConfig = field(default_factory=PointwiseBaselineConfig)
    safe_exploration: SafeExplorationBaselineConfig = field(default_factory=SafeExplorationBaselineConfig)
    nominal_mpc: NominalMPCBaselineConfig = field(default_factory=NominalMPCBaselineConfig)
    salnx: SALNXBaselineConfig = field(default_factory=SALNXBaselineConfig)
    tebbe: TebbeBaselineConfig = field(default_factory=TebbeBaselineConfig)
    initial_data: InitialDataConfig = field(default_factory=InitialDataConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["environment"] = self.environment.to_dict()
        return data

DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
