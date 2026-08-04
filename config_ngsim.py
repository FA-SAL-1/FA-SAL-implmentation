from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_NGSIM_DATA_PATH = PROJECT_ROOT / "Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv"


HORIZON=4

@dataclass
class NGSIMDataConfig:
    data_path: Path = DEFAULT_NGSIM_DATA_PATH
    max_transitions: int = 5000
    distance_unit_scale: float = 0.3048


@dataclass
class NGSIMGeneralConfig:
    methods: Tuple[str, ...] = ("salnx", "tebbe_abm")
    rounds: int = 150
    trials: int = 10
    seed: int = 0
    save_dir: Path = PROJECT_ROOT / "result_ngsim"
    quiet: bool = False
    parallel_workers: int = 0


@dataclass
class NGSIMEnvironmentConfig:
    horizon: int = HORIZON
    action_values: Tuple[float, ...] = (-4.0, -2.0, 0.0, 1.0, 2.0)
    min_headway: float = 8.0
    safety_headway_offset: float = 0.5
    dt: float = 0.1
    frame_stride: int = 5
    n_init: int = 80
    n_eval: int = 100
    reset_context_each_round: bool = True
    safe_start_headway_max: float = 14.5
    safe_start_rel_speed_max: float = 0.0
    safe_start_min_count: int = 20

    rmse_eval_headway_max: float = 14.5
    rmse_eval_rel_speed_max: float = 0.1
    rmse_eval_min_count: int = 20

    residual_neighbors: int = 16
    collision_absorbing: bool = True
    collision_headway: float = 0.0
    collision_safety_penalty: float = -10.0

    dynamics_noise_std: float = 0.15
    safety_noise_std: float = 0.02


@dataclass
class NGSIMLearningConfig:
    kernel: str = "se"
    length_scale: float = 1.0


@dataclass
class NGSIMFASALConfig:
    horizon: int = HORIZON
    beam_width: int = 1
    delta_f: float = 0.1
    delta_g: float = 0.1
    l_ell: float = 1.0
    l_ell_quantile: float = 0.9
    l_ell_quantile_sweep: Tuple[float, ...] = (0.9,)
    l_ell_scale: float = 0.01

    l_ell_scale_sweep: Tuple[float, ...] = (1e-7,1e-6,1e-5, 1e-4, 1e-3, 1e-2)

    uncertainty_criterion: str = "logdet"


@dataclass
class NGSIMFARandomConfig:
    horizon: int = HORIZON
    l_ell: float = 1.0
    l_ell_quantile: float = 0.9
    l_ell_quantile_sweep: Tuple[float, ...] = ()
    l_ell_scale: float = 0.01
    l_ell_scale_sweep: Tuple[float, ...] = (0.01,)


@dataclass
class NGSIMSALNXConfig:
    horizon: int = HORIZON
    alpha: float = 0.2
    alpha_sweep: Tuple[float, ...] = (0.2,)
    mc_samples: int = 128
    uncertainty_criterion: str = "logdet"


@dataclass
class NGSIMTebbeConfig:
    horizon: int = HORIZON
    alpha: float = 0.2
    confidence_delta: float = 1e-2
    sample_start: int = 32
    sample_stages: int = 6
    uncertainty_criterion: str = "logdet"


@dataclass
class NGSIMConfig:
    data: NGSIMDataConfig = field(default_factory=NGSIMDataConfig)
    general: NGSIMGeneralConfig = field(default_factory=NGSIMGeneralConfig)
    environment: NGSIMEnvironmentConfig = field(default_factory=NGSIMEnvironmentConfig)
    learning: NGSIMLearningConfig = field(default_factory=NGSIMLearningConfig)
    fa_sal: NGSIMFASALConfig = field(default_factory=NGSIMFASALConfig)
    fa_random: NGSIMFARandomConfig = field(default_factory=NGSIMFARandomConfig)
    salnx: NGSIMSALNXConfig = field(default_factory=NGSIMSALNXConfig)
    tebbe: NGSIMTebbeConfig = field(default_factory=NGSIMTebbeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


NGSIM_CONFIG = NGSIMConfig()
