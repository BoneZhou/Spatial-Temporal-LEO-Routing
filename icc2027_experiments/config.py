import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict


# Fixed adapted-baseline learning parameters used in the paper experiment.
# Architecture and reward adaptations are documented in docs/BASELINES.md.
BASELINE_LEARNING = {
    "graphpr_adapted": dict(learning_rate=0.001, replay_capacity=30000,
                            batch_size=2048, epsilon_start=0.90,
                            epsilon_end=0.01, gamma=0.90),
    "fdr_marl_adapted": dict(learning_rate=0.001, replay_capacity=30000,
                             batch_size=128, epsilon_start=0.95,
                             epsilon_end=0.001, gamma=0.90),
    "dqn_ir_fixed_link": dict(learning_rate=0.01, replay_capacity=200,
                              batch_size=32, epsilon_start=0.60,
                              epsilon_end=0.001, gamma=0.90),
}


@dataclass(frozen=True)
class SimulationConfig:
    num_satellites: int = 45
    orbital_planes: int = 3
    satellites_per_plane: int = 15
    phasing_factor: int = 1
    orbital_altitude_km: float = 570.0
    inclination_deg: float = 70.0
    earth_radius_km: float = 6371.0
    earth_gravitational_parameter_km3_s2: float = 398600.4418
    topology_update_interval_steps: int = 1
    slot_duration_s: float = 0.020
    packet_size_bytes: int = 1500
    max_queue_packets: int = 100
    max_hops: int = 30
    isl_capacity_mbps: float = 300.0
    service_packets_per_slot: int = 36
    processing_delay_ms: float = 0.10
    nhpp_period_steps: int = 30
    inter_plane_latitude_limit_deg: float = 60.0
    evaluation_warmup_slots: int = 30
    evaluation_measurement_slots: int = 100

    def __post_init__(self) -> None:
        if self.orbital_planes * self.satellites_per_plane != self.num_satellites:
            raise ValueError("orbital_planes * satellites_per_plane must equal num_satellites")
        if self.topology_update_interval_steps < 1:
            raise ValueError("topology_update_interval_steps must be at least one")

    @property
    def orbital_period_s(self) -> float:
        orbital_radius_km = self.earth_radius_km + self.orbital_altitude_km
        return 2.0 * math.pi * math.sqrt(
            orbital_radius_km**3 / self.earth_gravitational_parameter_km3_s2
        )

    @property
    def orbital_period_slots(self) -> float:
        return self.orbital_period_s / self.slot_duration_s

    @property
    def training_warmup_slots(self) -> int:
        """Keep focal training and evaluation in the same pre-traffic regime."""
        return self.evaluation_warmup_slots



@dataclass(frozen=True)
class LearningConfig:
    hidden_dim: int = 64
    lstm_hidden_dim: int = 128
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    replay_buffer_size: int = 100_000
    batch_size: int = 128
    target_update_steps: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    training_episodes: int = 1300
    evaluation_episodes: int = 200


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 11
    offered_load_mbps: float = 180.0
    observation_window: int = 5
    simulation: SimulationConfig = SimulationConfig()
    learning: LearningConfig = LearningConfig()
    mode: str = "full"

    @property
    def model_seed(self) -> int:
        return self.seed

    @property
    def traffic_seed(self) -> int:
        return self.seed * 100_000 + 17

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["simulation"]["orbital_period_s"] = self.simulation.orbital_period_s
        data["simulation"]["orbital_period_slots"] = self.simulation.orbital_period_slots
        data["simulation"]["topology_update_interval_s"] = (
            self.simulation.topology_update_interval_steps
            * self.simulation.slot_duration_s
        )
        data["simulation"]["training_warmup_slots"] = (
            self.simulation.training_warmup_slots
        )
        data["model_seed"] = self.model_seed
        data["traffic_seed"] = self.traffic_seed
        return data


def build_config(seed: int, offered_load_mbps: float, observation_window: int, mode: str) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seed=seed,
        offered_load_mbps=offered_load_mbps,
        observation_window=observation_window,
        mode=mode,
    )
    if mode == "smoke":
        smoke_learning = replace(
            cfg.learning,
            hidden_dim=24,
            lstm_hidden_dim=24,
            replay_buffer_size=256,
            batch_size=2,
            target_update_steps=4,
            training_episodes=2,
            evaluation_episodes=2,
        )
        smoke_simulation = replace(
            cfg.simulation,
            evaluation_warmup_slots=1,
            evaluation_measurement_slots=10,
        )
        cfg = replace(cfg, learning=smoke_learning, simulation=smoke_simulation)
    elif mode == "pilot":
        pilot_learning = replace(
            cfg.learning,
            hidden_dim=32,
            lstm_hidden_dim=64,
            replay_buffer_size=512,
            batch_size=8,
            target_update_steps=20,
            training_episodes=5,
            evaluation_episodes=3,
        )
        pilot_simulation = replace(
            cfg.simulation,
            evaluation_warmup_slots=3,
            evaluation_measurement_slots=20,
        )
        cfg = replace(cfg, learning=pilot_learning, simulation=pilot_simulation)
    return cfg

