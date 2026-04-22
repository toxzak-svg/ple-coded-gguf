from dataclasses import dataclass


@dataclass
class PLEDominanceConfig:
    variance_threshold: float = 0.5
    channel_threshold: float = 0.6
    calibration_seq_len: int = 512
    calibration_batch_size: int = 4
    num_calibration_samples: int = 256


PROFILING_DEFAULTS = PLEDominanceConfig()
