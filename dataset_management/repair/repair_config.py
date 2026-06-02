from __future__ import annotations


DATASET_REPAIR_CONFIG = {
    "UKDALE": {
        "resample_interval": "6S",
        "max_fill_duration": "60S",
        "min_power_threshold": 5.0,
        "max_power_clip": 20000.0,
    },
    "REDD": {
        "resample_interval": "6S",
        "max_fill_duration": "60S",
        "min_power_threshold": 5.0,
        "max_power_clip": 20000.0,
    },
    "REFIT": {
        "resample_interval": "6S",
        "max_fill_duration": "60S",
        "min_power_threshold": 5.0,
        "max_power_clip": 20000.0,
    },
    "ECO": {
        "resample_interval": "6S",
        "max_fill_duration": "60S",
        "min_power_threshold": 5.0,
        "max_power_clip": 20000.0,
    },
    "STANDARD_H5": {
        "resample_interval": "1ms",
        "max_fill_duration": "100ms",
        "min_power_threshold": 5.0,
        "max_power_clip": 20000.0,
    },
}
