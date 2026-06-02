from __future__ import annotations

import re


FILENAME_PATTERN = re.compile(r"^(?P<signal>[a-z0-9_]+)_H(?P<house>\d+)\.h5$", re.IGNORECASE)

DEFAULT_VALIDATION_THRESHOLDS = {
    "max_nan_ratio_warn": 0.05,
    "max_negative_ratio_warn": 0.001,
    "max_gap_multiplier_warn": 10.0,
    "max_abs_power_warn": 20000.0,
}

ISSUE_INFO = "INFO"
ISSUE_WARN = "WARN"
ISSUE_ERROR = "ERROR"
