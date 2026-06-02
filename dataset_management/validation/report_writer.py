from __future__ import annotations

import json
import os
from typing import Iterable

import pandas as pd


class ReportWriter:
    def write_json(self, reports: Iterable[dict], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(list(reports), handle, indent=2, ensure_ascii=False)

    def write_summary_csv(self, reports: Iterable[dict], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rows = []
        for report in reports:
            rows.append(
                {
                    "file": report.get("file"),
                    "house_id": report.get("house_id"),
                    "signal_name": report.get("signal_name"),
                    "status": report.get("status"),
                    "issue_codes": ",".join(issue.get("code", "") for issue in report.get("issues", [])),
                }
            )
        pd.DataFrame(rows).to_csv(output_path, index=False)
