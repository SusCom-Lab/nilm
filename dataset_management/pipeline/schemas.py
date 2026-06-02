from __future__ import annotations


def make_pipeline_summary(
    *,
    separated_dir: str,
    repaired_dir: str,
    exported_dir: str,
    validation_raw_report: str,
    validation_repaired_report: str,
    export_count: int,
) -> dict:
    return {
        "separated_dir": separated_dir,
        "repaired_dir": repaired_dir,
        "exported_dir": exported_dir,
        "validation_raw_report": validation_raw_report,
        "validation_repaired_report": validation_repaired_report,
        "export_count": export_count,
    }
