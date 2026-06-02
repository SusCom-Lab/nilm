from __future__ import annotations

import os

from dataset_management.data_separation.data_separator import DataSeparator
from dataset_management.export.csv_exporter import CSVExporter
from dataset_management.pipeline.schemas import make_pipeline_summary
from dataset_management.repair.file_repairer import FileRepairer
from dataset_management.validation.file_validator import FileValidator
from dataset_management.validation.report_writer import ReportWriter


class PipelineRunner:
    def __init__(self):
        self.validator = FileValidator()
        self.report_writer = ReportWriter()
        self.repairer = FileRepairer()
        self.exporter = CSVExporter()

    def run(self, dataset_type: str, input_path: str, workspace_dir: str) -> dict:
        dataset_key = dataset_type.upper()
        separated_root = os.path.join(workspace_dir, "separated")
        repaired_root = os.path.join(workspace_dir, "repaired")
        exported_root = os.path.join(workspace_dir, "exported_csv")
        reports_root = os.path.join(workspace_dir, "reports")

        separator = DataSeparator(
            file_path=input_path,
            save_path=separated_root,
            dataset_type=dataset_key,
        )
        separator.process_data()
        separated_dir = separator.output_dir

        raw_reports = self.validator.validate_directory(separated_dir)
        raw_report_json = os.path.join(reports_root, "validation_raw.json")
        raw_report_csv = os.path.join(reports_root, "validation_raw_summary.csv")
        self.report_writer.write_json(raw_reports, raw_report_json)
        self.report_writer.write_summary_csv(raw_reports, raw_report_csv)

        repaired_dir = os.path.join(repaired_root, os.path.basename(separated_dir))
        self.repairer.repair_directory(separated_dir, dataset_key, repaired_dir)

        repaired_reports = self.validator.validate_directory(repaired_dir)
        repaired_report_json = os.path.join(reports_root, "validation_repaired.json")
        repaired_report_csv = os.path.join(reports_root, "validation_repaired_summary.csv")
        self.report_writer.write_json(repaired_reports, repaired_report_json)
        self.report_writer.write_summary_csv(repaired_reports, repaired_report_csv)

        export_dir = os.path.join(exported_root, f"{dataset_key}_dataset")
        export_count = 0
        for name in sorted(os.listdir(repaired_dir)):
            house_dir = os.path.join(repaired_dir, name)
            if not os.path.isdir(house_dir):
                continue
            export_count += len(self.exporter.export_house_directory(house_dir, dataset_key, export_dir))

        return make_pipeline_summary(
            separated_dir=separated_dir,
            repaired_dir=repaired_dir,
            exported_dir=export_dir,
            validation_raw_report=raw_report_json,
            validation_repaired_report=repaired_report_json,
            export_count=export_count,
        )
