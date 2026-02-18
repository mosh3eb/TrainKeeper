import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from .validation import ValidationReport
from .drift import DriftReport

logger = logging.getLogger(__name__)

@dataclass
class ExperimentReport:
    experiment_id: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    validation_report: Optional[Dict[str, Any]] = None
    drift_report: Optional[Dict[str, Any]] = None
    artifacts: List[str] = None
    
    def to_markdown(self) -> str:
        md = [f"# Experiment Report: {self.experiment_id}", ""]
        
        md.append("## Configuration")
        md.append("```json")
        md.append(json.dumps(self.config, indent=2, default=str))
        md.append("```")
        md.append("")
        
        md.append("## Metrics")
        for k, v in self.metrics.items():
            md.append(f"- **{k}**: {v}")
        md.append("")
        
        if self.validation_report:
            md.append("## Model Validation")
            passed = self.validation_report.get("passed", False)
            icon = "✅" if passed else "❌"
            md.append(f"**Status**: {icon} {'Passed' if passed else 'Failed'}")
            
            failures = self.validation_report.get("failures", [])
            if failures:
                md.append("### Failures")
                for f in failures:
                    md.append(f"- {f}")
            md.append("")

        if self.drift_report:
            md.append("## Drift Detection")
            drifted = self.drift_report.get("is_drifted", False)
            icon = "⚠️" if drifted else "✅"
            md.append(f"**Status**: {icon} {'Drift Detected' if drifted else 'No Drift'}")
            md.append(f"- Score: {self.drift_report.get('drift_score', 0.0):.4f}")
            md.append("")
            
        if self.artifacts:
            md.append("## Artifacts")
            for a in self.artifacts:
                md.append(f"- {a}")
        
        return "\n".join(md)

class ReportGenerator:
    """
    Generates comprehensive reports for experiments.
    """
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def generate(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        validation_report: Optional[ValidationReport] = None,
        drift_report: Optional[DriftReport] = None,
        artifacts: List[str] = None
    ) -> Path:
        """
        Generate a report and save it to the run directory.
        """
        # Convert dataclasses to dicts for serialization
        val_dict = asdict(validation_report) if validation_report else None
        drift_dict = asdict(drift_report) if drift_report else None
        
        report = ExperimentReport(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics,
            validation_report=val_dict,
            drift_report=drift_dict,
            artifacts=artifacts or []
        )
        
        # Save JSON
        json_path = self.run_dir / "report.json"
        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
            
        # Save Markdown
        md_path = self.run_dir / "report.md"
        with open(md_path, "w") as f:
            f.write(report.to_markdown())
            
        logger.info(f"Generated experiment report at {md_path}")
        return md_path
