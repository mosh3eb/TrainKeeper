import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
import logging

# Optional dependencies
try:
    import requests
except ImportError:
    requests = None

stats = None
try:
    from scipy import stats
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class DriftReport:
    is_drifted: bool
    drift_score: float
    metric: str
    feature_name: str
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)

class DriftAlert:
    def send(self, report: DriftReport):
        raise NotImplementedError

class EmailAlert(DriftAlert):
    def __init__(self, smtp_server: str, smtp_port: int, from_addr: str, to_addrs: List[str], username: Optional[str] = None, password: Optional[str] = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.username = username
        self.password = password

    def send(self, report: DriftReport):
        if not report.is_drifted:
            return
            
        subject = f"🚨 Data Drift Detected: {report.feature_name}"
        body = f"""
        Data Drift Alert
        ----------------
        Feature: {report.feature_name}
        Metric: {report.metric}
        Score: {report.drift_score:.4f}
        Threshold: {report.threshold}
        
        Details:
        {json.dumps(report.details, indent=2)}
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.from_addr
        msg['To'] = ", ".join(self.to_addrs)
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.username is not None and self.password is not None:
                    server.starttls()
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            logger.info(f"Sent drift alert email to {self.to_addrs}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class WebhookAlert(DriftAlert):
    def __init__(self, url: str):
        self.url = url

    def send(self, report: DriftReport):
        if not report.is_drifted:
            return
        
        if requests is None:
            logger.error("requests library not found. Install with `pip install requests`")
            return

        payload = {
            "text": f"🚨 *Data Drift Detected* for feature `{report.feature_name}`\n"
                    f"• Metric: {report.metric}\n"
                    f"• Score: {report.drift_score:.4f}\n"
                    f"• Threshold: {report.threshold}"
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            logger.info(f"Sent drift alert to webhook: {self.url}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

class DriftDetector:
    def __init__(self, reference_data: Optional[np.ndarray] = None, alerts: Optional[List[DriftAlert]] = None):
        self.reference_data = reference_data
        self.alerts = alerts or []

    def set_reference(self, data: np.ndarray):
        self.reference_data = data

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        try:
            # Use percentiles from expected to define bins
            bins = np.percentile(expected, breakpoints)
            # Handle duplicate bin edges
            bins = np.unique(bins)
            if len(bins) < 2:
                # Fallback to linspace if percentiles are identical
                bins = np.linspace(min(min(expected), min(actual)), max(max(expected), max(actual)), buckets + 1)

            expected_percents = np.histogram(expected, bins)[0] / len(expected)
            actual_percents = np.histogram(actual, bins)[0] / len(actual)

            # Avoid division by zero
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
            return float(psi)
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def calculate_ks(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic (max distance between CDFs)"""
        if stats is not None:
            try:
                statistic, _ = stats.ks_2samp(expected, actual)
                return float(statistic)
            except Exception:
                pass
        
        # Fallback simple implementation if scipy missing or failed
        data1 = np.sort(expected)
        data2 = np.sort(actual)
        n1 = data1.shape[0]
        n2 = data2.shape[0]
        
        if n1 == 0 or n2 == 0: return 0.0
        
        data_all = np.concatenate([data1, data2])
        cdf1 = np.searchsorted(data1, data_all, side='right') / n1
        cdf2 = np.searchsorted(data2, data_all, side='right') / n2
        
        return float(np.max(np.abs(cdf1 - cdf2)))

    def check_drift(self, current_data: np.ndarray, feature_name: str, method: str = "psi", threshold: float = 0.1) -> DriftReport:
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        score = 0.0
        if method == "psi":
            score = self.calculate_psi(self.reference_data, current_data)
        elif method == "ks":
            score = self.calculate_ks(self.reference_data, current_data)
        else:
            raise ValueError(f"Unknown drift method: {method}")

        is_drifted = score > threshold
        
        report = DriftReport(
            is_drifted=is_drifted,
            drift_score=score,
            metric=method.upper(),
            feature_name=feature_name,
            threshold=threshold
        )

        if is_drifted:
            for alert in self.alerts:
                alert.send(report)

        return report
