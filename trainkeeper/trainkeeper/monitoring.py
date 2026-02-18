import logging
import time
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
from .drift import DriftDetector, DriftReport

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    request_count: int = 0
    drift_status: Dict[str, bool] = field(default_factory=dict)

class ModelMonitor:
    """
    Monitors model performance in production:
    - Latency tracking
    - Error rate tracking
    - Prediction drift detection
    """
    def __init__(
        self, 
        drift_detector: Optional[DriftDetector] = None,
        window_size: int = 1000
    ):
        self.drift_detector = drift_detector
        self.window_size = window_size
        
        # Metrics window
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        
        # Data window for drift checking (accumulate batch)
        self.prediction_window = []
        self.prediction_window_size = 100 # Check drift every 100 predictions
        
    def log_request(self, latency_ms: float, error: bool = False):
        """Log a single request's latency and error status."""
        self.latencies.append(latency_ms)
        self.errors.append(1 if error else 0)

    def log_prediction(self, prediction: Any):
        """Log a prediction for drift detection."""
        if not self.drift_detector:
            return

        self.prediction_window.append(prediction)
        
        if len(self.prediction_window) >= self.prediction_window_size:
            self._check_prediction_drift()
            self.prediction_window = [] # Reset window

    def _check_prediction_drift(self):
        try:
            current_data = np.array(self.prediction_window)
            if self.drift_detector.reference_data is not None:
                # Assuming single output feature for now, or flattened
                report = self.drift_detector.check_drift(
                    current_data, 
                    feature_name="prediction_output",
                    threshold=0.1
                )
                if report.is_drifted:
                    logger.warning(f"Prediction drift detected! Score: {report.drift_score:.4f}")
        except Exception as e:
            logger.error(f"Failed to check prediction drift: {e}")

    def get_health(self) -> ServiceHealth:
        """Calculate current health metrics."""
        if not self.latencies:
            return ServiceHealth()
            
        latencies = np.array(self.latencies)
        errors = np.array(self.errors)
        
        return ServiceHealth(
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=np.mean(errors),
            request_count=len(latencies),
            drift_status={} # TODO: Populate with recent drift checks
        )
