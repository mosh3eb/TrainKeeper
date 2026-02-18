import logging
import random
import hashlib
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from .monitoring import ModelMonitor

logger = logging.getLogger(__name__)

@dataclass
class ExperimentVariant:
    name: str
    model: Any
    traffic_percentage: float = 0.5
    monitor: Optional[ModelMonitor] = None
    
    def __post_init__(self):
        if self.monitor is None:
            self.monitor = ModelMonitor()

class ExperimentManager:
    """
    Manages A/B tests between multiple model variants.
    """
    def __init__(self, variants: List[ExperimentVariant]):
        self.variants = variants
        self._validate_config()
        
    def _validate_config(self):
        total_pct = sum(v.traffic_percentage for v in self.variants)
        if not 0.99 <= total_pct <= 1.01:
             logger.warning(f"Total traffic percentage is {total_pct}, expected 1.0. Normalizing...")
             for v in self.variants:
                 v.traffic_percentage /= total_pct

    def route(self, user_id: str) -> ExperimentVariant:
        """
        Deterministically route a user ID to a variant.
        """
        # Hash user ID to get a consistent number between 0 and 1
        hash_val = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        normalized_hash = (hash_val % 10000) / 10000.0
        
        cumulative_pct = 0.0
        for variant in self.variants:
            cumulative_pct += variant.traffic_percentage
            if normalized_hash < cumulative_pct:
                return variant
        
        return self.variants[-1] # Fallback

    def predict(self, user_id: str, input_data: Any) -> Any:
        """
        Route request, predict, and log to the variant's monitor.
        """
        variant = self.route(user_id)
        
        # In a real app, you might measure latency here too
        # For now, just a simple prediction wrapper
        try:
            prediction = variant.model(input_data)
            variant.monitor.log_prediction(prediction)
            # Log successful request
            variant.monitor.log_request(latency_ms=0, error=False)
            return prediction
        except Exception:
            variant.monitor.log_request(latency_ms=0, error=True)
            raise

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comparison results for all variants.
        """
        results = {}
        for v in self.variants:
            health = v.monitor.get_health()
            results[v.name] = {
                "traffic_split": v.traffic_percentage,
                "latency_p95": health.latency_p95,
                "error_rate": health.error_rate,
                "request_count": health.request_count
            }
        return results
