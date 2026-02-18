import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ValidationReport:
    passed: bool
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    failures: List[str] = field(default_factory=list)

class ModelValidator:
    """
    Validates a model against performance and latency thresholds.
    """
    def __init__(
        self,
        model: Any,
        data_loader: Any = None,
        predict_fn: Optional[Callable[[Any, Any], Any]] = None,
    ):
        """
        Args:
            model: The model object (e.g., torch.nn.Module)
            data_loader: Iterable of (inputs, targets) or just inputs
            predict_fn: Optional function to call model(input). Defaults to model(input).
        """
        self.model = model
        self.data_loader = data_loader
        self.predict_fn = predict_fn or (lambda m, x: m(x))

    def validate(
        self,
        min_metric: Optional[Dict[str, float]] = None,
        max_latency_ms: Optional[float] = None,
        smoke_test: bool = True,
        input_sample: Any = None
    ) -> ValidationReport:
        """
        Run validation checks.
        
        Args:
            min_metric: Dict of metric_name -> min_value (requires metric_fn in data_loader loop if implemented, 
                        or user provides pre-calculated metrics? 
                        Actually, best is to let user provide a metric function or just run a smoke test and latency check here.
                        For accuracy, user usually calculates it separately. 
                        Let's support a simple metric function if provided.)
            max_latency_ms: Maximum average latency per item (or batch) in milliseconds.
            smoke_test: Run a single forward pass to ensure no crashes.
            input_sample: Specific input for smoke test/latency check if data_loader is None.
        """
        checks = {}
        metrics = {}
        failures = []
        
        # 1. Smoke Test
        if smoke_test:
            try:
                self._run_smoke_test(input_sample)
                checks["smoke_test"] = True
            except Exception as e:
                checks["smoke_test"] = False
                failures.append(f"Smoke test failed: {e}")
                
        # 2. Latency Check
        if max_latency_ms is not None:
             avg_latency = self._measure_latency(input_sample)
             metrics["latency_ms"] = avg_latency
             if avg_latency <= max_latency_ms:
                 checks["latency"] = True
             else:
                 checks["latency"] = False
                 failures.append(f"Latency {avg_latency:.2f}ms > limit {max_latency_ms}ms")
        
        passed = all(checks.values())
        return ValidationReport(passed=passed, checks=checks, metrics=metrics, failures=failures)

    def _run_smoke_test(self, input_sample):
        if input_sample is not None:
            self.predict_fn(self.model, input_sample)
        elif self.data_loader:
             batch = next(iter(self.data_loader))
             # Handle batch tuple vs single input
             if isinstance(batch, (tuple, list)):
                 x = batch[0]
             else:
                 x = batch
             self.predict_fn(self.model, x)
        else:
            raise ValueError("No input data provided for smoke test")

    def _measure_latency(self, input_sample, num_warmup=5, num_runs=20) -> float:
        # Get input
        if input_sample is not None:
            x = input_sample
        elif self.data_loader:
            batch = next(iter(self.data_loader))
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
        else:
            return 0.0

        # Warmup
        for _ in range(num_warmup):
            self.predict_fn(self.model, x)
            
        # Measure
        start_time = time.time()
        for _ in range(num_runs):
            self.predict_fn(self.model, x)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / num_runs) * 1000
        return avg_time_ms
