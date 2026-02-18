import pickle
import logging
from pathlib import Path
from typing import Any, List, Optional, Union, Callable

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    A wrapper for data preprocessing steps to ensure reproducibility.
    Supports sklearn-like transformers and custom functions.
    """
    def __init__(self, steps: List[Any] = None):
        """
        Args:
            steps: List of transformers or (name, transformer) tuples.
                   Transformers must implement fit/transform.
        """
        self.steps = steps or []
        self._is_fitted = False

    def add_step(self, step: Any):
        self.steps.append(step)

    def fit(self, X: Any, y: Any = None):
        X_t = X
        for step in self.steps:
            if hasattr(step, "fit"):
                if hasattr(step, "transform") or hasattr(step, "fit_transform"):
                    if hasattr(step, "fit_transform"):
                        X_t = step.fit_transform(X_t, y)
                    else:
                        step.fit(X_t, y)
                        X_t = step.transform(X_t)
                else:
                    step.fit(X_t, y)
            elif callable(step):
                # Function-based step (stateless)
                X_t = step(X_t)
        
        self._is_fitted = True
        return self

    def transform(self, X: Any):
        if not self.steps:
            return X
            
        X_t = X
        for step in self.steps:
            if hasattr(step, "transform"):
                X_t = step.transform(X_t)
            elif callable(step) and not hasattr(step, "fit"):
                 X_t = step(X_t)
        return X_t

    def fit_transform(self, X: Any, y: Any = None):
        self.fit(X, y)
        return self.transform(X) # Re-transform to be safe or use cached X_t from fit if optimized

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Saved preprocessing pipeline to {path}")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise e

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PreprocessingPipeline':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
            
        try:
            with open(path, "rb") as f:
                pipeline = pickle.load(f)
            logger.info(f"Loaded preprocessing pipeline from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise e
