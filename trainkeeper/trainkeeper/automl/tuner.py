
import logging
from typing import Any, Callable, Dict, Optional, Union, List
import warnings

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    optuna = None
    Trial = Any

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    AutoML tuner using Optuna for hyperparameter optimization.
    
    Example:
        >>> def objective(trial):
        >>>     lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        >>>     # ... train model ...
        >>>     return accuracy
        >>> 
        >>> tuner = HyperparameterTuner(study_name="my_study", direction="maximize")
        >>> best_params = tuner.optimize(objective, n_trials=20)
    """
    
    def __init__(
        self,
        study_name: str,
        direction: str = "maximize",
        storage: Optional[str] = None,
        sampler: Optional[str] = "tpe",
        pruner: Optional[str] = "median"
    ):
        """
        Args:
            study_name: Name of the study
            direction: 'maximize' or 'minimize'
            storage: Database URL for persistent storage (e.g., 'sqlite:///db.sqlite3')
            sampler: Sampling strategy ('tpe', 'random', 'grid')
            pruner: Pruning strategy ('median', 'hyperband', 'none')
        """
        if optuna is None:
            raise ImportError("Optuna is required for AutoML. Install with `pip install optuna`")
            
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        
        # Configure sampler
        if sampler == "tpe":
            self.sampler = optuna.samplers.TPESampler()
        elif sampler == "random":
            self.sampler = optuna.samplers.RandomSampler()
        elif sampler == "grid":
            self.sampler = optuna.samplers.GridSampler({}) # Requires grid definition later
        else:
            self.sampler = None # Default
            
        # Configure pruner
        if pruner == "median":
            self.pruner = optuna.pruners.MedianPruner()
        elif pruner == "hyperband":
            self.pruner = optuna.pruners.HyperbandPruner()
        else:
            self.pruner = optuna.pruners.NopPruner()
            
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True
        )
        
    def optimize(
        self, 
        objective_fn: Callable[[Trial], float], 
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run the optimization.
        
        Args:
            objective_fn: Function taking `trial` and returning float metric
            n_trials: Number of trials
            timeout: Max time in seconds
            n_jobs: Parallel jobs (-1 for all CPUs)
            
        Returns:
            Best parameters
        """
        logger.info(f"Starting AutoML optimization for {self.study_name} with {n_trials} trials...")
        
        try:
            self.study.optimize(objective_fn, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
            
        logger.info(f"Optimization finished. Best value: {self.study.best_value}")
        return self.study.best_params
    
    @property
    def best_trial(self):
        return self.study.best_trial
        
    def get_dataframe(self):
        """Get study results as pandas DataFrame"""
        return self.study.trials_dataframe()
    
    def plot_optimization_history(self):
        """Plot optimization history"""
        return optuna.visualization.plot_optimization_history(self.study)
        
    def plot_param_importances(self):
        """Plot parameter importances"""
        return optuna.visualization.plot_param_importances(self.study)
