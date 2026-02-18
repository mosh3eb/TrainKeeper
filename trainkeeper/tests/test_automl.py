import pytest
import unittest.mock as mock
import sys
from trainkeeper.automl import tuner
from trainkeeper.automl import HyperparameterTuner



def test_tuner_optimization():
    """Test optimization loop with mocked optuna"""
    # Create valid mocks for Optuna components
    mock_study = mock.Mock()
    mock_study.best_params = {"x": 10}
    mock_study.best_value = 0.95
    
    # We need to mock the optuna module effectively
    with mock.patch('trainkeeper.automl.tuner.optuna') as mock_optuna_module:
        mock_optuna_module.create_study.return_value = mock_study
        # Ensure samplers/pruners are mockable
        mock_optuna_module.samplers.TPESampler = mock.Mock()
        mock_optuna_module.pruners.MedianPruner = mock.Mock()
        
        tuner = HyperparameterTuner("test_study")
        
        # Test objective function
        def objective(trial):
            return 0.95
            
        best_params = tuner.optimize(objective, n_trials=5)
        
        assert best_params == {"x": 10}
        mock_study.optimize.assert_called_once()
