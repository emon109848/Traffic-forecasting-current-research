"""Bounded reproducible training, without automatic test-set evaluation."""

from .trainer import TrainingConfig, masked_mae, run_training

__all__ = ['TrainingConfig', 'masked_mae', 'run_training']
