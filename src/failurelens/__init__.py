"""FailureLens: lightweight training-dynamics diagnostics."""

from .metrics import detection_metrics
from .tracker import DynamicsSummary, TrainingDynamicsTracker

__all__ = ["DynamicsSummary", "TrainingDynamicsTracker", "detection_metrics"]
__version__ = "0.2.0"
