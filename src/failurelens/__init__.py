"""FailureLens: training-dynamics and OOD reliability diagnostics."""

from .joint import false_alarm_association, local_training_signal
from .metrics import detection_metrics
from .tracker import DynamicsSummary, TrainingDynamicsTracker

__all__ = [
    "DynamicsSummary",
    "TrainingDynamicsTracker",
    "detection_metrics",
    "false_alarm_association",
    "local_training_signal",
]
__version__ = "0.3.0"
