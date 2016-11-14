"""
  The Metrics module includes the different type of metrics
  to measure distance among RAVEN dataobjects
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from OutStreamManagers.OutStreamPlot import OutStreamPlot' outside
## of this submodule
from .Metric import Metric
from .Minkowski import Minkowski

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass

__all__ = ['Minkowski','DTW']
