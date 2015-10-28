"""
The DataObjects module includes the different type of data representations
available in RAVEN

Created on September 16, 2015
@author: maljdp
supercedes DataObjects.py from alfoa (2/16/2013)
"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from DataObjects.Data import Data' outside of this submodule
from .Data import Data, NotConsistentData, ConstructError
from .Point import Point
from .PointSet import PointSet
from .History import History
from .HistorySet import HistorySet

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['Data','Point','PointSet','History','HistorySet']
