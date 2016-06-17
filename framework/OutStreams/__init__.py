"""
  The OutStreamManagers module includes the different type of ways to output
  data available in RAVEN

  Created on April 5, 2016
  @author: maljdp
  supercedes OutStreamManager.py from alfoa (11/14/2013)
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from OutStreamManagers.OutStreamPlot import OutStreamPlot' outside
## of this submodule
from .OutStreamManager import OutStreamManager
from .OutStreamPlot import OutStreamPlot
from .OutStreamPrint import OutStreamPrint

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['OutStreamManager','OutStreamPlot','OutStreamPrint']
