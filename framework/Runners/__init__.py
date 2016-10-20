"""
  The Runners module includes the different ways of parallelizing the MultiRuns
  of RAVEN.

  Created on September 12, 2016
  @author: maljdp
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from Runners.Runner import Runner' outside
## of this submodule
from .Runner import Runner
from .ExternalRunner import ExternalRunner
from .InternalRunner import InternalRunner
from .SharedMemoryRunner import SharedMemoryRunner
from .DistributedMemoryRunner import DistributedMemoryRunner

# from .Factory import knownTypes
# from .Factory import returnInstance
# from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['Runner', 'ExternalRunner', 'InternalRunner', 'SharedMemoryRunner', 'DistributedMemoryRunner']
