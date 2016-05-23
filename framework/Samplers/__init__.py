"""
  The Samplers module includes the different type of Sampling strategy available in RAVEN

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa (2/16/2013)
"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from Samplers.Sampler import Sampler' outside of this submodule
from .Sampler import Sampler
# Forward Samplers
from .ForwardSampler        import ForwardSampler
from .MonteCarlo            import MonteCarlo
from .Grid                  import Grid
from .Stratified            import Stratified
from .FactorialDesign       import FactorialDesign
from .ResponseSurfaceDesign import ResponseSurfaceDesign
from .Sobol                 import Sobol
from .SparseGridCollocation import SparseGridCollocation
from .EnsembleForward       import EnsembleForwardSampler
# Adaptive Samplers
from .AdaptiveSampler      import AdaptiveSampler
from .LimitSurfaceSearch   import LimitSurfaceSearch
from .AdaptiveSobol        import AdaptiveSobol
from .AdaptiveSparseGrid   import AdaptiveSparseGrid
# Dynamic Event Tree-based Samplers
from .DynamicEventTree         import DynamicEventTree
from .AdaptiveDynamicEventTree import AdaptiveDET
# Factory methods
from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['Sampler','AdaptiveSampler','ForwardSampler','MonteCarlo','Grid','Stratified','FactorialDesign',
           'ResponseSurfaceDesign','Sobol','EnsembleForward','SparseGridCollocation',
           'DynamicEventTree','LimitSurfaceSearch','AdaptiveDynamicEventTree',
           'AdaptiveSparseGrid','AdaptiveSobol']
