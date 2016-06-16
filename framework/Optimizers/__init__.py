"""
  The Optimizers module includes the different type of Optimization Sampling strategy available in RAVEN

  Created on June 16, 2016
  @author: chenj
"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from Optimizers.Optimizer import Sampler' outside of this submodule
from .Optimizer import Optimizer
from .Optimizer import SPSA
from .Optimizer import FiniteDifference

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass

# from .ForwardSampler        import ForwardSampler
# from .MonteCarlo            import MonteCarlo
# from .Grid                  import Grid
# from .Stratified            import Stratified
# from .FactorialDesign       import FactorialDesign
# from .ResponseSurfaceDesign import ResponseSurfaceDesign
# from .Sobol                 import Sobol
# from .SparseGridCollocation import SparseGridCollocation
# from .EnsembleForward       import EnsembleForwardSampler
# from .CustomSampler         import CustomSampler
# 
# # Adaptive Samplers
# from .AdaptiveSampler      import AdaptiveSampler
# from .LimitSurfaceSearch   import LimitSurfaceSearch
# from .AdaptiveSobol        import AdaptiveSobol
# from .AdaptiveSparseGrid   import AdaptiveSparseGrid
# # Dynamic Event Tree-based Samplers
# from .DynamicEventTree         import DynamicEventTree
# from .AdaptiveDynamicEventTree import AdaptiveDET
# # Factory methods
# from .Factory import knownTypes
# from .Factory import returnInstance
# from .Factory import returnClass

# # We should not really need this as we do not use wildcard imports
# __all__ = ['Sampler','AdaptiveSampler','ForwardSampler','MonteCarlo','Grid','CustomSampler','Stratified',
#            'FactorialDesign','ResponseSurfaceDesign','Sobol','EnsembleForward','SparseGridCollocation',
#            'DynamicEventTree','LimitSurfaceSearch','AdaptiveDynamicEventTree','AdaptiveSparseGrid','AdaptiveSobol']
