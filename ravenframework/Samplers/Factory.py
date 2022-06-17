# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Created on May 16, 2016
  @author: alfoa
  extracted from alfoa (2/16/2013) Samplers.py
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------

from ..EntityFactoryBase import EntityFactory

################################################################################
# Forward samplers
from .MonteCarlo import MonteCarlo
from .Grid import Grid
from .Stratified import Stratified
from .FactorialDesign import FactorialDesign
from .ResponseSurfaceDesign import ResponseSurfaceDesign
from .Sobol import Sobol
from .SparseGridCollocation import SparseGridCollocation
from .EnsembleForward import EnsembleForward
from .CustomSampler import CustomSampler

# Adaptive samplers
from .LimitSurfaceSearch import LimitSurfaceSearch
from .AdaptiveSobol import AdaptiveSobol
from .AdaptiveSparseGrid import AdaptiveSparseGrid
from .AdaptiveMonteCarlo import AdaptiveMonteCarlo

# Dynamic Event Tree-based Samplers
from .DynamicEventTree import DynamicEventTree
from .AdaptiveDynamicEventTree import AdaptiveDynamicEventTree

# MCMC Samplers
from .MCMC import Metropolis
from .MCMC import AdaptiveMetropolis

factory = EntityFactory('Sampler')
factory.registerType('MonteCarlo'              , MonteCarlo)
factory.registerType('Grid'                    , Grid)
factory.registerType('Stratified'              , Stratified)
factory.registerType('FactorialDesign'         , FactorialDesign)
factory.registerType('ResponseSurfaceDesign'   , ResponseSurfaceDesign)
factory.registerType('Sobol'                   , Sobol)
factory.registerType('SparseGridCollocation'   , SparseGridCollocation)
factory.registerType('CustomSampler'           , CustomSampler)
factory.registerType('EnsembleForward'         , EnsembleForward)
factory.registerType('LimitSurfaceSearch'      , LimitSurfaceSearch)
factory.registerType('AdaptiveSobol'           , AdaptiveSobol)
factory.registerType('AdaptiveSparseGrid'      , AdaptiveSparseGrid)
factory.registerType('DynamicEventTree'        , DynamicEventTree)
factory.registerType('AdaptiveDynamicEventTree', AdaptiveDynamicEventTree)
factory.registerType('AdaptiveMonteCarlo'      , AdaptiveMonteCarlo)
factory.registerType('Metropolis'              , Metropolis)
factory.registerType('AdaptiveMetropolis'      , AdaptiveMetropolis)
