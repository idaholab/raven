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
  The SupervisedLeaerning module includes different types of ROM strategies available in RAVEN

  Created on May 8, 2018
  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  supercedes SupervisedLearning.py
"""

from __future__ import absolute_import
#Internal Module Lazy Import-------------------------------------------------------------------------
from utils.lazyImporterUtils import import_collable_lazy
#Internal Module Lazy Import End---------------------------------------------------------------------
# These lines ensure that we do not have to do something like:
# 'from Samplers.Sampler import Sampler' outside of this submodule
from .SupervisedLearning import supervisedLearning
ARMA                     = import_collable_lazy("ARMA.ARMA")
GaussPolynomialRom       = import_collable_lazy("GaussPolynomialRom.GaussPolynomialRom")
HDMRRom                  = import_collable_lazy("HDMRRom.HDMRRom")
MSR                      = import_collable_lazy("MSR.MSR")
NDinterpolatorRom        = import_collable_lazy("NDinterpolatorRom.NDinterpolatorRom")
NDinvDistWeight          = import_collable_lazy("NDinvDistWeight.NDinvDistWeight")
NDsplineRom              = import_collable_lazy("NDsplineRom.NDsplineRom")
SciKitLearn              = import_collable_lazy("SciKitLearn.SciKitLearn")
pickledROM               = import_collable_lazy("pickledROM.pickledROM")
PolyExponential          = import_collable_lazy("PolyExponential.PolyExponential")
DynamicModeDecomposition = import_collable_lazy("DynamicModeDecomposition.DynamicModeDecomposition")
KerasClassifier          = import_collable_lazy("KerasClassifier.KerasClassifier")
KerasMLPClassifier       = import_collable_lazy("KerasMLPClassifier.KerasMLPClassifier")
KerasConvNetClassifier   = import_collable_lazy("KerasConvNetClassifier.KerasConvNetClassifier")
KerasLSTMClassifier      = import_collable_lazy("KerasLSTMClassifier.KerasLSTMClassifier")
#Collection               = import_collable_lazy("ROMCollection.Collection")
#Segments                 = import_collable_lazy("ROMCollection.Segments")
#Clusters                 = import_collable_lazy("ROMCollection.Clusters")
from .ROMCollection      import Collection, Segments, Clusters
#from .ARMA               import ARMA
#from .GaussPolynomialRom import GaussPolynomialRom
#from .HDMRRom            import HDMRRom
#from .MSR                import MSR
#from .NDinterpolatorRom  import NDinterpolatorRom
#from .NDinvDistWeight    import NDinvDistWeight
#from .NDsplineRom        import NDsplineRom
#from .SciKitLearn        import SciKitLearn
#from .pickledROM         import pickledROM
#from .PolyExponential    import PolyExponential
#from .DynamicModeDecomposition import DynamicModeDecomposition
#from .ROMCollection      import Collection, Segments, Clusters
# KERAS classifiers
#from .KerasClassifier import isTensorflowAvailable
#if isTensorflowAvailable():
#  from .KerasClassifier import KerasClassifier
#  from .KerasMLPClassifier import KerasMLPClassifier
#  from .KerasConvNetClassifier import KerasConvNetClassifier
#  from .KerasLSTMClassifier import KerasLSTMClassifier

# Factory methods
from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass
