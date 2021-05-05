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

# These lines ensure that we do not have to do something like:
# 'from Samplers.Sampler import Sampler' outside of this submodule
from .SupervisedLearning import supervisedLearning

from .ARMA               import ARMA
from .GaussPolynomialRom import GaussPolynomialRom
from .HDMRRom            import HDMRRom
from .MSR                import MSR
from .NDinterpolatorRom  import NDinterpolatorRom
from .NDinvDistWeight    import NDinvDistWeight
from .NDsplineRom        import NDsplineRom
from .SciKitLearn        import SciKitLearn
from .SyntheticHistory   import SyntheticHistory
from .pickledROM         import pickledROM
from .PolyExponential    import PolyExponential
from .DynamicModeDecomposition import DynamicModeDecomposition
from .ROMCollection      import Collection, Segments, Clusters, Interpolated

# KERAS classifiers
from .KerasClassifier import KerasClassifier
from .KerasMLPClassifier import KerasMLPClassifier
from .KerasConvNetClassifier import KerasConvNetClassifier

from .Factory import factory
