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
  Created on May 8, 2018
  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------

from EntityFactoryBase import EntityFactory

################################################################################

from SupervisedLearning.SupervisedLearning import supervisedLearning
# Forward Samplers
from SupervisedLearning.ARMA               import ARMA
from SupervisedLearning.GaussPolynomialRom import GaussPolynomialRom
from SupervisedLearning.HDMRRom            import HDMRRom
from SupervisedLearning.MSR                import MSR
from SupervisedLearning.NDinterpolatorRom  import NDinterpolatorRom
from SupervisedLearning.NDinvDistWeight    import NDinvDistWeight
from SupervisedLearning.NDsplineRom        import NDsplineRom
from SupervisedLearning.SciKitLearn        import SciKitLearn
from SupervisedLearning.SyntheticHistory   import SyntheticHistory
from SupervisedLearning.pickledROM         import pickledROM
from SupervisedLearning.PolyExponential    import PolyExponential
from SupervisedLearning.DynamicModeDecomposition import DynamicModeDecomposition
from SupervisedLearning.ROMCollection      import Collection, Segments, Clusters, Interpolated
from .KerasClassifier import KerasClassifier
from SupervisedLearning.KerasMLPClassifier import KerasMLPClassifier
from SupervisedLearning.KerasConvNetClassifier import KerasConvNetClassifier
from SupervisedLearning.KerasLSTMClassifier import KerasLSTMClassifier
from SupervisedLearning.KerasLSTMRegression import KerasLSTMRegression
from SupervisedLearning.ROMCollection      import Collection, Segments, Clusters

################################################################################

factory = EntityFactory('supervisedLearning')
factory.registerType('NDspline'              , NDsplineRom)
factory.registerType('NDinvDistWeight'       , NDinvDistWeight)
factory.registerType('NDsplineRom'           , NDsplineRom)
factory.registerType('SciKitLearn'           , SciKitLearn)
factory.registerType('GaussPolynomialRom'    , GaussPolynomialRom)
factory.registerType('HDMRRom'               , HDMRRom)
factory.registerType('MSR'                   , MSR)
factory.registerType('ARMA'                  , ARMA)
factory.registerType('SyntheticHistory'      , SyntheticHistory)
factory.registerType('pickledROM'            , pickledROM)
factory.registerType('PolyExponential'       , PolyExponential)
factory.registerType('DMD'                   , DynamicModeDecomposition)
factory.registerType('Segments'              , Segments)
factory.registerType('Clusters'              , Clusters)
factory.registerType('Interpolated'          , Interpolated)
factory.registerType('KerasMLPClassifier'    , KerasMLPClassifier)
factory.registerType('KerasConvNetClassifier', KerasConvNetClassifier)
factory.registerType('KerasLSTMClassifier'   , KerasLSTMClassifier)
factory.registerType('KerasLSTMRegression'   , KerasLSTMRegression)
