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
  @author: talbpaul, wangc
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
"""
from EntityFactoryBase import EntityFactory

################################################################################

from .SupervisedLearning import SupervisedLearning

from .GaussPolynomialRom import GaussPolynomialRom
from .HDMRRom            import HDMRRom
from .MSR                import MSR
from .NDinterpolatorRom  import NDinterpolatorRom
from .NDinvDistWeight    import NDinvDistWeight
from .NDspline           import NDspline
from .SyntheticHistory   import SyntheticHistory
from .pickledROM         import pickledROM
from .PolyExponential    import PolyExponential
from .DynamicModeDecomposition import DMD

## FIXME: convert the following ROM to use new ROM API
from .ARMA               import ARMA
from .ROMCollection      import Collection, Segments, Clusters, Interpolated

## Tensorflow-Keras Neural Network Models
from .KerasBase import KerasBase
from .KerasRegression import KerasRegression
from .KerasClassifier import KerasClassifier
from .KerasMLPClassifier import KerasMLPClassifier
from .KerasConvNetClassifier import KerasConvNetClassifier
from .KerasLSTMClassifier import KerasLSTMClassifier
from .KerasLSTMRegression import KerasLSTMRegression

# import ScikitLearn models
from .ScikitLearn.ScikitLearnBase import ScikitLearnBase
from .ScikitLearn.DiscriminantAnalysis import LinearDiscriminantAnalysis
from .ScikitLearn.DiscriminantAnalysis import QuadraticDiscriminantAnalysis

from .ScikitLearn.LinearModel.ARDRegression import ARDRegression
from .ScikitLearn.LinearModel.BayesianRidge import BayesianRidge
from .ScikitLearn.LinearModel.ElasticNet import ElasticNet
from .ScikitLearn.LinearModel.ElasticNetCV import ElasticNetCV
from .ScikitLearn.LinearModel.Lars import Lars
from .ScikitLearn.LinearModel.LarsCV import LarsCV
from .ScikitLearn.LinearModel.Lasso import Lasso
from .ScikitLearn.LinearModel.LassoCV import LassoCV
from .ScikitLearn.LinearModel.LassoLars import LassoLars
from .ScikitLearn.LinearModel.LassoLarsCV import LassoLarsCV
from .ScikitLearn.LinearModel.LassoLarsIC import LassoLarsIC
from .ScikitLearn.LinearModel.LinearRegression import LinearRegression
from .ScikitLearn.LinearModel.LogisticRegression import LogisticRegression
from .ScikitLearn.LinearModel.MultiTaskElasticNet import MultiTaskElasticNet
from .ScikitLearn.LinearModel.MultiTaskElasticNetCV import MultiTaskElasticNetCV
from .ScikitLearn.LinearModel.MultiTaskLasso import MultiTaskLasso
from .ScikitLearn.LinearModel.MultiTaskLassoCV import MultiTaskLassoCV
from .ScikitLearn.LinearModel.OrthogonalMatchingPursuit import OrthogonalMatchingPursuit
from .ScikitLearn.LinearModel.OrthogonalMatchingPursuitCV import OrthogonalMatchingPursuitCV
from .ScikitLearn.LinearModel.PassiveAggressiveClassifier import PassiveAggressiveClassifier
from .ScikitLearn.LinearModel.PassiveAggressiveRegressor import PassiveAggressiveRegressor
from .ScikitLearn.LinearModel.Perceptron import Perceptron
from .ScikitLearn.LinearModel.Ridge import Ridge
from .ScikitLearn.LinearModel.RidgeCV import RidgeCV
from .ScikitLearn.LinearModel.RidgeClassifier import RidgeClassifier
from .ScikitLearn.LinearModel.RidgeClassifierCV import RidgeClassifierCV
from .ScikitLearn.LinearModel.SGDClassifier import SGDClassifier
from .ScikitLearn.LinearModel.SGDRegressor import SGDRegressor

from .ScikitLearn.NaiveBayes.ComplementNBClassifier import ComplementNB
from .ScikitLearn.NaiveBayes.CategoricalNBClassifier import CategoricalNB
from .ScikitLearn.NaiveBayes.BernoulliNBClassifier import BernoulliNB
from .ScikitLearn.NaiveBayes.MultinomialNBClassifier import MultinomialNB
from .ScikitLearn.NaiveBayes.GaussianNBClassifier import GaussianNB

from .ScikitLearn.NeuralNetwork.MLPClassifier import MLPClassifier
from .ScikitLearn.NeuralNetwork.MLPRegressor import MLPRegressor

from .ScikitLearn.GaussianProcess.GaussianProcessClassifier import GaussianProcessClassifier
from .ScikitLearn.GaussianProcess.GaussianProcessRegressor import GaussianProcessRegressor

from .ScikitLearn.MultiClass.OneVsOneClassifier import OneVsOneClassifier
from .ScikitLearn.MultiClass.OneVsRestClassifier import OneVsRestClassifier
from .ScikitLearn.MultiClass.OutputCodeClassifier import OutputCodeClassifier


from .ScikitLearn.Neighbors.KNeighborsClassifier import KNeighborsClassifier
from .ScikitLearn.Neighbors.NearestCentroidClassifier import NearestCentroid
from .ScikitLearn.Neighbors.RadiusNeighborsRegressor import RadiusNeighborsRegressor
from .ScikitLearn.Neighbors.KNeighborsRegressor import KNeighborsRegressor
from .ScikitLearn.Neighbors.RadiusNeighborsClassifier import RadiusNeighborsClassifier

from .ScikitLearn.SVM.LinearSVC import LinearSVC
from .ScikitLearn.SVM.LinearSVR import LinearSVR
from .ScikitLearn.SVM.NuSVC import NuSVC
from .ScikitLearn.SVM.NuSVR import NuSVR
from .ScikitLearn.SVM.SVC import SVC
from .ScikitLearn.SVM.SVR import SVR

from .ScikitLearn.Tree.DecisionTreeClassifier import DecisionTreeClassifier
from .ScikitLearn.Tree.DecisionTreeRegressor import DecisionTreeRegressor
from .ScikitLearn.Tree.ExtraTreeClassifier import ExtraTreeClassifier
from .ScikitLearn.Tree.ExtraTreeRegressor import ExtraTreeRegressor
################################################################################

factory = EntityFactory('SupervisedLearning')
factory.registerAllSubtypes(SupervisedLearning)
