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
  The LinearModel folder includes different types of ScikitLearn Linear Model-based ROMs
  that are available via RAVEN

  Created on Jan 21, 2020
  @author: alfoa
"""
# These lines ensure that we do not have to do something like:
# 'from Samplers.Sampler import Sampler' outside of this submodule
from .ARDRegression import ARDRegression
from .LassoCV import LassoCV
from .Lasso import Lasso
from .LogisticRegression import LogisticRegression
from .BayesianRidge import BayesianRidge
from .ElasticNet import ElasticNet
from .Lars import Lars
from .LarsCV import LarsCV
from .LassoLars import LassoLars
from .LassoLarsCV import LassoLarsCV
from .LassoLarsIC import LassoLarsIC
from .LinearRegression import LinearRegression
from .MultiTaskLasso import MultiTaskLasso
from .MultiTaskElasticNet import MultiTaskElasticNet
from .MultiTaskElasticNetCV import MultiTaskElasticNetCV
from .MultiTaskLassoCV import MultiTaskLassoCV
from .OrthogonalMatchingPursuit import OrthogonalMatchingPursuit
from .OrthogonalMatchingPursuitCV import OrthogonalMatchingPursuitCV
from .ElasticNetCV import ElasticNetCV
from .Perceptron import Perceptron
from .PassiveAggressiveRegressor import PassiveAggressiveRegressor
from .Ridge import Ridge
from .RidgeCV import RidgeCV
from .RidgeClassifier import RidgeClassifier
from .PassiveAggressiveClassifier import PassiveAggressiveClassifier
from .SGDClassifier import SGDClassifier
from .SGDRegressor import SGDRegressor
from .RidgeClassifierCV import RidgeClassifierCV
# Factory methods
from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass
