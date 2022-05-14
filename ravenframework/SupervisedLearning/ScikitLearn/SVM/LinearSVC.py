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
  Created on Jan 21, 2020

  @author: alfoa, wangc
  Linear Support Vector Classifier

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class LinearSVC(ScikitLearnBase):
  """
    Linear Support Vector Classifier
  """
  info = {'problemtype':'classification', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.svm
    self.model = sklearn.svm.LinearSVC

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(LinearSVC, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{LinearSVC} \textit{Linear Support Vector Classification} is
                            similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm,
                            so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
                            This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.
                            \zNormalizationPerformed{LinearSVC}
                            """
    specs.addSub(InputData.parameterInputFactory("penalty", contentType=InputTypes.makeEnumType("penalty", "penaltyType",['l1','l2']),
                                                 descr=r"""Specifies the norm used in the penalization. The ``l2'' penalty is the standard used in SVC.
                                                 The ``l1' leads to coefficients vectors that are sparse.""", default='l2'))
    specs.addSub(InputData.parameterInputFactory("loss", contentType=InputTypes.makeEnumType("loss", "lossType",['hinge','squared_hinge']),
                                                 descr=r"""Specifies the loss function. ``hinge'' is the standard SVM loss (used e.g. by the SVC class)
                                                 while ``squared_hinge'' is the square of the hinge loss. The combination of penalty=``l1' and loss=``hinge''
                                                 is not supported.""", default='squared_hinge'))
    specs.addSub(InputData.parameterInputFactory("dual", contentType=InputTypes.BoolType,
                                                 descr=r"""Select the algorithm to either solve the dual or primal optimization problem.
                                                 Prefer dual=False when $n_samples > n_features$.""", default=True))
    specs.addSub(InputData.parameterInputFactory('C', contentType=InputTypes.FloatType,
                                                 descr=r"""Regularization parameter. The strength of the regularization is inversely
                                                          proportional to C.
                                                           Must be strictly positive. The penalty is a squared l2 penalty..""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-4))
    specs.addSub(InputData.parameterInputFactory("multi_class", contentType=InputTypes.makeEnumType("multi_class", "multiclassType",['crammer_singer','ovr']),
                                                 descr=r"""Determines the multi-class strategy if y contains more than two classes. ``ovr'' trains
                                                 $n_classes$ one-vs-rest classifiers, while ``crammer_singer'' optimizes a joint objective over all classes.
                                                 While crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom used
                                                 in practice as it rarely leads to better accuracy and is more expensive to compute. If ``crammer_singer''
                                                 is chosen, the options loss, penalty and dual will be ignored.""", default='ovr'))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to calculate the intercept for this model. If set to false, no
                                                 intercept will be used in calculations (i.e. data is expected to be already centered).""", default=True))
    specs.addSub(InputData.parameterInputFactory("intercept_scaling", contentType=InputTypes.FloatType,
                                                 descr=r"""When fit_intercept is True, instance vector x becomes $[x, intercept_scaling]$,
                                                 i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended
                                                 to the instance vector. The intercept becomes $intercept_scaling * synthetic feature weight$
                                                 \nb the synthetic feature weight is subject to $l1/l2$ regularization as all other features.
                                                 To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept)
                                                 $intercept_scaling$ has to be increased.""", default=1.))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""Hard limit on iterations within solver.``-1'' for no limit""", default=1000))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.IntegerType,
                                                 descr=r"""Enable verbose output. Note that this setting takes advantage
                                                 of a per-process runtime setting in liblinear that, if enabled, may not
                                                 work properly in a multithreaded context.""", default=0))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""Controls the pseudo random number generation for shuffling
                                                 the data for the dual coordinate descent (if dual=True). When dual=False
                                                 the underlying implementation of LinearSVC is not random and
                                                 random_state has no effect on the results. Pass an int for reproducible
                                                 output across multiple function calls. """, default=None))
    specs.addSub(InputData.parameterInputFactory("class_weight", contentType=InputTypes.makeEnumType("classWeight", "classWeightType",['balanced']),
                                                 descr=r"""If not given, all classes are supposed to have weight one.
                                                 The “balanced” mode uses the values of y to automatically adjust weights
                                                 inversely proportional to class frequencies in the input data""", default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['C', 'dual', 'penalty', 'loss', 'tol', 'fit_intercept',
                                                               'intercept_scaling',  'max_iter', 'multi_class', 'verbose',
                                                               'random_state', 'class_weight'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
