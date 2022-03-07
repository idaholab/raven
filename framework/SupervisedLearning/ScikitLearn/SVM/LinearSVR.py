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

class LinearSVR(ScikitLearnBase):
  """
    Linear Support Vector Regressor
  """
  info = {'problemtype':'regression', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.svm
    self.model = sklearn.svm.LinearSVR

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(LinearSVR, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{LinearSVR} \textit{Linear Support Vector Regressor} is
                            similar to SVR with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm,
                            so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
                            This class supports both dense and sparse input.
                            \zNormalizationPerformed{LinearSVR}
                            """
    specs.addSub(InputData.parameterInputFactory('epsilon', contentType=InputTypes.FloatType,
                                                 descr=r"""Epsilon parameter in the epsilon-insensitive loss function. The value of
                                                 this parameter depends on the scale of the target variable y. If unsure, set $epsilon=0.$""", default=0.0))
    specs.addSub(InputData.parameterInputFactory("loss", contentType=InputTypes.makeEnumType("loss", "lossType",['epsilon_insensitive’','squared_epsilon_insensitive']),
                                                 descr=r"""Specifies the loss function. The epsilon-insensitive loss (standard SVR)
                                                 is the L1 loss, while the squared epsilon-insensitive loss (``squared_epsilon_insensitive'') is the L2 loss.""",
                                                 default='squared_epsilon_insensitive'))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-4))
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
    specs.addSub(InputData.parameterInputFactory("dual", contentType=InputTypes.BoolType,
                                                 descr=r"""Select the algorithm to either solve the dual or primal optimization problem.
                                                 Prefer dual=False when $n_samples > n_features$.""", default=True))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""Hard limit on iterations within solver.``-1'' for no limit""", default=-1))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['epsilon', 'dual', 'loss', 'tol', 'fit_intercept',
                                                               'intercept_scaling',  'max_iter'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
