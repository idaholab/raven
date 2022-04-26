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
  Lasso model fit with Lars using BIC or AIC for model selection.

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from numpy import finfo
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class LassoLarsIC(ScikitLearnBase):
  """
    Lasso model fit with Lars using BIC or AIC for model selection
  """
  info = {'problemtype':'regression', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.linear_model
    self.model = sklearn.linear_model.LassoLarsIC

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(LassoLarsIC, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{LassoLarsIC} (\textit{Lasso model fit with Lars using BIC or AIC for model selection})
                        is a Lasso model fit with Lars using BIC or AIC for model selection.
                        The optimization objective for Lasso is:
                        $(1 / (2 * n\_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1$
                        AIC is the Akaike information criterion and BIC is the Bayes Information criterion. Such criteria
                        are useful to select the value of the regularization parameter by making a trade-off between the
                        goodness of fit and the complexity of the model. A good model should explain well the data
                        while being simple.
                        \zNormalizationNotPerformed{LassoLarsIC}
                        """
    specs.addSub(InputData.parameterInputFactory("criterion", contentType=InputTypes.makeEnumType("criterion", "criterionType",['bic', 'aic']),
                                                 descr=r"""The type of criterion to use.""", default='aic'))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                 the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=True))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of iterations.""", default=500))
    specs.addSub(InputData.parameterInputFactory("precompute", contentType=InputTypes.StringType,
                                                 descr=r"""Whether to use a precomputed Gram matrix to speed up calculations.
                                                 For sparse input this option is always True to preserve sparsity.""", default='auto'))
    specs.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType,
                                                 descr=r"""The machine-precision regularization in the computation of the Cholesky
                                                 diagonal factors. Increase this for very ill-conditioned systems. Unlike the tol
                                                 parameter in some iterative optimization-based algorithms, this parameter does not
                                                 control the tolerance of the optimization.""", default=finfo(float).eps))
    specs.addSub(InputData.parameterInputFactory("positive", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, forces the coefficients to be positive.""", default=False))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.BoolType,
                                                 descr=r"""Amount of verbosity.""", default=False))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['fit_intercept','max_iter', 'normalize', 'precompute',
                                                               'eps','positive','criterion', 'verbose'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
