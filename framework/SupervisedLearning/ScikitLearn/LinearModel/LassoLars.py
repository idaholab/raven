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
  Lasso model fit with Least Angle Regression a.k.a. Lars

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

class LassoLars(ScikitLearnBase):
  """
    Lasso model fit with Least Angle Regression
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
    self.model = sklearn.linear_model.LassoLars

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(LassoLars, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{LassoLars} (\textit{Lasso model fit with Least Angle Regression})
                        It is a Linear Model trained with an L1 prior as regularizer.
                        The optimization objective for Lasso is:
                        \begin{equation}
                         (1 / (2 * n\_samples)) * ||y - Xw||^2\_2 + alpha * ||w||\_1
                        \end{equation}
                        \zNormalizationNotPerformed{LassoLars}
                        """
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Constant that multiplies the L1 term. Defaults to 1.0.
                                                 $alpha = 0$ is equivalent to an ordinary least square, solved by
                                                 the LinearRegression object. For numerical reasons, using $alpha = 0$
                                                 with the Lasso object is not advised.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=False))
    specs.addSub(InputData.parameterInputFactory("precompute", contentType=InputTypes.StringType,
                                                 descr=r"""Whether to use a precomputed Gram matrix to speed up calculations.
                                                 For sparse input this option is always True to preserve sparsity.""", default='auto'))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of iterations.""", default=500))
    specs.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType,
                                                 descr=r"""The machine-precision regularization in the computation of the Cholesky
                                                 diagonal factors. Increase this for very ill-conditioned systems. Unlike the tol
                                                 parameter in some iterative optimization-based algorithms, this parameter does not
                                                 control the tolerance of the optimization.""", default=finfo(float).eps))
    specs.addSub(InputData.parameterInputFactory("positive", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, forces the coefficients to be positive.""", default=False))
    # New in sklearn version 0.23
    # specs.addSub(InputData.parameterInputFactory("jitter", contentType=InputTypes.FloatType,
    #                                              descr=r"""Upper bound on a uniform noise parameter to be added to the y values,
    #                                              to satisfy the model’s assumption of one-at-a-time computations. Might help
    #                                              with stability.""", default=None))
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
    settings, notFound = paramInput.findNodesAndExtractValues(['alpha','fit_intercept', 'normalize', 'precompute',
                                                               'max_iter','eps','positive', 'verbose'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
