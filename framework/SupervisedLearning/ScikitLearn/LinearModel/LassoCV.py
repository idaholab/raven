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
  Lasso linear model with iterative fitting along a regularization path

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class LassoCV(ScikitLearnBase):
  """
    Lasso linear model with iterative fitting along a regularization path
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
    self.model = sklearn.linear_model.LassoCV

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(LassoCV, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{LassoCV} (\textit{Lasso linear model with iterative fitting along a regularization path})
                        is an algorithm for regression problem. The best model is selected by cross-validation.
                        It minimizes the usual sum of squared errors, with a bound on the sum of the
                        absolute values of the coefficients:
                        \begin{equation}
                         (1 / (2 * n\_samples)) * ||y - Xw||^2\_2 + alpha * ||w||\_1
                        \end{equation}
                        \zNormalizationNotPerformed{LassoCV}
                        """
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-4))
    specs.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType,
                                                 descr=r"""Length of the path. $eps=1e-3$ means that
                                                 $alpha_min / alpha_max = 1e-3$.""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("n_alphas", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of iterations.""", default=100))
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
                                                 descr=r"""The maximum number of iterations.""", default=1000))
    specs.addSub(InputData.parameterInputFactory("positive", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, forces the coefficients to be positive.""", default=False))
    specs.addSub(InputData.parameterInputFactory("selection", contentType=InputTypes.makeEnumType("selection", "selectionType",['cyclic', 'random']),
                                                 descr=r"""If set to ``random'', a random coefficient is updated every iteration
                                                 rather than looping over features sequentially by default. This (setting to `random'')
                                                 often leads to significantly faster convergence especially when tol is higher than $1e-4$""", default='cyclic'))
    specs.addSub(InputData.parameterInputFactory("cv", contentType=InputTypes.IntegerType,
                                                 descr=r"""Determines the cross-validation splitting strategy.
                                                 It specifies the number of folds..""", default=None))
    specs.addSub(InputData.parameterInputFactory("alphas", contentType=InputTypes.FloatListType,
                                                 descr=r"""List of alphas where to compute the models. If None alphas
                                                 are set automatically.""", default=None))
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
    settings, notFound = paramInput.findNodesAndExtractValues(['tol','eps', 'n_alphas', 'fit_intercept','normalize',
                                                               'precompute','max_iter','positive','selection','cv',
                                                               'alphas', 'verbose'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
