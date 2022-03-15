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
  Cross-Validated Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class MultiTaskLassoCV(ScikitLearnBase):
  """
    Cross-Validated Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
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
    self.model = sklearn.linear_model.MultiTaskLassoCV

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(MultiTaskLassoCV, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{MultiTaskLassoCV} (\textit{Multi-task Lasso model trained
                        with L1/L2 mixed-norm as regularizer}) is an algorithm for regression problem
                        where the optimization objective for Lasso is:
                        $(1 / (2 * n\_samples)) * ||Y - XW||^2_{Fro} + alpha * ||W||_{21}$
                        \\Where:
                        $||W||_{21} = \sum_i \sqrt{\sum_j w_{ij}^2}$
                        i.e. the sum of norm of each row.
                        In this model, the cross-validation is embedded for the automatic selection
                        of the best hyper-parameters.
                        \zNormalizationNotPerformed{MultiTaskLassoCV}
                        """
    specs.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType,
                                                 descr=r"""Length of the path. $eps=1e-3$ means that $alpha\_min / alpha\_max = 1e-3$.""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("n_alpha", contentType=InputTypes.IntegerType,
                                                 descr=r"""Number of alphas along the regularization path.""", default=100))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=False))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of iterations.""", default=1000))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-4))
    specs.addSub(InputData.parameterInputFactory("selection", contentType=InputTypes.makeEnumType("selection", "selectionType",['cyclic', 'random']),
                                                 descr=r"""If set to ``random'', a random coefficient is updated every iteration
                                                 rather than looping over features sequentially by default. This (setting to `random'')
                                                 often leads to significantly faster convergence especially when tol is higher than $1e-4$""", default='cyclic'))
    specs.addSub(InputData.parameterInputFactory("cv", contentType=InputTypes.IntegerType,
                                                 descr=r"""Determines the cross-validation splitting strategy.
                                                 It specifies the number of folds..""", default=5))

    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['eps','tol', 'fit_intercept','n_alpha'
                                                               'normalize','max_iter','selection','cv'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
