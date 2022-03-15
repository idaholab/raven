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
  Multi-task Elasting Net Regressor

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class MultiTaskElasticNet(ScikitLearnBase):
  """
    Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer.
  """
  info = {'problemtype':'regression', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.multioutputWrapper = False
    import sklearn
    import sklearn.linear_model
    self.model = sklearn.linear_model.MultiTaskElasticNet

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(MultiTaskElasticNet, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{MultiTaskElasticNet} employs
                        Linear regression with combined L1 and L2 priors as regularizer.
                        The optimization objective for MultiTaskElasticNet is:
                        $(1 / (2 * n\_samples)) * ||Y - XW||^{Fro}_2
                        + alpha * l1\_ratio * ||W||_{21}
                        + 0.5 * alpha * (1 - l1\_ratio) * ||W||_{Fro}^2$
                        \\Where:
                        $||W||_{21} = \sum_i \sqrt{\sum_j w_{ij}^2}$
                        i.e. the sum of norm of each row.
                        \zNormalizationNotPerformed{MultiTaskElasticNet}
                        """
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-4))
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""specifies a constant
                                                 that multiplies the penalty terms.
                                                 $alpha = 0$ is equivalent to an ordinary least square, solved by the
                                                 \textbf{LinearRegression} object.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("l1_ratio", contentType=InputTypes.FloatType,
                                                 descr=r"""specifies the
                                                 ElasticNet mixing parameter, with $0 <= l1\_ratio <= 1$.
                                                 For $l1\_ratio = 0$ the penalty is an L2 penalty.
                                                 For $l1\_ratio = 1$ it is an L1 penalty.
                                                 For $0 < l1\_ratio < 1$, the penalty is a combination of L1 and L2.""", default=0.5))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of iterations.""", default=1000))
    specs.addSub(InputData.parameterInputFactory("selection", contentType=InputTypes.makeEnumType("selection", "selectionType",['cyclic', 'random']),
                                                 descr=r"""If set to ``random'', a random coefficient is updated every iteration
                                                 rather than looping over features sequentially by default. This (setting to `random'')
                                                 often leads to significantly faster convergence especially when tol is higher than $1e-4$""", default='cyclic'))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=False))
    specs.addSub(InputData.parameterInputFactory("warm_start", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, reuse the solution of the previous call
                                                 to fit as initialization, otherwise, just erase the previous solution.""", default=False))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['tol','alpha','l1_ratio', 'fit_intercept',
                                                               'max_iter', 'normalize','selection', 'warm_start'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
