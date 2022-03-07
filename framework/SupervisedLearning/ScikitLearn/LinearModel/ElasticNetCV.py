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
  Elastic Net model with cross validation.

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class ElasticNetCV(ScikitLearnBase):
  """
    Linear Elastic Net model with iterative fitting along a regularization path.
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
    self.model = sklearn.linear_model.ElasticNetCV

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(ElasticNetCV, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{ElasticNetCV} employs
                        Linear regression with combined L1 and L2 priors as regularizer.
                        This model is similar to the \xmlNode{ElasticNet}
                        with the addition of an iterative fitting along a regularization path (via cross-validation).
                        \zNormalizationNotPerformed{ElasticNetCV}
                        """
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-4))
    specs.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType,
                                                 descr=r"""Length of the path. $eps=1e-3$ means that
                                                 $alpha_min / alpha_max = 1e-3$.""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("l1_ratio", contentType=InputTypes.FloatType,
                                                 descr=r"""specifies the
                                                 float between 0 and 1 passed to ElasticNet (scaling between l1 and l2 penalties).
                                                 For $l1\_ratio = 0$ the penalty is an L2 penalty. For $l1\_ratio = 1$ it is
                                                 an L1 penalty. For $0 < l1\_ratio < 1$, the penalty is a combination of L1
                                                 and L2 This parameter can be a list, in which case the different values
                                                 are tested by cross-validation and the one giving the best prediction score
                                                 is used. Note that a good choice of list of values for l1\_ratio is often to
                                                 put more values close to 1 (i.e. Lasso) and less close to 0 (i.e. Ridge),
                                                 as in $[.1, .5, .7, .9, .95, .99, 1]$.""", default=0.5))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("precompute", contentType=InputTypes.StringType,
                                                 descr=r"""Whether to use a precomputed Gram matrix to speed up calculations.
                                                 For sparse input this option is always True to preserve sparsity.""", default='auto'))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of iterations.""", default=1000))
    specs.addSub(InputData.parameterInputFactory("cv", contentType=InputTypes.IntegerType,
                                                 descr=r"""Determines the cross-validation splitting strategy.
                                                 It specifies the number of folds.""", default=None))
    specs.addSub(InputData.parameterInputFactory("positive", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, forces the coefficients to be positive.""", default=True))
    specs.addSub(InputData.parameterInputFactory("selection", contentType=InputTypes.makeEnumType("selection", "selectionType",['cyclic', 'random']),
                                                 descr=r"""If set to ``random'', a random coefficient is updated every iteration
                                                 rather than looping over features sequentially by default. This (setting to `random'')
                                                 often leads to significantly faster convergence especially when tol is higher than $1e-4$""", default='cyclic'))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=False))
    specs.addSub(InputData.parameterInputFactory("n_alphas", contentType=InputTypes.IntegerType,
                                                 descr=r"""Number of alphas along the regularization path,
                                                 used for each l1_ratio.""", default=100))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['tol', 'eps','l1_ratio',
                                                               'precompute', 'fit_intercept','cv',
                                                               'max_iter', 'normalize','selection','positive', 'n_alphas'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
