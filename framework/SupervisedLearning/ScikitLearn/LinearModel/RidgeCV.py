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

  @author: alfoa
  Ridge Regressor with cross-validation

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .ScikitLearnBase import SciktLearnBase
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class RidgeCV(SciktLearnBase):
  """
   Ridge Regressor with cross-validation
  """
  info = {'problemtype':'regression 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.linear_model
    import sklearn.multioutput
    # we wrap the model with the multi output regressor (for multitarget)
    self.model = sklearn.linear_model.RidgeCV()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(RidgeCV, cls).getInputSpecification()
    specs.description = r"""The \\xmlNode{RidgeCV} regressor also known as
                             \textit{linear least squares with l2 regularization} solves a regression
                             model where the loss function is the linear least squares function and the
                             regularization is given by the l2-norm.
                             In addition, a cross-validation method is applied to optimize the hyper-parameter.
                        """
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True, the
                                                 regressors X will be normalized before regression by subtracting the mean and dividing
                                                 by the l2-norm. """, default=False))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Precision of the solution""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("gcv_mode", contentType=InputTypes.makeEnumType("gcv_mode", "gcvType",['auto', 'svd', 'eigen']),
                                                 descr=r"""Flag indicating which strategy to use when performing Leave-One-Out Cross-Validation.
                                                 Options are:
                                                 \\begin{itemize}
                                                   \\item \textit{auto}, use ``svd'' if $n\_samples > n\_features$, otherwise use ``eigen''
                                                   \\item \textit{svd}, force use of singular value decomposition of X when X is
                                                    dense, eigenvalue decomposition of X^T.X when X is sparse
                                                   \\item \textit{eigen}, force computation via eigendecomposition of X.X^T
                                                 \\end{itemize}
                                                 """, default='auto'))
    specs.addSub(InputData.parameterInputFactory("alpha_per_target", contentType=InputTypes.BoolType,
                                                 descr=r"""Flag indicating whether to optimize the alpha value for each target separately
                                                 (for multi-output settings: multiple prediction targets). When set to True, after fitting,
                                                 the alpha_ attribute will contain a value for each target. When set to False, a single alpha
                                                  is used for all targets.""", default=False))
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
    settings, notFound = paramInput.findNodesAndExtractValues(['gcv_mode','fit_intercept','tol',
                                                               'normalize','alpha_per_target','cv'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
