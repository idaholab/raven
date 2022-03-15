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
  Least Angle Regression model

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

class Lars(ScikitLearnBase):
  """
    Least Angle Regression model
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
    self.model = sklearn.linear_model.Lars

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(Lars, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{Lars} (\textit{Least Angle Regression model})
                        is a regression algorithm for high-dimensional data.
                        The LARS algorithm provides a means of producing an estimate of which variables
                        to include, as well as their coefficients, when a response variable is
                        determined by a linear combination of a subset of potential covariates.
                        \zNormalizationNotPerformed{Lars}
                        """
    specs.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType,
                                                 descr=r"""The machine-precision regularization in the computation of the Cholesky
                                                 diagonal factors. Increase this for very ill-conditioned systems. Unlike the tol
                                                 parameter in some iterative optimization-based algorithms, this parameter does not
                                                 control the tolerance of the optimization.""", default=finfo(float).eps))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("precompute", contentType=InputTypes.StringType,
                                                 descr=r"""Whether to use a precomputed Gram matrix to speed up calculations.
                                                 For sparse input this option is always True to preserve sparsity.""", default='auto'))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=True))
    specs.addSub(InputData.parameterInputFactory("n_nonzero_coefs", contentType=InputTypes.IntegerType,
                                                 descr=r"""Target number of non-zero coefficients.""", default=500))
    # new in sklearn version 0.23
    # specs.addSub(InputData.parameterInputFactory("jitter", contentType=InputTypes.FloatType,
    #                                              descr=r"""Upper bound on a uniform noise parameter to be added to the
    #                                              y values, to satisfy the model’s assumption of one-at-a-time computations.
    #                                              Might help with stability.""", default=None))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.BoolType,
                                                 descr=r"""Sets the verbosity amount.""", default=False))
    specs.addSub(InputData.parameterInputFactory("fit_path", contentType=InputTypes.BoolType,
                                                 descr=r"""If True the full path is stored in the coef_path_ attribute.
                                                 If you compute the solution for a large problem or many targets,
                                                 setting fit_path to False will lead to a speedup, especially with a
                                                 small alpha.""", default=True))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['eps','precompute', 'fit_intercept',
                                                               'normalize','n_nonzero_coefs', 'verbose',
                                                               'fit_path'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
