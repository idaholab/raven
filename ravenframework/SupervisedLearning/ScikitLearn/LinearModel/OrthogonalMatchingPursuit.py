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
  Orthogonal Matching Pursuit model (OMP).

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class OrthogonalMatchingPursuit(ScikitLearnBase):
  """
    Orthogonal Matching Pursuit model (OMP).
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
    self.model = sklearn.linear_model.OrthogonalMatchingPursuit

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(OrthogonalMatchingPursuit, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{OrthogonalMatchingPursuit}
                        implements the OMP algorithm for approximating the fit of a
                        linear model with constraints imposed on the number of non-zero
                        coefficients (ie. the $\ell_0$ pseudo-norm). OMP is based on a greedy
                        algorithm that includes at each step the atom most highly correlated
                        with the current residual. It is similar to the simpler matching
                        pursuit (MP) method, but better in that at each iteration, the residual
                        is recomputed using an orthogonal projection on the space of the
                        previously chosen dictionary elements.
                        \zNormalizationNotPerformed{OrthogonalMatchingPursuit}
                        """
    specs.addSub(InputData.parameterInputFactory("n_nonzero_coefs", contentType=InputTypes.IntegerType,
                                                 descr=r"""Desired number of non-zero entries in the solution. If None (by default)
                                                 this value is set to ten-percent of n\_features.""", default=None))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Maximum norm of the residual.""", default=None))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=True))
    specs.addSub(InputData.parameterInputFactory("precompute", contentType=InputTypes.StringType,
                                                 descr=r"""Whether to use a precomputed Gram and Xy matrix to speed up calculations.
                                                 Improves performance when n\_targets or n\_samples is very large. """, default='auto'))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['fit_intercept','normalize','precompute','tol','n_nonzero_coefs'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
