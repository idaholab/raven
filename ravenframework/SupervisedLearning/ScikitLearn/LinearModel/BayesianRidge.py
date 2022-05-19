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
  Bayesian ridge regression

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class BayesianRidge(ScikitLearnBase):
  """
    Bayesian ARD regression
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
    self.model = sklearn.linear_model.BayesianRidge

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(BayesianRidge, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{BayesianRidge} is Bayesian Ridge regression.
                        It estimates a probabilistic model of the regression problem as
                        described above. The prior for the coefficient is given by a
                        spherical Gaussian:
                        $p(w|\lambda) = \mathcal{N}(w|0,\lambda^{-1}\mathbf{I}_{p})$
                        The parameters $w$, $\alpha$ and $\lambda$ are estimated jointly during
                        the fit of the model, the regularization parameters $\alpha$ and $\lambda$
                        being estimated by maximizing the log marginal likelihood.
                        \zNormalizationNotPerformed{BayesianRidge}
                        """
    specs.addSub(InputData.parameterInputFactory("n_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""Maximum number of iterations.""", default=300))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("alpha_1", contentType=InputTypes.FloatType,
                                                 descr=r"""Hyper-parameter : shape parameter for the Gamma
                                                 distribution prior over the alpha parameter.""", default=1e-6))
    specs.addSub(InputData.parameterInputFactory("alpha_2", contentType=InputTypes.FloatType,
                                                 descr=r"""Hyper-parameter : inverse scale parameter (rate parameter)
                                                 for the Gamma distribution prior over the alpha parameter.""", default=1e-6))
    specs.addSub(InputData.parameterInputFactory("lambda_1", contentType=InputTypes.FloatType,
                                                 descr=r"""Hyper-parameter : shape parameter for the Gamma distribution
                                                 prior over the lambda parameter.""", default=1e-6))
    specs.addSub(InputData.parameterInputFactory("lambda_2", contentType=InputTypes.FloatType,
                                                 descr=r"""Hyper-parameter : inverse scale parameter (rate parameter) for
                                                 the Gamma distribution prior over the lambda parameter.""", default=1e-6))
    specs.addSub(InputData.parameterInputFactory("alpha_init", contentType=InputTypes.FloatType,
                                                 descr=r"""Initial value for alpha (precision of the noise).
                                                  If not set, alpha_init is $1/Var(y)$.""", default=None))
    specs.addSub(InputData.parameterInputFactory("lambda_init", contentType=InputTypes.FloatType,
                                                 descr=r"""Initial value for lambda (precision of the weights).""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("compute_score", contentType=InputTypes.BoolType,
                                                 descr=r"""If True, compute the objective function at each step of the
                                                 model.""", default=False))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to calculate the intercept for this model. Specifies if a constant (a.k.a. bias or intercept)
                                                  should be added to the decision function.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True,
                                                 the regressors X will be normalized before regression by subtracting the mean and
                                                 dividing by the l2-norm.""", default=False))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.BoolType,
                                                 descr=r"""Verbose mode when fitting the model.""", default=False))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['tol', 'alpha_1','alpha_2','lambda_1','lambda_2',
                                                               'compute_score', 'fit_intercept',
                                                               'n_iter', 'normalize', 'verbose'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
