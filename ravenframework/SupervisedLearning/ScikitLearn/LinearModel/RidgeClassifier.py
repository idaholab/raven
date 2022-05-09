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
  Ridge Classifier

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

class RidgeClassifier(ScikitLearnBase):
  """
    Ridge Classifier
  """
  info = {'problemtype':'classification', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.linear_model
    self.model = sklearn.linear_model.RidgeClassifier

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(RidgeClassifier, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{RidgeClassifier} is a classifier that uses Ridge regression.
                        This classifier first converts the target values into {-1, 1} and then treats
                        the problem as a regression task (multi-output regression in the multiclass case).
                        \zNormalizationNotPerformed{RidgeClassifier}
                        """
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Regularization strength; must be a positive float. Regularization
                                                 improves the conditioning of the problem and reduces the variance of the estimates.
                                                 Larger values specify stronger regularization. Alpha corresponds to $1 / (2C)$ in other
                                                 linear models such as LogisticRegression or LinearSVC.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True, the
                                                 regressors X will be normalized before regression by subtracting the mean and dividing
                                                 by the l2-norm. """, default=False))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""Maximum number of iterations for conjugate gradient solver.""", default=None))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Precision of the solution""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("solver", contentType=InputTypes.makeEnumType("solver", "solverType",['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                                                 descr=r"""Solver to use in the computational routines:
                                                 \begin{itemize}
                                                   \item auto, chooses the solver automatically based on the type of data.
                                                   \item svd, uses a Singular Value Decomposition of X to compute the Ridge coefficients. More stable for singular
                                                               matrices than ``cholesky''.
                                                   \item cholesky, uses the standard scipy.linalg.solve function to obtain a closed-form solution.
                                                   \item sparse\_cg, uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. As an iterative algorithm,
                                                              this solver is more appropriate than ‘cholesky’ for large-scale data (possibility to set tol and max\_iter).
                                                   \item lsqr, uses the dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. It is the fastest and uses
                                                               an iterative procedure.
                                                   \item sag, uses a Stochastic Average Gradient descent, and ``saga'' uses its improved, unbiased version named SAGA.
                                                              Both methods also use an iterative procedure, and are often faster than other solvers when both
                                                              n\_samples and n\_features are large. Note that ``sag'' and ``saga'' fast convergence is only guaranteed on
                                                              features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
                                                 \end{itemize}""", default='auto'))
    specs.addSub(InputData.parameterInputFactory("class_weight", contentType=InputTypes.makeEnumType("classWeight", "classWeightType",['balanced']),
                                                 descr=r"""If not given, all classes are supposed to have weight one.
                                                 The “balanced” mode uses the values of y to automatically adjust weights
                                                 inversely proportional to class frequencies in the input data""", default=None))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""Used to shuffle the training data, when shuffle is set to
                                                 True. Pass an int for reproducible output across multiple function calls.""",
                                                 default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['alpha','fit_intercept','max_iter',
                                                               'normalize','tol','solver', 'random_state',
                                                               'class_weight'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
