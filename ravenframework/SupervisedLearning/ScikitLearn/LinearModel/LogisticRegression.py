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
  Linear Logistic Regression

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class LogisticRegression(ScikitLearnBase):
  """
    Linear Logistic Regression
  """
  info = {'problemtype':'regression', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.linear_model
    self.model = sklearn.linear_model.LogisticRegression

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(LogisticRegression, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{LogisticRegression}  is
                            a logit, MaxEnt classifier.
                            In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme
                            if the ``multi_class'' option is set to ``ovr'', and uses the cross-entropy loss if the
                            ``multi_class'' option is set to ``multinomial''. (Currently the ``multinomial'' option
                            is supported only by the ``lbfgs'', ``sag'', ``saga'' and ``newton-cg'' solvers.)
                            This class implements regularized logistic regression using the ``liblinear'' library, ``newton-cg'',
                            ``sag'', ``saga'' and ``lbfgs'' solvers. Regularization is applied by default. It can handle both dense and sparse input.
                            The ``newton-cg'', ``sag'', and ``lbfgs'' solvers support only L2 regularization with primal formulation,
                            or no regularization. The ``liblinear'' solver supports both L1 and L2 regularization, with a dual formulation
                            only for the L2 penalty. The Elastic-Net regularization is only supported by the ``saga'' solver.
                            \zNormalizationPerformed{LogisticRegression}
                            """
    specs.addSub(InputData.parameterInputFactory("penalty", contentType=InputTypes.makeEnumType("penalty", "penaltyType",['l1','l2', 'elasticnet', 'none']),
                                                 descr=r"""Used to specify the norm used in the penalization. The newton-cg, sag and lbfgs solvers
                                                 support only l2 penalties. elasticnet is only supported by the saga solver. If none (
                                                 not supported by the liblinear solver), no regularization is applied.""", default='l2'))
    specs.addSub(InputData.parameterInputFactory("dual", contentType=InputTypes.BoolType,
                                                 descr=r"""Select the algorithm to either solve the dual or primal optimization problem.
                                                 Prefer dual=False when $n_samples > n_features$.""", default=True))
    specs.addSub(InputData.parameterInputFactory('C', contentType=InputTypes.FloatType,
                                                 descr=r"""Regularization parameter. The strength of the regularization is inversely
                                                 proportional to C.Must be strictly positive.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for stopping criterion""", default=1e-4))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to calculate the intercept for this model. Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.""", default=True))
    specs.addSub(InputData.parameterInputFactory("intercept_scaling", contentType=InputTypes.FloatType,
                                                 descr=r"""When fit_intercept is True, instance vector x becomes $[x, intercept_scaling]$,
                                                 i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended
                                                 to the instance vector. The intercept becomes $intercept_scaling * synthetic_feature_weight$
                                                 \nb the synthetic feature weight is subject to $l1/l2$ regularization as all other features.
                                                 To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept)
                                                 $intercept_scaling$ has to be increased.""", default=1.))
    specs.addSub(InputData.parameterInputFactory("solver", contentType=InputTypes.makeEnumType("solver", "solverType",['newton-cg','lbfgs', 'liblinear','sag','saga']),
                                                 descr=r"""Algorithm to use in the optimization problem.
                                                 \begin{itemize}
                                                   \item For small datasets, ``liblinear'' is a good choice, whereas ``sag'' and ``saga'' are faster for large ones.
                                                   \item For multiclass problems, only ``newton-cg'', ``sag'', ``saga'' and ``lbfgs'' handle multinomial loss; `
                                                   `liblinear'' is limited to one-versus-rest schemes.
                                                   \item ``newton-cg'', ``lbfgs'', ``sag'' and ``saga'' handle L2 or no penalty
                                                   \item ``liblinear'' and ``saga'' also handle L1 penalty
                                                   \item ``saga'' also supports ``elasticnet'' penalty
                                                   \item ``liblinear'' does not support setting penalty=``none''
                                                 \end{itemize}""", default='lbfgs'))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""Hard limit on iterations within solver.``-1'' for no limit""", default=100))
    specs.addSub(InputData.parameterInputFactory("multi_class", contentType=InputTypes.makeEnumType("multiClass", "multiClassType",['auto','ovr', 'multinomial']),
                                                 descr=r"""If the option chosen is ``ovr'', then a binary problem is fit for each label. For ``multinomial''
                                                 the loss minimised is the multinomial loss fit across the entire probability distribution, even when the
                                                 data is binary. ``multinomial' is unavailable when solver=``liblinear''. ``auto'' selects ``ovr'' if the data is
                                                 binary, or if solver=``liblinear'', and otherwise selects ``multinomial''.""", default='auto'))
    specs.addSub(InputData.parameterInputFactory("l1_ratio", contentType=InputTypes.FloatType,
                                                 descr=r"""The Elastic-Net mixing parameter, with $0 <= l1_ratio <= 1$. Only used if penalty=``elasticnet''.
                                                 Setting $l1_ratio=0$ is equivalent to using penalty=``l2'', while setting $l1_ratio=1$ is equivalent to using
                                                 $penalty=``l1''$. For $0 < l1_ratio <1$, the penalty is a combination of L1 and L2.""", default=0.5))
    specs.addSub(InputData.parameterInputFactory("class_weight", contentType=InputTypes.makeEnumType("classWeight", "classWeightType",['balanced']),
                                                 descr=r"""If not given, all classes are supposed to have weight one.
                                                 The “balanced” mode uses the values of y to automatically adjust weights
                                                 inversely proportional to class frequencies in the input data""", default=None))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data.""",
                                                 default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['C', 'dual', 'penalty', 'l1_ratio', 'tol', 'fit_intercept',
                                                               'solver','intercept_scaling',  'max_iter', 'multi_class',
                                                               'class_weight', 'random_state'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
