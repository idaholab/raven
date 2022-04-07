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
  SGD Classifier

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class SGDClassifier(ScikitLearnBase):
  """
    SGD Classifier
  """
  info = {'problemtype':'classification', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.linear_model
    self.model = sklearn.linear_model.SGDClassifier

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(SGDClassifier, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{SGDClassifier} implements regularized linear models with stochastic
                        gradient descent (SGD) learning for classification: the gradient of the loss is estimated each sample at
                        a time and the model is updated along the way with a decreasing strength schedule
                        (aka learning rate). For best results using the default learning rate schedule, the
                        data should have zero mean and unit variance.
                        This implementation works with data represented as dense or sparse arrays of floating
                        point values for the features. The model it fits can be controlled with the loss parameter;
                        by default, it fits a linear support vector machine (SVM).
                        The regularizer is a penalty added to the loss function that shrinks model parameters towards
                        the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a
                        combination of both (Elastic Net). If the parameter update crosses the 0.0 value because
                        of the regularizer, the update is truncated to $0.0$ to allow for learning sparse models and
                        achieve online feature selection.
                        \zNormalizationPerformed{SGDClassifier}
                        """
    specs.addSub(InputData.parameterInputFactory("loss", contentType=InputTypes.makeEnumType("loss", "lossType",['hinge', 'log', 'modified_huber', 'squared_hinge','perceptron',
                                                                                                                 'squared_loss', 'huber','epsilon_insensitive','squared_epsilon_insensitive']),
                                                 descr=r"""The loss function to be used. Defaults to ``hinge'', which gives a linear SVM.The ``log'' loss gives logistic regression, a
                                                 probabilistic classifier. ``modified\_huber'' is another smooth loss that brings tolerance to outliers as well as probability estimates.
                                                 ``squared_hinge'' is like hinge but is quadratically penalized. ``perceptron'' is the linear loss used by the perceptron algorithm.
                                                 The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.""", default='hinge'))
    specs.addSub(InputData.parameterInputFactory("penalty", contentType=InputTypes.makeEnumType("penalty", "penaltyType",['l2', 'l1', 'elasticnet']),
                                                 descr=r"""The penalty (aka regularization term) to be used. Defaults to ``l2'' which is the standard regularizer for linear SVM models.
                                                 ``l1'' and ``elasticnet'' might bring sparsity to the model (feature selection) not achievable with ``l2''.""", default='l2'))
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute
                                                 the learning rate when set to learning_rate is set to ``optimal''.""", default=0.0001))
    specs.addSub(InputData.parameterInputFactory("l1_ratio", contentType=InputTypes.FloatType,
                                                 descr=r"""The Elastic Net mixing parameter, with $0 <= l1\_ratio <= 1$. $l1\_ratio=0$ corresponds to L2 penalty, $l1\_ratio=1$ to L1.
                                                 Only used if penalty is ``elasticnet''.""", default=0.15))
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of passes over the training data (aka epochs).""", default=1000))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""The stopping criterion. If it is not None, training will stop when $(loss > best\_loss - tol)$ for $n\_iter\_no\_change$
                                                 consecutive epochs.""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("shuffle", contentType=InputTypes.BoolType,
                                                 descr=r"""TWhether or not the training data should be shuffled after each epoch """, default=True))
    specs.addSub(InputData.parameterInputFactory("epsilon", contentType=InputTypes.FloatType,
                                                 descr=r"""Epsilon in the epsilon-insensitive loss functions; only if loss is ``huber'', ``epsilon\_insensitive'', or
                                                 ``squared\_epsilon\_insensitive''. For ``huber'', determines the threshold at which it becomes less important to get the
                                                 prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label
                                                 are ignored if they are less than this threshold.""", default=0.1))
    specs.addSub(InputData.parameterInputFactory("learning_rate", contentType=InputTypes.makeEnumType("learning_rate", "learningType",['constant', 'optimal', 'invscaling','adaptive']),
                                                 descr=r"""The learning rate schedule:
                                                 \begin{itemize}
                                                  \item constant: $eta = eta0$
                                                  \item optimal: $eta = 1.0 / (alpha * (t + t0))$ where t0 is chosen by a heuristic proposed by Leon Bottou.
                                                  \item invscaling: $eta = eta0 / pow(t, power\_t)$
                                                  \item adaptive: $eta = eta0$, as long as the training keeps decreasing. Each time n\_iter\_no\_change consecutive epochs fail
                                                  to decrease the training loss by tol or fail to increase validation score by tol if early\_stopping is True, the current
                                                  learning rate is divided by 5.
                                                 \end{itemize}
                                                 """, default='optimal'))
    specs.addSub(InputData.parameterInputFactory("eta0", contentType=InputTypes.FloatType,
                                                 descr=r"""The initial learning rate for the ``constant'', ``invscaling'' or ``adaptive'' schedules. The default value is 0.0
                                                 as eta0 is not used by the default schedule ``optimal''.""", default=0.0))
    specs.addSub(InputData.parameterInputFactory("power_t", contentType=InputTypes.FloatType,
                                                 descr=r"""The exponent for inverse scaling learning rate.""", default=0.5))
    specs.addSub(InputData.parameterInputFactory("early_stopping", contentType=InputTypes.BoolType,
                                                 descr=r"""hether to use early stopping to terminate training when validation score is not
                                                 improving. If set to True, it will automatically set aside a stratified fraction of training
                                                 data as validation and terminate training when validation score is not improving by at least
                                                 tol for n\_iter\_no\_change consecutive epochs.""", default=False))
    specs.addSub(InputData.parameterInputFactory("validation_fraction", contentType=InputTypes.FloatType,
                                                 descr=r"""The proportion of training data to set aside as validation set for early stopping.
                                                 Must be between 0 and 1. Only used if early\_stopping is True.""", default=0.1))
    specs.addSub(InputData.parameterInputFactory("n_iter_no_change", contentType=InputTypes.IntegerType,
                                                descr=r"""Number of iterations with no improvement to wait before early stopping.""", default=5))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""Used to shuffle the training data, when shuffle is set to
                                                 True. Pass an int for reproducible output across multiple function calls.""",
                                                 default=None))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.IntegerType,
                                                 descr=r"""The verbosity level""", default=0))
    specs.addSub(InputData.parameterInputFactory("class_weight", contentType=InputTypes.makeEnumType("classWeight", "classWeightType",['balanced']),
                                                 descr=r"""If not given, all classes are supposed to have weight one.
                                                 The “balanced” mode uses the values of y to automatically adjust weights
                                                 inversely proportional to class frequencies in the input data""", default=None))
    specs.addSub(InputData.parameterInputFactory("warm_start", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, reuse the solution of the previous call
                                                 to fit as initialization, otherwise, just erase the previous solution.""", default=False))
    specs.addSub(InputData.parameterInputFactory("average", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, computes the averaged SGD weights accross
                                                 all updates and stores the result in the coef_ attribute.""", default=False))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['loss','penalty','alpha','l1_ratio','fit_intercept',
                                                               'max_iter','tol','shuffle','epsilon', 'learning_rate',
                                                               'eta0','power_t','early_stopping','validation_fraction',
                                                               'n_iter_no_change', 'random_state', 'verbose',
                                                               'class_weight', 'warm_start', 'average'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
