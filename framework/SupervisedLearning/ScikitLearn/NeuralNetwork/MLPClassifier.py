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
  Created on Jun 30, 2021

  @author: wangc
   Multi-layer Perceptron classifier.

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class MLPClassifier(ScikitLearnBase):
  """
    Multi-layer Perceptron Classifier
  """
  info = {'problemtype':'classification', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.multioutputWrapper = False
    import sklearn
    import sklearn.neural_network
    self.model = sklearn.neural_network.MLPClassifier

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlNode{MLPClassifier} implements a multi-layer perceptron algorithm that trains using \textbf{Backpropagation}
                            More precisely, it trains using some form of gradient descent and the gradients are calculated using Backpropagation.
                            For classification, it minimizes the Cross-Entropy loss function, and it supports multi-class classification.
                            \zNormalizationPerformed{MLPClassifier}
                        """
    specs.addSub(InputData.parameterInputFactory("hidden_layer_sizes", contentType=InputTypes.IntegerTupleType,
                                                 descr=r"""The ith element represents the number of neurons in the ith hidden layer.
                                                 lenght = n\_layers - 2""", default=(100,)))
    specs.addSub(InputData.parameterInputFactory("activation", contentType=InputTypes.makeEnumType("activation", "activationType",['identity', 'logistic', 'tanh','tanh']),
                                                 descr=r"""Activation function for the hidden layer:
                                                 \begin{itemize}
                                                  \item identity:  no-op activation, useful to implement linear bottleneck, returns $f(x) = x$
                                                  \item logistic: the logistic sigmoid function, returns $f(x) = 1 / (1 + exp(-x))$.
                                                  \item tanh: the hyperbolic tan function, returns $f(x) = tanh(x)$.
                                                  \item relu:  the rectified linear unit function, returns $f(x) = max(0, x)$
                                                 \end{itemize}
                                                 """, default='relu'))
    specs.addSub(InputData.parameterInputFactory("solver", contentType=InputTypes.makeEnumType("solver", "solverType",['lbfgs', 'sgd', 'adam']),
                                                 descr=r"""The solver for weight optimization:
                                                 \begin{itemize}
                                                  \item lbfgs: is an optimizer in the family of quasi-Newton methods.
                                                  \item sgd: refers to stochastic gradient descent.
                                                  \item adam: refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
                                                 \end{itemize}
                                                 """, default='adam'))
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""L2 penalty (regularization term) parameter.""", default=0.0001))
    specs.addSub(InputData.parameterInputFactory("batch_size", contentType=InputTypes.IntegerOrStringType,
                                                 descr=r"""Size of minibatches for stochastic optimizers. If the solver is `lbfgs',
                                                 the classifier will not use minibatch. When set to ``auto", batch\_size=min(200, n\_samples)""",
                                                 default='auto'))
    specs.addSub(InputData.parameterInputFactory("learning_rate", contentType=InputTypes.makeEnumType("learningRate", "learningRateType",['constant', 'invscaling', 'adaptive']),
                                                 descr=r"""Learning rate schedule for weight updates.:
                                                 \begin{itemize}
                                                  \item constant: is a constant learning rate given by `learning\_rate\_init'.
                                                  \item invscaling: gradually decreases the learning rate at each time step `t' using
                                                  an inverse scaling exponent of `power\_t'. effective\_learning\_rate = learning\_rate\_init / pow(t, power\_t)
                                                  \item adaptive: keeps the learning rate constant to `learning\_rate\_init' as long as training
                                                  loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at
                                                  least tol, or fail to increase validation score by at least tol if `early\_stopping' is on,
                                                  the current learning rate is divided by 5. Only used when solver=`sgd'.
                                                 \end{itemize}
                                                 """, default='constant'))
    specs.addSub(InputData.parameterInputFactory("learning_rate_init", contentType=InputTypes.FloatType,
                                                 descr=r"""The initial learning rate used. It controls the step-size in updating the weights.
                                                 Only used when solver=`sgd' or `adam'.""", default=0.001))
    specs.addSub(InputData.parameterInputFactory("power_t", contentType=InputTypes.FloatType,
                                                 descr=r"""The exponent for inverse scaling learning rate. It is used in updating effective
                                                 learning rate when the learning\_rate is set to `invscaling'. Only used when solver=`sgd'.""",
                                                 default=0.5))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.IntegerType,
                                                 descr=r"""Maximum number of iterations. The solver iterates until convergence
                                                 (determined by `tol') or this number of iterations. For stochastic solvers (`sgd', `adam'),
                                                 note that this determines the number of epochs (how many times each data point will be used),
                                                 not the number of gradient steps.""", default=200))
    specs.addSub(InputData.parameterInputFactory("shuffle", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to shuffle samples in each iteration. Only used when solver=`sgd' or `adam'.""", default=True))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""Determines random number generation for weights and bias initialization,
                                                 train-test split if early stopping is used, and batch sampling when solver=`sgd' or `adam'.""",
                                                 default=None))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""Tolerance for the optimization.""", default=1e-4))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to print progress messages to stdout.""", default=False))
    specs.addSub(InputData.parameterInputFactory("warm_start", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, reuse the solution of the previous call to fit as initialization, otherwise,
                                                 just erase the previous solution.""", default=False))
    specs.addSub(InputData.parameterInputFactory("momentum", contentType=InputTypes.FloatType,
                                                 descr=r"""Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=`sgd'.""", default=0.9))
    specs.addSub(InputData.parameterInputFactory("nesterovs_momentum", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to use Nesterov's momentum. Only used when solver=`sgd' and momentum > 0.""", default=True))
    specs.addSub(InputData.parameterInputFactory("early_stopping", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to use early stopping to terminate training when validation score is not improving.
                                                 If set to true, it will automatically set aside ten-percent of training data as validation and terminate
                                                 training when validation score is not improving by at least tol for n\_iter\_no\_change consecutive
                                                 epochs. The split is stratified, except in a multilabel setting. If early stopping is False, then
                                                 the training stops when the training loss does not improve by more than tol for n\_iter\_no\_change
                                                 consecutive passes over the training set. Only effective when solver=`sgd' or `adam'.""", default=False))
    specs.addSub(InputData.parameterInputFactory("validation_fraction", contentType=InputTypes.FloatType,
                                                 descr=r"""The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1.
                                                 Only used if early\_stopping is True""", default=0.1))
    specs.addSub(InputData.parameterInputFactory("beta_1", contentType=InputTypes.FloatType,
                                                 descr=r"""Exponential decay rate for estimates of first moment vector in adam, should be in $[0, 1)$.
                                                 Only used when solver=`adam'.""", default=0.9))
    specs.addSub(InputData.parameterInputFactory("beta_2", contentType=InputTypes.FloatType,
                                                 descr=r"""Exponential decay rate for estimates of second moment vector in adam, should be in $[0, 1)$.
                                                 Only used when solver=`adam'.""", default=0.999))
    specs.addSub(InputData.parameterInputFactory("epsilon", contentType=InputTypes.FloatType,
                                                 descr=r"""Value for numerical stability in adam. Only used when solver=`adam'.""", default=1e-8))
    specs.addSub(InputData.parameterInputFactory("n_iter_no_change", contentType=InputTypes.IntegerType,
                                                descr=r"""Maximum number of epochs to not meet tol improvement. Only effective when
                                                solver=`sgd' or `adam'""", default=10))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['hidden_layer_sizes','activation','solver','alpha','batch_size',
                                                               'learning_rate','learning_rate_init','power_t','max_iter', 'shuffle',
                                                               'random_state','tol','verbose','warm_start','momentum','nesterovs_momentum',
                                                               'early_stopping','validation_fraction','beta_1','beta_2','epsilon',
                                                               'n_iter_no_change'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
