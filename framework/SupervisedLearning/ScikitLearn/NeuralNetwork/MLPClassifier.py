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
from .ScikitLearnBase import SciktLearnBase
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class MLPClassifier(SciktLearnBase):
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
    import sklearn.neural_network
    import sklearn.multioutput
    # we wrap the model with the multi output classifier (for multitarget)
    self.model = sklearn.multioutput.MultiOutputClassifier(sklearn.neural_network.MLPClassifier)

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
    specs.description = r"""The \\xmlNode{MLPClassifier} implements a multi-layer perceptron algorithm that trains using \textbf{Backpropagation}
                            More precisely, it trains using some form of gradient descent and the gradients are calculated using Backpropagation.
                            For classification, it minimizes the Cross-Entropy loss function, and it supports multi-class classification.
                        """
    specs.addSub(InputData.parameterInputFactory("hidden_layer_sizes", contentType=InputTypes.IntegerTupleType,
                                                 descr=r"""The ith element represents the number of neurons in the ith hidden layer.
                                                 lenght = n\_layers - 2""", default=(100,)))
    specs.addSub(InputData.parameterInputFactory("activation", contentType=InputTypes.makeEnumType("activation", "activationType",['identity', 'logistic', 'tanh','tanh']),
                                                 descr=r"""Activation function for the hidden layer:
                                                 \\begin{itemize}
                                                  \\item identity:  no-op activation, useful to implement linear bottleneck, returns $f(x) = x$
                                                  \\item logistic: the logistic sigmoid function, returns $f(x) = 1 / (1 + exp(-x))$.
                                                  \\item tanh: the hyperbolic tan function, returns $f(x) = tanh(x)$.
                                                  \\item relu:  the rectified linear unit function, returns $f(x) = max(0, x)$
                                                 \\end{itemize}
                                                 """, default='relu'))
    specs.addSub(InputData.parameterInputFactory("solver", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("alphas", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("batch_size", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("learning_rate", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("learning_rate_init", contentType=InputTypes.FloatType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("power_t", contentType=InputTypes.FloatType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("shuffle", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.FloatType,
                                                 descr=r"""The stopping criterion. If it is not None, training will stop when $(loss > best\_loss - tol)$ for $n\_iter\_no\_change$
                                                 consecutive epochs.""", default=1e-3))
    specs.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("warm_start", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("momentum", contentType=InputTypes.FloatType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("nesterovs_momentum", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("early_stopping", contentType=InputTypes.StringType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("validation_fraction", contentType=InputTypes.FloatType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("beta_1", contentType=InputTypes.FloatType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("beta_2", contentType=InputTypes.FloatType,
                                                 descr=r"""""", default=))
    specs.addSub(InputData.parameterInputFactory("epsilon", contentType=InputTypes.FloatType,
                                                 descr=r"""Epsilon in the epsilon-insensitive loss functions; only if loss is ``huber'', ``epsilon\_insensitive'', or
                                                 ``squared\_epsilon\_insensitive''. For ``huber'', determines the threshold at which it becomes less important to get the
                                                 prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label
                                                 are ignored if they are less than this threshold.""", default=0.1))
    specs.addSub(InputData.parameterInputFactory("n_iter_no_change", contentType=InputTypes.IntegerType,
                                                descr=r"""Number of iterations with no improvement to wait before early stopping.""", default=5))
    specs.addSub(InputData.parameterInputFactory("max_fun", contentType=InputTypes.IntegerType,
                                                 descr=r"""""", default=))

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
                                                               'n_iter_no_change'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
