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
  Created on 3-Nov-2021

  @author: cogljj
  module for Multi-layer perceptron regression
"""
#Internal Modules------------------------------------------------------------------------------------
from .KerasRegression import KerasRegression
#Internal Modules End--------------------------------------------------------------------------------

class KerasMLPRegression(KerasRegression):
  """
    Multi-layer perceptron regressor constructed using Keras API in TensorFlow
  """
  info = {'problemtype':'regression', 'normalize':True}

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
    specs.description = r"""Multi-Layer Perceptron (MLP) (or Artificial Neural Network - ANN), a class of feedforward
        ANN, can be viewed as a logistic regression where input is first transformed
        using a non-linear transformation. This transformation probjects the input data into a
        space where it becomes linearly separable. This intermediate layer is referred to as a
        \textbf{hidden layer}. An MLP consists of at least three layers of nodes. Except for the
        input nodes, each node is a neuron that uses a nonlinear \textbf{activation function}. MLP
        utilizes a suppervised learning technique called \textbf{Backpropagation} for training.
        Generally, a single hidden layer is sufficient to make MLPs a universal approximator.
        However, many hidden layers, i.e. deep learning, can be used to model more complex nonlinear
        relationships. The extra layers enable composition of features from lower layers, potentially
        modeling complex data with fewer units than a similarly performing shallow network.
        \\
        \zNormalizationPerformed{KerasMLPRegression}
        \\
        In order to use this ROM, the \xmlNode{ROM} attribute \xmlAttr{subType} needs to
        be \xmlString{KerasMLPRegression}"""
    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'KerasMLPRegression'
    self.allowedLayers = self.basicLayers

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
