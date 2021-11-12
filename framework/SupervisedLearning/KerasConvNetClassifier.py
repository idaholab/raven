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
  Created on Dec. 20, 2018
  @author: wangc
  module for Convolutional neural network (CNN)
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
######
#Internal Modules------------------------------------------------------------------------------------
from .KerasClassifier import KerasClassifier
#Internal Modules End--------------------------------------------------------------------------------

class KerasConvNetClassifier(KerasClassifier):
  """
    Convolutional neural network (CNN) classifier constructed using Keras API in TensorFlow
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
    specs.description = r"""Convolutional Neural Network (CNN) is a deep learning algorithm which can take in an input image, assign
        importance to various objects in the image and be able to differentiate one from the other. The
        architecture of a CNN is analogous to that of the connectivity pattern of Neurons in the Human Brain
        and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only
        in a restricted region of the visual field known as the Receptive Field. A collection of such fields
        overlap to cover the entire visual area. CNN is able to successfully capture the spatial and temporal
        dependencies in an image through the applicaiton of relevant filters. The architecture performs
        a better fitting to the image dataset due to the reduction in the number of parameters involved
        and reusability of weights. In other words, the network can be trained to understand the sophistication
        of the image better.
        \\
        \zNormalizationPerformed{KerasConvNetClassifier}
        \\
        In order to use this ROM, the \xmlNode{ROM} attribute \xmlAttr{subType} needs to
        be \xmlString{KerasConvNetClassifier}."""
    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'KerasConvNetClassifier'
    self.allowedLayers = self.basicLayers + self.kerasDict['kerasConvNetLayersList'] + self.kerasDict['kerasPoolingLayersList']

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  def _preprocessInputs(self,featureVals):
    """
      Perform input feature values before sending to ROM prediction
      @ In, featureVals, numpy.array, i.e. [shapeFeatureValue,numFeatures], values of features
      @ Out, featureVals, numpy.array, predicted values
    """
    shape = featureVals.shape
    if len(shape) == 2:
      featureVals = np.reshape(featureVals,(1, shape[0], shape[1]))
    return featureVals
