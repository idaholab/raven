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
Created on Noverber 16, 2017

@author: wangc
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import inspect
import abc
import copy
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils import mathUtils
from utils import utils
import MessageHandler
import Distributions
#Internal Modules End--------------------------------------------------------------------------------
class MetricDistributor(utils.metaclass_insert(abc.ABCMeta,BaseType),MessageHandler.MessageUser):
  """
    This class represents an interface with all the metrics algorithms
    It is a utility class needed to hide the discernment between time-dependent and static
    metrics
  """
  def __init__(self, estimator, messageHandler):
    """
      A constructor
      @ In, estimator, instance of given metric
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag                = 'MetricDistributor'
    self.messageHandler          = messageHandler
    self.estimator                = estimator
    self.canHandleDynamicData = self.estimator.isDynamic()

  def __getstate__(self):
    """
      This function return the state of the class
      @ In, None
      @ Out, state, dict, it contains all the information needed by the class to be initialized
    """
    state = self.__dict__.copy()
    return state

  def __setstate__(self, newState):
    """
      Initialize the class with the data contained in newstate
      @ In, newState, dict, it contains all the information needed by the class to be initialized
      @ Out, None
    """
    self.__dict__.update(newState)
    return newState

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    return paramDict

  def evaluate(self,pairedData, weights = None, multiOutput='mean'):
    """
      Method to perform the evaluation of given paired data
      @ In, pairedData, tuple, ((featureValues, probabilityWeight), (targetValues, probabilityWeight)), both
        featureValues and targetValues  have the same shape (numRealizations,  numHistorySteps)
      @ In, weights, None or array_like, An array of weights associated with the pairedData
      @ In, multiOutput, string, 'mean', 'max', 'min' or 'raw_values'
      @ Out, output, numpy.array, 1D array, processed output from the estimator
    """
    # FIXME: check the consistence of provided data
    assert(type(pairedData).__name__ == 'tuple', "The paired data is not a tuple!")
    # Error check for input data
    for i in range(len(pairedData)):
      if not self.estimator.acceptsDistribution and isinstance(pairedData[i], Distributions.Distribution):
        self.raiseAnError(IOError, "Distribution is provided, but the metric ", self.estimator.name, " can not handle it!")
      if isinstance(pairedData[i], Distributions.Distribution):
        self.raiseAnError(IOError, "Not implemented yet!")
    featureValues = np.asarray(pairedData[0][0])
    featureWeights = np.asarray(pairedData[0][1])
    targetValues = np.asarray(pairedData[1][0])
    dynamicOutput = []
    # FIXME: Currently, we only use the weights of given features to compute the metric, this
    # can be biased or uncorrect. The correct way is to use the joint probability weight.
    # This needs to be improved in the future when RAVEN can handle the joint probability weight correctly.
    if self.canHandleDynamicData:
      dynamicOutput = self.estimator.evaluate(featureValues, targetValues,featureWeights)
    else:
      for hist in range(len(featureValues)):
        out = self.estimator.evaluate(featureValues[:,hist],targetValues[:,hist],featureWeights)
        dynamicOutput.append(out)
    if multiOutput == 'mean':
      output = [np.average(dynamicOutput, weights = weights)]
    elif multiOutput == 'max':
      output = [np.amax(dynamicOutput)]
    elif multiOutput == 'min':
      output = [np.amin(dynamicOutput)]
    elif multiOutput == 'raw_values':
      output = dynamicOutput
    else:
      self.raiseAnError(IOError, "multiOutput: ", multiOutput, " is not acceptable! Please use 'mean', 'max', 'min' or 'full'")
    output = np.asarray(output)
    return output

__interfaceDict                         = {}
__interfaceDict['MetricDistributor'      ] = MetricDistributor
__base                                  = 'Distributor'

def returnInstance(distributorType, estimator, caller):
  """
    This function return an instance of the request model type
    @ In, distributorType, string, string representing the class to retrieve
    @ In, estimator, list of instance of given metrics
    @ In, caller, instance, object that will share its messageHandler instance
    @ Out, returnInstance, instance, an instance of this class
  """
  try:
    return __interfaceDict[distributorType](estimator, caller.messageHandler)
  except KeyError as ae:
    caller.raiseAnError(NameError,'not known '+__base+' type '+str(distributorType))

def returnClass(distributorType,caller):
  """
    This function return an instance of the request model type
    @ In, distributorType, string, string representing the class to retrieve
    @ In, caller, instnace, object that will share its messageHandler instance
    @ Out, returnClass, the class definition
  """
  try:
    return __interfaceDict[distributorType]
  except KeyError:
    caller.raiseAnError(NameError,'not known '+__base+' type '+distributorType)
