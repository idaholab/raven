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
    # object of message handler
    self.messageHandler          = messageHandler
    # instance of given Metric
    self.estimator                = estimator
    # True if the instance of given metric, i.e. 'estimator', can handle time-dependent data, else False
    self.canHandleDynamicData = self.estimator.isDynamic()
    # True if the instance of given metric, i.e. 'estimator', can handle pairwise data, else False
    self.canHandlePairwiseData = self.estimator.isPairwise()

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
    paramDict['Handle dynamic data'] = self.canHandleDynamicData
    paramDict['Handle pairwise data'] = self.canHandlePairwiseData
    paramDict['Metric name'] = self.estimator.name
    return paramDict

  def evaluatePairwise(self, pairedData):
    """
      Method to compute the the metric between each pair of rows of matrices in pairedData
      @ In, pairedData, tuple, (featureValues, targetValues), both featureValues and targetValues
        are 2D numpy array with the same number of columns. For example, featureValues with shape
        (numRealizations1,numParameters), targetValues with shape (numRealizations2, numParameters)
      @ Out, output, numpy.ndarray, 2D array, with shape (numRealizations1,numRealization2)
    """
    assert(type(pairedData).__name__ == 'tuple'), "The paired data is not a tuple!"
    if not self.canHandlePairwiseData:
      self.raiseAnError(IOError, "The metric", self.estimator.name, "can not handle pairwise data")
    feat, targ = pairedData
    output = self.estimator.evaluate(feat,targ)
    return output

  def evaluate(self,pairedData, weights = None, multiOutput='mean'):
    """
      Method to perform the evaluation of given paired data
      @ In, pairedData, tuple, ((featureValues, probabilityWeight), (targetValues, probabilityWeight)), both
        featureValues and targetValues  have the same shape (numRealizations,  numHistorySteps)
      @ In, weights, array_like (numpy.ndarray or list), optional,  An array of weights associated with the pairedData
      @ In, multiOutput, string, optional, 'mean', 'max', 'min' or 'raw_values'
      @ Out, output, numpy.ndarray, 1D array, processed output from the estimator
    """
    assert(type(pairedData).__name__ == 'tuple')
    # Error check for input data
    dynamicOutput = []
    for pData in pairedData:
      if not self.estimator.acceptsDistribution and isinstance(pData, Distributions.Distribution):
        self.raiseAnError(IOError, "Distribution is provided, but the metric ", self.estimator.name, " can not handle it!")
    feat, targ = pairedData
    if isinstance(feat, Distributions.Distribution) and isinstance(targ, Distributions.Distribution):
      out = self.estimator.evaluate(feat, targ)
      dynamicOutput.append(out)
    elif isinstance(feat, Distributions.Distribution):
      targVals = np.asarray(targ[0])
      for hist in range(targVals.shape[1]):
        if targ[1] is not None:
          assert(len(targVals[:,hist]) == len(targ[1]))
          targIn = (targVals[:,hist], targ[1])
        else:
          targIn = targVals[:,hist]
        out = self.estimator.evaluate(feat, targIn)
        dynamicOutput.append(out)
    elif isinstance(targ, Distributions.Distribution):
      featVals = np.asarray(feat[0])
      for hist in range(featVals.shape[1]):
        if feat[1] is not None:
          assert(len(featVals[:,hist]) == len(feat[1]))
          featIn = (featVals[:,hist], feat[1])
        else:
          featIn = featVals[:,hist]
        out = self.estimator.evaluate(featIn, targ)
        dynamicOutput.append(out)
    elif self.estimator.type in ['CDFAreaDifference', 'PDFCommonArea']:
      featVals = np.asarray(feat[0])
      targVals = np.asarray(targ[0])
      for hist in range(featVals.shape[1]):
        if feat[1] is not None:
          featIn = (featVals[:,hist], feat[1])
        else:
          featIn = featVals[:,hist]
        if targ[1] is not None:
          assert(len(targVals[:,hist]) == len(targ[1]))
          targIn = (targVals[:,hist], targ[1])
        else:
          targIn = targVals[:,hist]
        out = self.estimator.evaluate(featIn, targIn)
        dynamicOutput.append(out)
    else:
      featVals = np.asarray(feat[0])
      targVals = np.asarray(targ[0])
      assert(featVals.shape[0] == targVals.shape[0])
      if feat[1] is not None:
        dataWeight = np.asarray(feat[1])
        assert(featVals.shape[0] == dataWeight.shape[0])
      else:
        dataWeight = None
      # FIXME: Currently, we only use the weights of given features to compute the metric, this
      # can be biased or uncorrect. The correct way is to use the joint probability weight.
      # This needs to be improved in the future when RAVEN can handle the joint probability weight correctly.
      if self.canHandleDynamicData:
        dynamicOutput = self.estimator.evaluate(featVals, targVals, dataWeight)
      else:
        for hist in range(featVals.shape[1]):
          out = self.estimator.evaluate(featVals[:,hist], targVals[:,hist], dataWeight)
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
