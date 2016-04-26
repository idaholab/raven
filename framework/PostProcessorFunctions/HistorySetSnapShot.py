"""
Created on October 28, 2015

"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase

import os
import numpy as np
from scipy import interpolate
import copy
import importlib

import HistorySetSync as HSS

class HistorySetSnapShot(PostProcessorInterfaceBase):
  """
   This Post-Processor performs the conversion from HistorySet to PointSet
   The conversion is made so that each history H is converted to a single point P.
   Assume that each history H is a dict of n output variables x_1=[...],x_n=[...], then the resulting point P can be as follows accordingly to the specified type:
   - type = timeSlice: at time instant t: P=[x_1[t],...,x_n[t]]
   - type = min, max, avg, value

  """

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """

    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'PointSet'

    self.type     = None
    self.timeID   = None
    self.pivotVar = None
    self.pivotVal = None
    self.timeInstant = None

    self.numberOfSamples = None
    self.timeID          = None
    self.interpolation   = None


  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag =='type':
        self.type = child.text
      elif child.tag == 'numberOfSamples':
        self.numberOfSamples = int(child.text)
      elif child.tag == 'extension':
        self.extension = child.text
      elif child.tag == 'timeID':
        self.timeID = child.text
      elif child.tag == 'pivotVar':
        self.pivotVar = child.text
      elif child.tag == 'pivotVal':
        self.pivotVal = float(child.text)
      elif child.tag == 'timeInstant':
        self.timeInstant = int(child.text)
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child.tag) + ' is not recognized')

    if self.type == 'timeSlice':
      PostProcessorInterfaces = importlib.import_module("PostProcessorInterfaces")
      self.HSsyncPP = PostProcessorInterfaces.returnPostProcessorInterface('HistorySetSync',self)
      self.HSsyncPP.initialize(self.numberOfSamples,self.timeID,self.extension,syncMethod='grid')

    if self.type not in set(['min','max','avg','value','timeSlice']):
      self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor ' + str(self.name) + ' : type is not recognized')


  def run(self,inputDic, pivotVal=None):
    """
     Method to post-process the dataObjects
     @ In,  inputDic, dict, input dictionary
     @ Out, outputPSDic, dict, output dictionary
    """
    outputPSDic = {}
    outputHSDic = {}

    if self.type == 'timeSlice':
      outputHSDic = self.HSsyncPP.run(inputDic)
      outputPSDic = historySetWindow(outputHSDic,self.timeInstant,self.timeID)
    else:
      outputPSDic = historySnapShot(inputDic,self.pivotVar,self.type,self.pivotVal,self.timeID)
    return outputPSDic


def historySnapShot(inputDic, pivotVar, snapShotType, pivotVal=None, tempID = None):
  """
  Method do to compute a conversion from HistorySet to PointSet using the methods: min,max,avg,value
  @ In, vars, dict, it is an historySet
  @ In, pivotVar,  string, variable considered
  @ In, pivotVal,  double, value associated to the variable considered
  @ In, snapShotType, string, type of snapShot: min, max, avg, value
  @ Out, outputDic, dict, it contains the temporal slice of all histories
  """

  outputDic={}
  outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
  outputDic['data'] = {}
  outputDic['data']['input'] = {}
  outputDic['data']['output'] = {}

  historiesID = inputDic['data']['output'].keys()
  inVars  = inputDic['data']['input'][historiesID[0]].keys()
  outVars = inputDic['data']['output'][historiesID[0]].keys()
  #if tempID != None:
  #  outVars.remove(tempID)

  for var in inVars:
    outputDic['data']['input'][var] = np.zeros(0)

  for var in outVars:
    outputDic['data']['output'][var] = np.zeros(0)

  for history in inputDic['data']['input']:
    for key in inVars:
      outputDic['data']['input'][key]  = np.append(outputDic['data']['input'][key],copy.deepcopy(inputDic['data']['input'][history][key]))

  for history in inputDic['data']['output']:
    if snapShotType == 'min':
      idx = inputDic['data']['output'][history][pivotVar].returnIndexMin()
      for vars in outVars:
        outputDic['data']['output'][vars] = np.append(outputDic['data']['output'][vars] , copy.deepcopy(inputDic['data']['output'][history][vars][idx]))
    elif snapShotType == 'max':
      idx = inputDic['data']['output'][history][pivotVar].returnIndexMax()
      for vars in outVars:
        outputDic['data']['output'][vars] = np.append(outputDic['data']['output'][vars] , copy.deepcopy(inputDic['data']['output'][history][vars][idx]))
    elif snapShotType == 'value':
      idx = inputDic['data']['output'][history][pivotVar].returnIndex(pivotVal)
      if inputDic['data']['output'][history][pivotVar][idx]>pivotVal:
        intervalFraction = (pivotVal-inputDic['data']['output'][history][pivotVar][idx-1])/(outputDic['data']['output'][history][pivotVar][idx]-inputDic['data']['output'][history][pivotVar][idx-1])
        for keys in outVars:
          value = inputDic['data']['output'][history][keys][idx-1] + (inputDic['data']['output'][history][keys][idx]-inputDic['data']['output'][history][keys][idx-1])*intervalFraction
          outputDic['data']['output'][keys] = np.append(outputDic['data']['output'][keys],value)
      else:
        intervalFraction = (pivotVal-inputDic['data']['output'][history][pivotVar][idx])/(inputDic['data']['output'][history][pivotVar][idx+1]-inputDic['data']['output'][history][pivotVar][idx])
        for keys in outVars:
          value = inputDic['data']['output'][history][keys][idx] + (inputDic['data']['output'][history][keys][idx+1]-inputDic['data']['output'][history][keys][idx])*intervalFraction
          outputDic['data']['output'][keys] = np.append(outputDic['data']['output'][keys],value)
    elif snapShotType == 'avg':
       for keys in outVars:
         cumulative=0.0
         for t in range(1,len(inputDic['data']['output'][history][tempID])):
           cumulative += (inputDic['data']['output'][history][keys][t] + inputDic['data']['output'][history][keys][t-1]) / 2.0 * (inputDic['data']['output'][history][tempID][t] - inputDic['data']['output'][history][tempID][t-1])
         value = cumulative / (inputDic['data']['output'][history][tempID][-1] - inputDic['data']['output'][history][tempID][0])
         outputDic['data']['output'][keys] = np.append(outputDic['data']['output'][keys],value)

  return outputDic


def historySetWindow(inputDic,timeStepID,timeID):
  """
  Method do to compute a conversion from HistorySet to PointSet using the temporal slice of the historySet
  @ In, inputDic, dict, it is an historySet
  @ In, timeStepID, int, number of time sample of each history
  @ In, timeID, string, ID name of the temporal variable
  @ Out, outDic, dict, it contains the temporal slice of all histories
  """

  outputDic={}
  outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
  outputDic['data'] = {}
  outputDic['data']['input'] = {}
  outputDic['data']['output'] = {}

  historiesID = inputDic['data']['output'].keys()
  inVars  = inputDic['data']['input'][historiesID[0]].keys()
  outVars = inputDic['data']['output'][historiesID[0]].keys()
  outVars.remove(timeID)

  for var in inVars:
    outputDic['data']['input'][var] = np.zeros(0)

  for var in outVars:
    outputDic['data']['output'][var] = np.zeros(0)

  for history in inputDic['data']['input']:
    for key in inVars:
      outputDic['data']['input'][key]  = np.append(outputDic['data']['input'][key],copy.deepcopy(inputDic['data']['input'][history][key]))

  for history in inputDic['data']['output']:
    for key in outVars:
      outputDic['data']['output'][key] = np.append(outputDic['data']['output'][key],copy.deepcopy(inputDic['data']['output'][history][key][timeStepID]))

  return outputDic

