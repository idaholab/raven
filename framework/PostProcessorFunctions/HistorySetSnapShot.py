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
   - type = min, max, average, value

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
    self.pivotParameter   = None
    self.pivotVar = None
    self.pivotVal = None
    self.timeInstant = None

    self.numberOfSamples = None
    self.pivotParameter          = None
    self.interpolation   = None

    self.classifiers = {} #for "mixed" mode

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      tag = child.tag
      if tag =='type':
        self.type = child.text
      elif tag == 'numberOfSamples':
        self.numberOfSamples = int(child.text)
      elif tag == 'extension':
        self.extension = child.text
      elif tag == 'pivotParameter':
        self.pivotParameter = child.text
      elif tag == 'pivotVar':
        self.pivotVar = child.text
      elif tag == 'pivotVal':
        self.pivotVal = float(child.text)
      elif tag == 'timeInstant':
        self.timeInstant = int(child.text)
      elif self.type == 'mixed':
        entries = list(c.strip() for c in child.text.strip().split(','))
        if tag not in self.classifiers.keys():
          self.classifiers[tag] = []
        #min,max,avg need no additional information to run, so list is [varName, varName, ...]
        if tag in ['min','max','average']:
          self.classifiers[tag].extend(entries)
        #for now we remove timeSlice in mixed mode, until we recall why it might be desirable for a user
        #timeSlice requires the time at which to slice, so list is [ (varName,time), (varName,time), ...]
        #elif tag in ['timeSlice']:
        #  time = child.attrib.get('value',None)
        #  if time is None:
        #    self.raiseAnError('For "mixed" mode, must specify "value" as an attribute for each "timeSlice" node!')
        #  for entry in entries:
        #    self.classifiers[tag].append( (entry,float(time)) )
        #value requires the dependent variable and dependent value, so list is [ (varName,depVar,depVal), ...]
        elif tag == 'value':
          depVar = child.attrib.get('pivotVar',None)
          depVal = child.attrib.get('pivotVal',None)
          if depVar is None or depVal is None:
            self.raiseAnError('For "mixed" mode, must specify both "pivotVar" and "pivotVal" as an attribute for each "value" node!')
          for entry in entries:
            self.classifiers[tag].append( (entry,depVar,float(depVal)) )
        else:
          self.raiseAnError(IOError,'Unrecognized node for HistorySetSnapShot in "mixed" mode:',tag)
      elif tag !='method':
        self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child.tag) + ' is not recognized')

    needspivotParameter = ['average','timeSlice']
    if self.type in needspivotParameter or any(mode in self.classifiers.keys() for mode in needspivotParameter):
      if self.pivotParameter is None:
        self.raiseAnError(IOError,'"pivotParameter" is required for',needspivotParameter,'but not provided!')

    #sync if needed
    if self.type == 'timeSlice':
      #for syncing, need numberOfSamples, extension
      if self.numberOfSamples is None:
        self.raiseIOError(IOError,'When using "timeSlice" a "numberOfSamples" must be specified for synchronizing!')
      if self.extension is None:
        self.raiseAnError(IOError,'When using "timeSlice" an "extension" method must be specified for synchronizing!')
      if self.extension not in ['zeroed','extended']:
        self.raiseAnError(IOError,'Unrecognized "extension" method:',self.extension)
      #perform sync
      PostProcessorInterfaces = importlib.import_module("PostProcessorInterfaces")
      self.HSsyncPP = PostProcessorInterfaces.returnPostProcessorInterface('HistorySetSync',self)
      self.HSsyncPP.initialize(self.numberOfSamples,self.pivotParameter,self.extension,syncMethod='grid')

    if self.type not in set(['min','max','average','value','timeSlice','mixed']):
      self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor "' + str(self.name) + '" : type "%s" is not recognized' %self.type)


  def run(self,inputDic, pivotVal=None):
    """
     Method to post-process the dataObjects
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ Out, outputPSDic, dict, output dictionary
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')
    else:
      inputDic = inputDic[0]
      outputPSDic = {}
      outputHSDic = {}

      #for timeSlice we call historySetWindow
      if self.type == 'timeSlice':
        outputHSDic = self.HSsyncPP.run([inputDic])
        outputPSDic = historySetWindow(outputHSDic,self.timeInstant,self.pivotParameter)
        return outputPSDic
      #for other non-mixed methods we call historySnapShot
      elif self.type != 'mixed':
        outputPSDic = historySnapShot(inputDic,self.pivotVar,self.type,self.pivotVal,self.pivotParameter)
        return outputPSDic
      #mixed is more complicated: we pull out values by method instead of a single slice type
      #   We use the same methods to get slices, then pick out only the requested variables
      else:
        #establish the output dict
        outDict = {'data':{'input':{},'output':{}}}
        #replicate input space
        for var in inputDic['data']['input'].values()[0].keys():
          outDict['data']['input'][var] = np.array(list(inputDic['data']['input'][prefix][var] for prefix in inputDic['data']['input'].keys()))
        #replicate metadata
          outDict['metadata'] = inputDic['metadata']
        #loop over the methods requested to fill output space
        for method,entries in self.classifiers.items():
          #min, max take no special effort
          if method in ['min','max']:
            for var in entries:
              getDict = historySnapShot(inputDic,var,method)
              outDict['data']['output'][var] = getDict['data']['output'][var]
          #average requires the pivotParameter
          elif method == 'average':
            for var in entries:
              getDict = historySnapShot(inputDic,var,method,tempID=self.pivotParameter)
              outDict['data']['output'][var] = getDict['data']['output'][var]
          #timeSlice requires the time value
          #functionality removed for now until we recall why it's desirable
          #elif method == 'timeSlice':
          #  for var,time in entries:
          #    getDict = historySetWindow(inputDic,time,self.pivotParameter)
          #value requires the dependent variable and value
          elif method == 'value':
            for var,depVar,depVal in entries:
              getDict = historySnapShot(inputDic,depVar,method,pivotVal=depVal)
              outDict['data']['output'][var] = getDict['data']['output'][var]
        return outDict


def historySnapShot(inputDic, pivotVar, snapShotType, pivotVal=None, tempID = None):
  """
  Method do to compute a conversion from HistorySet to PointSet using the methods: min,max,average,value
  @ In, vars, dict, it is an historySet
  @ In, pivotVar,  string, variable considered
  @ In, pivotVal,  double, value associated to the variable considered
  @ In, snapShotType, string, type of snapShot: min, max, average, value
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
      idx = inputDic['data']['output'][history][pivotVar].returnIndexFirstPassage(pivotVal)
      if inputDic['data']['output'][history][pivotVar][idx]>pivotVal:
        intervalFraction = (pivotVal-inputDic['data']['output'][history][pivotVar][idx-1])/(inputDic['data']['output'][history][pivotVar][idx]-inputDic['data']['output'][history][pivotVar][idx-1])
        for keys in outVars:
          value = inputDic['data']['output'][history][keys][idx-1] + (inputDic['data']['output'][history][keys][idx]-inputDic['data']['output'][history][keys][idx-1])*intervalFraction
          outputDic['data']['output'][keys] = np.append(outputDic['data']['output'][keys],value)
      else:
        intervalFraction = (pivotVal-inputDic['data']['output'][history][pivotVar][idx])/(inputDic['data']['output'][history][pivotVar][idx+1]-inputDic['data']['output'][history][pivotVar][idx])
        for keys in outVars:
          value = inputDic['data']['output'][history][keys][idx] + (inputDic['data']['output'][history][keys][idx+1]-inputDic['data']['output'][history][keys][idx])*intervalFraction
          outputDic['data']['output'][keys] = np.append(outputDic['data']['output'][keys],value)
    elif snapShotType == 'average':
      for keys in outVars:
        cumulative=0.0
        for t in range(1,len(inputDic['data']['output'][history][tempID])):
          cumulative += (inputDic['data']['output'][history][keys][t] + inputDic['data']['output'][history][keys][t-1]) / 2.0 * (inputDic['data']['output'][history][tempID][t] - inputDic['data']['output'][history][tempID][t-1])
        value = cumulative / (inputDic['data']['output'][history][tempID][-1] - inputDic['data']['output'][history][tempID][0])
        outputDic['data']['output'][keys] = np.append(outputDic['data']['output'][keys],value)

  return outputDic


def historySetWindow(inputDic,timeStepID,pivotParameter):
  """
  Method do to compute a conversion from HistorySet to PointSet using the temporal slice of the historySet
  @ In, inputDic, dict, it is an historySet
  @ In, timeStepID, int, number of time sample of each history
  @ In, pivotParameter, string, ID name of the temporal variable
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
  outVars.remove(pivotParameter)

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
