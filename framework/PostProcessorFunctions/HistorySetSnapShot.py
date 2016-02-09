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
        self.pivotVal = child.text
      elif child.tag == 'timeInstant':
        self.timeInstant = int(child.text)
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child.tag) + ' is not recognized')
              
    if self.type == 'timeSlice':
      PostProcessorInterfaces = importlib.import_module("PostProcessorInterfaces")
      self.HSsyncPP = PostProcessorInterfaces.returnPostProcessorInterface('HistorySetSync',self)
      self.HSsyncPP.initialize(self.numberOfSamples,self.timeID,self.extension)
    
    if self.type not in set(['min','max','avg','value','timeSlice']):
      self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor ' + str(self.name) + ' : type is not recognized')


  def run(self,inputDic, pivotVal=None):
    """
     Method to post-process the dataObjects
     @ In,  inputDic , dictionary
     @ Out, outputPSDic, dictionary
    """
    outputPSDic = {}
    outputHSDic = {}

    if self.type == 'timeSlice':
      outputHSDic = self.HSsyncPP.run(inputDic)
      outputPSDic = historySetWindow(outputHSDic,self.timeInstant,self.timeID)
    else:
      outputPSDic = historySnapShot(inputDic,self.pivotVar,self.type,self.pivotVal,self.timeID)
    return outputPSDic


#def historySnapShot(vars, pivotVar, snapShotType, pivotVal=None, tempID = None):
def historySnapShot(**kwargs):
  for item in kwargs:
    if item=='vars':
      vars = kwargs[item]
    if item=='pivotVar':
      pivotVar = kwargs[item]
    if item=='snapShotType':
      snapShotType = kwargs[item]
    if item=='pivotVal':
      pivotVal = kwargs[item]
    if item=='tempID':
      tempID = kwargs[item]
          
  newVars={}

  if snapShotType == 'min':
    index = np.argmin(vars[pivotVar])
    for keys in vars.keys():
      newVars[keys] = var[keys][index]

  elif snapShotType == 'max':
    index = np.argmax(vars[pivotVar])
    for keys in vars.keys():
      newVars[keys] = var[keys][index]

  elif snapShotType == 'value':
    if pivotVal==None:
      self.raiseAnError(RuntimeError,'type ' + snapShotType + ' is not a valid type with variable pivotVal=None. Post-processor: HistorySetSnapShot')
    else:
      idx = np.argmin(np.abs(vars[pivotVar] - pivotVal))
      if pivotVar[idx]>pivotVal:
        intervalFraction = (val-var[pivotVar][idx-1])/(var[pivotVar][idx]-var[pivotVar][idx-1])
        for keys in vars.keys():
          newVars[keys] = var[keys][idx-1] + (var[keys][idx]-var[keys][idx-1])*intervalFraction
      else:
        intervalFraction = (val-var[pivotVar][idx])/(var[pivotVar][idx+1]-var[pivotVar][idx])
        for keys in vars.keys():
          newVars[keys] = var[keys][idx] + (var[keys][idx+1]-var[keys][idx])*intervalFraction

  elif snapShotType == 'avg':
     for keys in vars.keys():
       cumulative=0.0
       for t in range(1,vars[keys].shape()):
         cumulative += (vars[keys][t] + vars[keys][t-1]) / 2.0 * (vars[tempID][t] - vars[tempID][t-1])
       newVars[keys] = cumulative / (vars[tempID][-1] - vars[tempID][0])

  return newVars


def historySetWindow(inputDic,timeStepID,timeID):
  """
  Method do to compute
  @ In, inputDic is an historySet
  @ In, timeStepID, int, number of time samples of each history
  @ Out, outDic, dictionary, it contains the temporal slice of all histories
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

