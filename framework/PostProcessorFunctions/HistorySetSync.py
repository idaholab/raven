"""
Created on October 28, 2015

"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import itertools
import numpy as np
#External Modules End--------------------------------------------------------------------------------

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase


class HistorySetSync(PostProcessorInterfaceBase):
  """
   This function does things.
   @ In : inputDict
   @ Out: outputDict
  """

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'HistorySet'
    
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
      if child.tag == 'numberOfSamples':
        self.numberOfSamples = int(child.text)
      elif child.tag == 'timeID':
        self.timeID = child.text
      elif child.tag == 'extension':
        self.extension = child.text
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')
    
    if not isinstance(self.numberOfSamples, int):
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : number of samples is not correctly specified (either not specified or not integer)')
    if self.timeID == None:
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : timeID is not specified')
    if self.extension == None or not (self.extension == 'zeroed' or self.extension == 'extended'):
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : extension type is not correctly specified (either not specified or not one of its possible allowed values: zeroed or extended)')
    
    
  def run(self,inputDic):
    """
    This method is transparent: it passes the inputDic directly as output
    """
    outputDic={}
    outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
    outputDic['data'] = {}
    outputDic['data']['input'] = copy.deepcopy(inputDic['data']['input'])
    outputDic['data']['output'] = {}
    
    maxEndTime = []
    minInitTime = []
    for hist in inputDic['data']['output']:
      maxEndTime.append(inputDic['data']['output'][hist][self.timeID][-1])
      minInitTime.append(inputDic['data']['output'][hist][self.timeID][0])
    maxTime = max(maxEndTime)  
    minTime = min(minInitTime)
    
    newTime = np.linspace(minTime,maxTime,self.numberOfSamples)
    
    for hist in inputDic['data']['output']:
      outputDic['data']['output'][hist] = self.resampleHist(inputDic['data']['output'][hist],newTime)      
      
    return outputDic


  def resampleHist(self, vars, newTime):
    newVars={}
    for key in vars.keys():
      if key != self.timeID:
        newVars[key]=np.zeros(newTime.size)
        pos=0
        for newT in newTime:  
          if newT<vars[self.timeID][0]: 
            if self.extension == 'extended':
              newVars[key][pos] = vars[key][0] 
            elif self.extension == 'zeroed':
              newVars[key][pos] = 0.0
          elif newT>vars[self.timeID][-1]:
            if self.extension == 'extended':
              newVars[key][pos] = vars[key][-1]             
            elif self.extension == 'zeroed':
              newVars[key][pos] = 0.0
          else:
            index = np.searchsorted(vars[self.timeID],newT)
            newVars[key][pos] = vars[key][index-1] + (vars[key][index]-vars[key][index-1])/(vars[self.timeID][index]-vars[self.timeID][index-1])*(newT-vars[self.timeID][index-1]) 
          pos=pos+1
    
    newVars[self.timeID] = copy.deepcopy(newTime)
    return newVars 
     

      
    
    
       