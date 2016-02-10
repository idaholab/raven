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


class HS2PS(PostProcessorInterfaceBase):
  """
   This Post-Processor performs the conversion from HistorySet to PointSet
   The conversion is made so that each history H is converted to a single point P.
   Assume that each history H is a dict of n output variables x_1=[...],x_n=[...], then the resulting point P is as follows; P=[x_1,...,x_n]
   Note!!!! Here it is assumed that all histories have been sync so that they have the same length, start point and end point.
            If you are not sure, do a pre-processing the the original history set
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

    self.timeID       = None
    self.features     = 'all'

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'timeID':
        self.timeID = child.text
      elif child.tag == 'features':
        self.features = child.text.split(',')
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if self.timeID == None:
      self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : timeID is not specified')


  def run(self,inputDic):
    """
    This method is transparent: it passes the inputDic directly as output
      @ In, inputDic, dict, input dictionary
      @ Out, outputDic, dict, output dictionary
    """
    outputDic={}
    outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
    outputDic['data'] = {}
    outputDic['data']['output'] = {}
    outputDic['data']['input']  = {}

    ''' generate the input part of the output dictionary'''
    inputVars = inputDic['data']['input'][inputDic['data']['input'].keys()[0]].keys()
    for inputVar in inputVars:
      outputDic['data']['input'][inputVar] = np.empty(0)

    for hist in inputDic['data']['input']:
      for inputVar in inputVars:
        outputDic['data']['input'][inputVar] = np.append(outputDic['data']['input'][inputVar], copy.deepcopy(inputDic['data']['input'][hist][inputVar]))

    ''' generate the output part of the output dictionary'''
    if self.features == 'all':
      self.features = []
      historiesID = inputDic['data']['output'].keys()
      self.features = inputDic['data']['output'][historiesID[0]].keys()

    tempDict = {}

    for hist in inputDic['data']['output'].keys():
      tempDict[hist] = np.empty(0)
      for feature in self.features:
        if feature != self.timeID:
          tempDict[hist] = np.append(tempDict[hist],copy.deepcopy(inputDic['data']['output'][hist][feature]))
      length = np.size(tempDict[hist])

    for hist in tempDict:
      if np.size(tempDict[hist]) != length:
        self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : one or more histories in the historySet have different length')

    for key in range(length):
      if key != self.timeID:
        outputDic['data']['output'][str(key)] = np.empty(0)

    for hist in inputDic['data']['output'].keys():
      for key in outputDic['data']['output'].keys():
        outputDic['data']['output'][key] = np.append(outputDic['data']['output'][key], copy.deepcopy(tempDict[hist][int(key)]))

    return outputDic


