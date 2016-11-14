
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

    self.pivotParameter       = None
    ''' pivotParameter identify the ID of the temporal variable in the data set; it is used so that in the
    conversion the time array is not inserted since it is not needed (all histories have same length)'''
    self.features     = 'all'


  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'pivotParameter':
        self.pivotParameter = child.text
      elif child.tag == 'features':
        self.features = child.text.split(',')
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if self.pivotParameter == None:
      self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : pivotParameter is not specified')


  def run(self,inputDic):
    """
    This method performs the actual transformation of the data object from history set to point set
      @ In, inputDic, dict, input dictionary
      @ Out, outputDic, dict, output dictionary
    """
    outputDic={}
    outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
    outputDic['data'] = {}
    outputDic['data']['output'] = {}
    outputDic['data']['input']  = {}

    #generate the input part of the output dictionary
    inputVars = inputDic['data']['input'][inputDic['data']['input'].keys()[0]].keys()
    for inputVar in inputVars:
      outputDic['data']['input'][inputVar] = np.empty(0)

    for hist in inputDic['data']['input']:
      for inputVar in inputVars:
        outputDic['data']['input'][inputVar] = np.append(outputDic['data']['input'][inputVar], copy.deepcopy(inputDic['data']['input'][hist][inputVar]))

    #generate the output part of the output dictionary
    if self.features == 'all':
      self.features = []
      historiesID = inputDic['data']['output'].keys()
      self.features = inputDic['data']['output'][historiesID[0]].keys()

    referenceHistory = inputDic['data']['output'].keys()[0]
    referenceTimeAxis = inputDic['data']['output'][referenceHistory][self.pivotParameter]
    for hist in inputDic['data']['output']:
      if (str(inputDic['data']['output'][hist][self.pivotParameter]) == str(referenceTimeAxis)):
        pass
      else:
        self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : one or more histories in the historySet have different time scale')

    tempDict = {}

    for hist in inputDic['data']['output'].keys():
      tempDict[hist] = np.empty(0)
      for feature in self.features:
        if feature != self.pivotParameter:
          tempDict[hist] = np.append(tempDict[hist],copy.deepcopy(inputDic['data']['output'][hist][feature]))
      length = np.size(tempDict[hist])

    for hist in tempDict:
      if np.size(tempDict[hist]) != length:
        self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : one or more histories in the historySet have different length')

    for key in range(length):
      if key != self.pivotParameter:
        outputDic['data']['output'][key] = np.empty(0)

    for hist in inputDic['data']['output'].keys():
      for key in outputDic['data']['output'].keys():
        outputDic['data']['output'][key] = np.append(outputDic['data']['output'][key], copy.deepcopy(tempDict[hist][int(key)]))

    self.transformationSettings['vars'] = copy.deepcopy(self.features)
    self.transformationSettings['vars'].remove(self.pivotParameter)
    self.transformationSettings['timeLength'] = int(length/len(self.transformationSettings['vars']))
    self.transformationSettings['timeAxis'] = inputDic['data']['output'][1][self.pivotParameter]
    self.transformationSettings['dimID'] = outputDic['data']['output'].keys()

    return outputDic



  def _inverse(self,inputDic):

    data = {}
    for hist in inputDic.keys():
      data[hist]= {}
      tempData = inputDic[hist].reshape((len(self.transformationSettings['vars']),self.transformationSettings['timeLength']))
      for index,var in enumerate(self.transformationSettings['vars']):
        data[hist][var] = tempData[index,:]
      data[hist][self.pivotParameter] = self.transformationSettings['timeAxis']

    return data


