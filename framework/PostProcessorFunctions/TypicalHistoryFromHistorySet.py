'''
  Created on Feb 17, 2016

'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
import numpy as np
import copy
import mathUtils

class TypicalHistoryFromHistorySet(PostProcessorInterfaceBase):
  """
    This class forms a typical history from a history set
    The methodology can be found at:
    S. WIlcox and W. Marion, "User Manual for TMY3 Data Sets," Technical Report, NREL/TP-581-43156,
    National Renewable Energy Laboratory Golden, CO, May 2008
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
    if not hasattr(self, 'pivotParameterID'):   self.pivotParameterID = 'Time'
    if not hasattr(self, 'outputLen'):          self.outputLen = None

  def run(self,inputDic):
    """
      @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, dictionary which contains the data to be collected by output DataObject
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'TypicalHistoryFromHistorySet Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')
    else:
      inputDict = inputDic[0]['data']
      self.features = inputDict['output'][inputDict['output'].keys()[0]].keys()
      if self.pivotParameterID in self.features:
        self.features.remove(self.pivotParameterID)

      if self.outputLen is None: self.outputLen = np.asarray(inputDict['output'][inputDict['output'].keys()[0]][self.pivotParameterID])[-1]

      tempInData = {}
      keyNewH = 0
      for keyH in inputDict['output'].keys():
        localPivotParameter = np.asarray(inputDict['output'][keyH][self.pivotParameterID])
        if self.outputLen >= localPivotParameter[-1]:
          tempInData[keyNewH] = inputDict['output'][keyH]
          keyNewH += 1
        else:
          startL, endL = 0, self.outputLen
          while endL <= localPivotParameter[-1]:
            tempInDataH = {}
            extractCondition = (localPivotParameter>=startL) * (localPivotParameter<=endL)
            tempInDataH[self.pivotParameterID] = np.extract(extractCondition, localPivotParameter)-startL
            for keyF in self.features:
              tempInDataH[keyF] = np.extract(extractCondition, inputDict['output'][keyH][keyF])
            tempInData[keyNewH] = copy.deepcopy(tempInDataH)
            keyNewH += 1
            startL = copy.copy(endL)
            endL += self.outputLen

      inputDict['output'] = copy.deepcopy(tempInData)
      self.numHistory = len(inputDict['output'].keys())
      self.pivotParameter = np.asarray(inputDict['output'][inputDict['output'].keys()[0]][self.pivotParameterID])

      self.subsequence = {}
      startLocation, n = 0, 0
      while True:
        subsequenceLength = self.subseqLen[n % len(self.subseqLen)]
        if startLocation + subsequenceLength < self.pivotParameter[-1]:
          self.subsequence[n] = [startLocation, startLocation+subsequenceLength]
        else:
          self.subsequence[n] = [startLocation, self.pivotParameter[-1]]
          break
        startLocation += subsequenceLength
        n+= 1
      subKeys = self.subsequence.keys()
      subKeys.sort()

      tempData = {'all':{}}
      for keyF in self.features:
        tempData['all'][keyF] = np.array([])
        for keyH in inputDict['output'].keys():
          tempData['all'][keyF] = np.concatenate((tempData['all'][keyF],inputDict['output'][keyH][keyF]))
      for keySub in subKeys:
        tempData[keySub] = {}
        extractCondition = (self.pivotParameter>=self.subsequence[keySub][0]) * (self.pivotParameter<self.subsequence[keySub][1])
        tempData[keySub][self.pivotParameterID] = np.extract(extractCondition, self.pivotParameter)
        if self.pivotParameter[-1] == self.subsequence[keySub][1]:
          tempData[keySub][self.pivotParameterID] = np.concatenate((tempData[keySub][self.pivotParameterID], np.asarray([self.pivotParameter[-1]])))

        for keyF in self.features:
          tempData[keySub][keyF] = np.zeros(shape=(self.numHistory,len(tempData[keySub][self.pivotParameterID])))
          for cnt, keyH in enumerate(inputDict['output'].keys()):
            if self.pivotParameter[-1] == self.subsequence[keySub][1]:
              tempData[keySub][keyF][cnt,0:-1] = np.extract(extractCondition, inputDict['output'][keyH][keyF])
              tempData[keySub][keyF][cnt,-1] = inputDict['output'][keyH][keyF][-1]
            else:
              tempData[keySub][keyF][cnt,:] = np.extract(extractCondition, inputDict['output'][keyH][keyF])

      tempCDF = {'all':{}}
      for keyF in self.features:
        numBins , binEdges = mathUtils.numBinsDraconis(tempData['all'][keyF])
        tempCDF['all'][keyF] = self.__computeECDF(tempData['all'][keyF], binEdges)# numBins, dataRange)
        for keySub in subKeys:
          if keySub not in tempCDF.keys(): tempCDF[keySub] = {}
          tempCDF[keySub][keyF] = np.zeros(shape=(self.numHistory,numBins))
          for cnt in range(tempData[keySub][keyF].shape[0]):
            tempCDF[keySub][keyF][cnt,:] = self.__computeECDF(tempData[keySub][keyF][cnt,:], binEdges)#numBins, dataRange)

      tempTyp = {}
      for keySub in subKeys:
        tempTyp[keySub] = {}
        tempTyp[keySub][self.pivotParameterID] = tempData[keySub][self.pivotParameterID]
        d = np.inf
        for cnt, keyH in enumerate(inputDict['output'].keys()):
          FS = 0
          for keyF in self.features:
            FS += self.__computeDist(tempCDF['all'][keyF],tempCDF[keySub][keyF][cnt,:])
          if FS < d:
            d = FS
            for keyF in self.features:
              tempTyp[keySub][keyF] = tempData[keySub][keyF][cnt,:]

      typicalTS = {self.pivotParameterID:np.array([])}
      for keySub in subKeys:
        typicalTS[self.pivotParameterID] = np.concatenate((typicalTS[self.pivotParameterID], tempTyp[keySub][self.pivotParameterID]))
        for keyF in self.features:
          if keyF not in typicalTS.keys():  typicalTS[keyF] = np.array([])
          typicalTS[keyF] = np.concatenate((typicalTS[keyF], tempTyp[keySub][keyF]))

      for t in range(1,len(typicalTS[self.pivotParameterID])):
        if typicalTS[self.pivotParameterID][t] < typicalTS[self.pivotParameterID][t-1]: return None

      outputDic ={'data':{'input':{},'output':{}}, 'metadata':{}}
      outputDic['data']['output'][1] = typicalTS
      keyH = inputDict['input'].keys()[0]
      outputDic['data']['input'][1] = {}
      for keyIn in inputDict['input'][keyH].keys():
        outputDic['data']['input'][1][keyIn] = np.array(inputDict['input'][keyH][keyIn][0])

      return outputDic

  def __computeECDF(self, data, binEdgesIn):
    """
      Method to generate empirical CDF of input data, with the bins given.
      @ In, data, numpy array, data for which empirical CDF is computed
      @ In, binEdgesIn, numpy array, bins over which CDF value is computed
      @ Out, , numpy array, empirical CDF of the input data.
    """
    (counts, _) = np.histogram(data,bins=binEdgesIn,normed=True)
    return np.cumsum(counts)/max(np.cumsum(counts))

  def __computeDist(self, x1, x2):
    """
      Method to compute absolute difference of two points.
      @ In, x1, numpy array, input 1
      @ In, x2, numpy array, input 2
      @ Out, , float, difference between x1 and x2
    """
    return np.average(np.absolute(x1-x2))

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'subseqLen':
        self.subseqLen = map(int, child.text.split(','))
      elif child.tag == 'pivotParameter':
        self.pivotParameterID = child.text
      elif child.tag == 'outputLen':
        self.outputLen = float(child.text)
