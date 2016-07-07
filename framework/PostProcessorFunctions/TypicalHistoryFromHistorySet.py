'''
Created on Feb 17, 2016

'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
import numpy as np

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
    if not hasattr(self, 'timeID'):   self.timeID = 'Time'

  def run(self,inputDic):
    """
     @ In, inputDic, dict, dictionary which contains the data inside the input DataObject
     @ Out, outputDic, dict, dictionary which contains the data to be collected by output DataObject

    """
    inputDict = inputDic['data']
    self.features = inputDict['output'][inputDict['output'].keys()[0]].keys()
    self.features.remove(self.timeID)
    self.noHistory = len(inputDict['output'].keys())
    self.time = np.asarray(inputDict['output'][inputDict['output'].keys()[0]][self.timeID])

    self.subsequence = {}
    startLocation, n = 0, 0
    while True:
      subsequenceLength = self.subseqLen[n % len(self.subseqLen)]
      if startLocation + subsequenceLength < self.time[-1]:
        self.subsequence[n] = [startLocation, startLocation+subsequenceLength]
      else:
        self.subsequence[n] = [startLocation, self.time[-1]]
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
      extractCondition = (self.time>=self.subsequence[keySub][0]) * (self.time<self.subsequence[keySub][1])
      tempData[keySub][self.timeID] = np.extract(extractCondition, self.time)
      for keyF in self.features:
        tempData[keySub][keyF] = np.zeros(shape=(self.noHistory,len(tempData[keySub][self.timeID])))
        for cnt, keyH in enumerate(inputDict['output'].keys()):
          tempData[keySub][keyF][cnt,:] = np.extract(extractCondition, inputDict['output'][keyH][keyF])

    tempCDF = {'all':{}}
    for keyF in self.features:
#       Bin size and number of bins determined by Freedman Diaconis rule
#       https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
      IQR = np.percentile(tempData['all'][keyF], 75) - np.percentile(tempData['all'][keyF], 25)
      binSize = 2.0*IQR*(tempData['all'][keyF].size**(-1.0/3.0))
      numBins = int((max(tempData['all'][keyF])-min(tempData['all'][keyF]))/binSize)
      binEdges = np.linspace(start=min(tempData['all'][keyF]),stop=max(tempData['all'][keyF]),num=numBins+1)
#       dataRange = (min(tempData['all'][keyF]), max(tempData['all'][keyF]))
      tempCDF['all'][keyF] = self.__computeECDF(tempData['all'][keyF], binEdges)# numBins, dataRange)
      for keySub in subKeys:
        if keySub not in tempCDF.keys(): tempCDF[keySub] = {}
        tempCDF[keySub][keyF] = np.zeros(shape=(self.noHistory,numBins))
        for cnt in range(tempData[keySub][keyF].shape[0]):
          tempCDF[keySub][keyF][cnt,:] = self.__computeECDF(tempData[keySub][keyF][cnt,:], binEdges)#numBins, dataRange)

    tempTyp = {}
    for keySub in subKeys:
      tempTyp[keySub] = {}
      tempTyp[keySub][self.timeID] = tempData[keySub][self.timeID]
      d = np.inf
      for cnt, keyH in enumerate(inputDict['output'].keys()):
        FS = 0
        for keyF in self.features:
          FS += self.__computeDist(tempCDF['all'][keyF],tempCDF[keySub][keyF][cnt,:])
        if FS < d:
          d = FS
          for keyF in self.features:
            tempTyp[keySub][keyF] = tempData[keySub][keyF][cnt,:]

    typicalTS = {self.timeID:np.array([])}
    for keySub in subKeys:
      typicalTS[self.timeID] = np.concatenate((typicalTS[self.timeID], tempTyp[keySub][self.timeID]))
      for keyF in self.features:
        if keyF not in typicalTS.keys():  typicalTS[keyF] = np.array([])
        typicalTS[keyF] = np.concatenate((typicalTS[keyF], tempTyp[keySub][keyF]))

    for t in range(1,len(typicalTS[self.timeID])):
      if typicalTS[self.timeID][t] < typicalTS[self.timeID][t-1]: return None

    outputDic ={'data':{'input':{},'output':{}}, 'metadata':{0:{}}}
    outputDic['data']['output'][1] = typicalTS
    keyH = inputDict['input'].keys()[0]
    outputDic['data']['input'][1] = {}
    for keyIn in inputDict['input'][keyH].keys():
      outputDic['data']['input'][1][keyIn] = np.array(inputDict['input'][keyH][keyIn][0])

    return outputDic

#   def __computeECDF(self, data, numBins, bounds):
#     """
#        Method to generate empirical CDF of input data, with the bin number given.
#        @ In, data, numpy array, data for which empirical CDF is computed
#        @ In, numBins, int, bin number for computing empirical CDF
#        @ In, bounds, tuple (lowerBound, upperBound), the boundaries within which the empirical CDF is computed
#        @ Out, , numpy array, empirical CDF of the input data.
# 
#     """
#     (cumFreqs, _, _, _) = scipy.stats.cumfreq(data, numBins, bounds)
#     return cumFreqs/max(cumFreqs)

  def __computeECDF(self, data, binEdgesIn):
    """
       Method to generate empirical CDF of input data, with the bins given.
       @ In, data, numpy array, data for which empirical CDF is computed
       @ In, binEdgesIn, numpy array, bins over which CDF value is computed
       @ Out, , numpy array, empirical CDF of the input data.
  
    """
    (counts, _) = np.histogram(data,bins=binEdgesIn,normed=True)
#     numBins = len(counts)
#     binDiffArray = np.zeros(shape=(numBins,))
#     for n in range(numBins):
#       binDiffArray[n] = binEdges[n+1]-binEdges[n]
#     binDiff = np.average(binDiffArray)
#     return np.cumsum(counts)*binDiff    
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
      elif child.tag == 'timeID':
        self.timeID = child.text

