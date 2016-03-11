'''
Created on Feb 17, 2016

'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
import numpy as np

class TypicalHistoryFromHistorySet(PostProcessorInterfaceBase):
  """ This class forms a typical history from a history set
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
    self.outputFormat = 'History'

  def run(self,inputDic):
    """
     @ In, inputDic, dict, dictionary which contains the data inside the input DataObject
     @ Out, outputDic, dict, dictionary which contains the data to be collected by output DataObject

    """
    inputDict = inputDic['data']
    self.features = inputDict['output'][inputDict['output'].keys()[0]].keys()
    self.features.remove('Time')
    self.noHistory = len(inputDict['output'].keys())
    self.Time = np.asarray(inputDict['output'][inputDict['output'].keys()[0]][self.timeID])

    if self.subseqP == 'Month':
      self.subsequence = {1:[0,2678400], 2:[2678400,5097600], 3:[5097600,7776000], 4:[7776000,10368000], 5:[10368000,13046400], 6:[13046400,15638400], 7:[15638400,18316800], 8:[18316800,20995200], 9:[20995200,23587200], 10:[23587200,26265600],11:[26265600,28857600],12:[28857600,31536000]}
    else:
      self.subsequence = {}
      n = 1;
      while True:
        if n*self.subseqP<self.Time[-1]:
          self.subsequence[n] = [(n-1)*self.subseqP,n*self.subseqP]
          n += 1
        else:
          self.subsequence[n] = [(n-1)*self.subseqP,self.Time[-1]]
          break
    subKeys = self.subsequence.keys()
    subKeys.sort()

    tempData = {'all':{}}
    for keyF in self.features:
      tempData['all'][keyF] = np.array([])
      for keyH in inputDict['output'].keys():
        tempData['all'][keyF] = np.concatenate((tempData['all'][keyF],inputDict['output'][keyH][keyF]))
    for keySub in subKeys:
      tempData[keySub] = {}
      extractCondition = (self.Time>=self.subsequence[keySub][0]) * (self.Time<self.subsequence[keySub][1])
      tempData[keySub]['Time'] = np.extract(extractCondition, self.Time)
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
      tempCDF['all'][keyF] = self.__computeCDF(tempData['all'][keyF],binEdges)
      for keySub in subKeys:
        if keySub not in tempCDF.keys(): tempCDF[keySub] = {}
        tempCDF[keySub][keyF] = np.zeros(shape=(self.noHistory,numBins))
        for cnt in range(tempData[keySub][keyF].shape[0]):
          tempCDF[keySub][keyF][cnt,:] = self.__computeCDF(tempData[keySub][keyF][cnt,:], binEdges)

    tempTyp = {}
    for keySub in subKeys:
      tempTyp[keySub] = {}
      tempTyp[keySub]['Time'] = tempData[keySub]['Time']
      d = np.inf
      for cnt, keyH in enumerate(inputDict['output'].keys()):
        FS = 0
        for keyF in self.features:
          FS += self.__computeDist(tempCDF['all'][keyF],tempCDF[keySub][keyF][cnt,:])
          print(FS)
        if FS < d:
          d = FS
          for keyF in self.features:
            tempTyp[keySub][keyF] = tempData[keySub][keyF][cnt,:]

    typicalTS = {'Time':np.array([])}
    for keySub in subKeys:
      typicalTS['Time'] = np.concatenate((typicalTS['Time'], tempTyp[keySub]['Time']))
      for keyF in self.features:
        if keyF not in typicalTS.keys():  typicalTS[keyF] = np.array([])
        typicalTS[keyF] = np.concatenate((typicalTS[keyF], tempTyp[keySub][keyF]))

    for t in range(1,len(typicalTS['Time'])):
      if typicalTS['Time'][t] < typicalTS['Time'][t-1]: return None

    outputDic ={'data':{'input':{},'output':{}}, 'metadata':{0:{}}}
    outputDic['data']['output'][1] = typicalTS
    keyH = inputDict['input'].keys()[0]
    outputDic['data']['input'][1] = {}
    for keyIn in inputDict['input'][keyH].keys():
      outputDic['data']['input'][1][keyIn] = np.array(inputDict['input'][keyH][keyIn][0])

    return outputDic

  def __computeCDF(self, data, binEdgesIn):
    """
     Method to generate empirical CDF of input data, with the bins given.
     @ In, data, array, data for which empirical CDF is computed
     @ In, binEdgesIn, array, bins over which CDF value is computed
     @ Out, , array, empirical CDF of the input data.

    """
    counts, binEdges = np.histogram(data,bins=binEdgesIn,normed=True)
    numBins = len(counts)
    Delta = np.zeros(shape=(numBins,))
    for n in range(numBins):
      Delta[n] = binEdges[n+1]-binEdges[n]
    delta = np.average(Delta)
    return np.cumsum(counts)*delta

  def __computeDist(self, x1, x2):
    """
    Method to compute absolute difference of two points.
    @ In, x1, array, input 1
    @ In, x2, array, input 2
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
      if child.tag == 'subsequence':
        if child.text == 'Month': self.subseqP = child.text
        else:                     self.subseqP = int(child.text)
      elif child.tag == 'timeID':
        self.timeID = child.text
