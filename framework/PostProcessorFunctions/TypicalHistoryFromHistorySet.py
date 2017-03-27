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
'''
  Created on Feb 17, 2016

'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
import numpy as np
import copy
from utils import mathUtils

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

    #get actual data
    inputDict = inputDic[0]['data']
    #identify features
    self.features = inputDict['output'][inputDict['output'].keys()[0]].keys()
    #don't keep the pivot parameter in the feature space
    if self.pivotParameterID in self.features:
      self.features.remove(self.pivotParameterID)

    #if output length (size of desired output history) not set, set it now
    if self.outputLen is None:
      self.outputLen = np.asarray(inputDict['output'][inputDict['output'].keys()[0]][self.pivotParameterID])[-1]

    # task: reshape the data into histories with the size of the output I'm looking for
    #data dictionaries have form {historyNumber:{VarName:[data], VarName:[data]}}
    reshapedData = {}
    #keyNewH = 0
    newHistoryCounter = 0 #new history tracking labels
    for historyNumber in inputDict['output'].keys():
      #array of the pivot values provided in the history
      pivotValues = np.asarray(inputDict['output'][historyNumber][self.pivotParameterID])
      #if the desired output pivot value length is (equal to or) longer than the provided history ...
      #   -> (i.e. I have a year and I want output of a year)
      if self.outputLen >= pivotValues[-1]:
        #don't change the shape of this history; it's fine as is
        reshapedData[newHistoryCounter] = inputDict['output'][historyNumber]
        newHistoryCounter += 1
      #if the provided history is longer than the requested output period
      #   -> (i.e., I have a year of data and I only want output of 1 year)
      else:
        #reshape the history into multiple histories to use
        startPivot = 0
        endPivot = self.outputLen
        # until you find the last observed pivot point...
        while endPivot <= pivotValues[-1]:
          #create a storage place for each new usable history
          reshapedData[newHistoryCounter] = {}
          # acceptable is if the pivot value is greater than start and less than end
          # this doesn't work with "and" because of ambiguity, so we use "*" instead.
          extractCondition = (pivotValues>=startPivot) * (pivotValues<=endPivot)
          # extract out the acceptable parts from the pivotValues, and reset the base pivot point to 0
          reshapedData[newHistoryCounter][self.pivotParameterID] = np.extract(extractCondition, pivotValues)-startPivot
          # for each feature...
          for feature in self.features:
            # extract applicable information from the feature set
            reshapedData[newHistoryCounter][feature] = np.extract(extractCondition, inputDict['output'][historyNumber][feature])
          #increment history counter
          newHistoryCounter += 1
          #update new start/end points for grabbing the next history
          startPivot = endPivot
          endPivot += self.outputLen

    inputDict['output'] = reshapedData
    self.numHistory = len(inputDict['output'].keys()) #should be same as newHistoryCounter - 1, if that's faster
    #update the set of pivot parameter values to match the first of the reshaped histories
    self.pivotParameter = np.asarray(inputDict['output'][inputDict['output'].keys()[0]][self.pivotParameterID])

    # task: split the history into multiple subsequences so that the typical history can be constructed
    #  -> i.e., split the year history into multiple months, so we get a typical January, February, ..., hence a typical year
    self.subsequence = {} #TODO change this to a list instead of dict keyed on integers
    startLocation = 0 #tracks the point in the history being evaluated
    n = 0 #counts the number of the subsequence
    # in this loop we collect the similar (in time) subsequences in each history
    while True:
      subsequenceLength = self.subseqLen[n % len(self.subseqLen)] # "%" lets different subsequence lengths be used
      # if the history is longer than the subsequence we need, take the whole subsequence
      if startLocation + subsequenceLength < self.pivotParameter[-1]:
        self.subsequence[n] = [startLocation, startLocation+subsequenceLength]
      # otherwise, take only as much as the history has, and exit
      else:
        self.subsequence[n] = [startLocation, self.pivotParameter[-1]]
        break
      # iterate forward
      startLocation += subsequenceLength
      n+= 1
    # FIXME who is this guy
    subKeys = self.subsequence.keys()
    subKeys.sort() #FIXME why would we reorder them, this is just a range(n)

    # FIXME terrible variable name
    tempData = {'all':{}}  # eventually keys are 'all', each subsequence indicator index (fix this!)
    # stack the similar histories in numpy arrays
    for feature in self.features:
      #tempData['all'][feature] = np.array([])
      tempData['all'][feature] = np.concatenate(list(inputDict['output'][h][feature] for h in inputDict['output'].keys()))
      #for each provided history ...
      #for historyNumber in inputDict['output'].keys():
      #  tempData['all'][feature] = np.concatenate((tempData['all'][feature],inputDict['output'][historyNumber][feature]))
        # FIXME this is a fairly expensive operation to do in double-for-loop fashion, can we do them all at once?

    for keySub in subKeys:
      tempData[keySub] = {}
      extractCondition = (self.pivotParameter>=self.subsequence[keySub][0]) * (self.pivotParameter<self.subsequence[keySub][1])
      tempData[keySub][self.pivotParameterID] = np.extract(extractCondition, self.pivotParameter)
      if self.pivotParameter[-1] == self.subsequence[keySub][1]:
        tempData[keySub][self.pivotParameterID] = np.concatenate((tempData[keySub][self.pivotParameterID], np.asarray([self.pivotParameter[-1]])))

      for keyF in self.features:
        tempData[keySub][keyF] = np.zeros(shape=(self.numHistory,len(tempData[keySub][self.pivotParameterID])))
        for cnt, historyNumber in enumerate(inputDict['output'].keys()):
          if self.pivotParameter[-1] == self.subsequence[keySub][1]:
            tempData[keySub][keyF][cnt,0:-1] = np.extract(extractCondition, inputDict['output'][historyNumber][keyF])
            tempData[keySub][keyF][cnt,-1] = inputDict['output'][historyNumber][keyF][-1]
          else:
            tempData[keySub][keyF][cnt,:] = np.extract(extractCondition, inputDict['output'][historyNumber][keyF])

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
      for cnt, historyNumber in enumerate(inputDict['output'].keys()):
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
    historyNumber = inputDict['input'].keys()[0]
    outputDic['data']['input'][1] = {}
    for keyIn in inputDict['input'][historyNumber].keys():
      outputDic['data']['input'][1][keyIn] = np.array(inputDict['input'][historyNumber][keyIn][0])

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
