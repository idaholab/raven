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
import numpy as np
import copy
from collections import defaultdict
from functools import partial

from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import mathUtils, utils, InputData, InputTypes

class TypicalHistoryFromHistorySet(PostProcessorReadyInterface):
  """
    This class forms a typical history from a history set
    The methodology can be found at:
    S. WIlcox and W. Marion, "User Manual for TMY3 Data Sets," Technical Report, NREL/TP-581-43156,
    National Renewable Energy Laboratory Golden, CO, May 2008
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("subseqLen", contentType=InputTypes.IntegerListType))
    inputSpecification.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("outputLen", contentType=InputTypes.FloatType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.setInputDataType('dict')
    self.keepInputMeta(True)
    self.outputMultipleRealizations = True # True indicate multiple realizations are returned
    self.validDataType = ['HistorySet'] # The list of accepted types of DataObject
    self.pivotParameter = 'time' #FIXME this assumes the ARMA model!  Dangerous assumption.
    self.outputLen = None

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs)>1:
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only one dataObject')
    if inputs[0].type != 'HistorySet':
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only HistorySet dataObject, but got "{}"'.format(inputs[0].type))

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    self.name = paramInput.parameterValues['name']
    for child in paramInput.subparts:
      if child.getName() == 'subseqLen':
        self.subseqLen = child.value
      elif child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      elif child.getName() == 'outputLen':
        self.outputLen = child.value
    # checks
    if not hasattr(self, 'pivotParameter'):
      self.raiseAnError(IOError,'"pivotParameter" was not specified for "{}" PostProcessor!'.format(self.name))

  def retrieveHistory(self,dictIn,N):
    """
      Function that returns a dictionary containing the data of history N
      @ In, dictIn, dict, dictionary containing the full historySet
      @ In, N, int, numerical ID of the history to be retrieved
      @ Out, outputDict, dict, dictionary containing all data of history N
    """
    outputDict = {}
    for var in dictIn.keys():
      outputDict[var]=dictIn[var][N]
    return outputDict

  def run(self,inputIn):
    """
      @ In, inputIn, dict, dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, dictionary which contains the data to be collected by output DataObject
    """
    inpVars, outVars, inputDic = inputIn['Data'][0]
    numSamples = inputDic['numberRealizations']
    inputDict = inputDic['data']

    #identify features
    self.features = outVars
    #don't keep the pivot parameter in the feature space
    if self.pivotParameter in self.features:
      self.features.remove(self.pivotParameter)

    #if output length (size of desired output history) not set, set it now
    if self.outputLen is None:
      self.outputLen = np.asarray(inputDict['output'][utils.first(inputDict['output'].keys())][self.pivotParameter])[-1]

    ## Check if data is synchronized
    referenceHistory = 0
    referenceTimeAxis = inputDict[self.pivotParameter][referenceHistory]
    for hist in range(numSamples):
      if str(inputDict[self.pivotParameter][hist]) != str(referenceTimeAxis):
        errorMessage = '{} Interfaced Post-Processor "{}": one or more histories in the historySet have different time scales (e.g., reference points: {} and {})'.format(self.__class__.__name__, self.name,referenceHistory, hist)
        self.raiseAnError(IOError, errorMessage)

    # task: reshape the data into histories with the size of the output I'm looking for
    #data dictionaries have form {historyNumber:{VarName:[data], VarName:[data]}}
    reshapedData = {}
    newHistoryCounter = 0 #new history tracking labels
    for historyNumber in range(numSamples):
      #array of the pivot values provided in the history
      pivotValues = np.asarray(inputDict[self.pivotParameter][historyNumber])
      #if the desired output pivot value length is (equal to or) longer than the provided history ...
      #   -> (i.e. I have a year and I want output of a year)
      if self.outputLen >= pivotValues[-1]:
        #don't change the shape of this history; it's fine as is
        reshapedData[newHistoryCounter] = self.retrieveHistory(inputDict,historyNumber)
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
          extractCondition = np.logical_and(pivotValues>=startPivot, pivotValues<=endPivot)
          # extract out the acceptable parts from the pivotValues, and reset the base pivot point to 0
          reshapedData[newHistoryCounter][self.pivotParameter] = np.extract(extractCondition, pivotValues)-startPivot
          # for each feature...
          for feature in self.features:
            # extract applicable information from the feature set
            reshapedData[newHistoryCounter][feature] = np.extract(extractCondition, inputDict[feature][historyNumber])
          #increment history counter
          newHistoryCounter += 1
          #update new start/end points for grabbing the next history
          startPivot = endPivot
          endPivot += self.outputLen

    inputDict['output'] = reshapedData
    self.numHistory = len(inputDict['output'].keys()) #should be same as newHistoryCounter - 1, if that's faster
    #update the set of pivot parameter values to match the first of the reshaped histories
    self.pivotValues = np.asarray(inputDict['output'][utils.first(inputDict['output'].keys())][self.pivotParameter])

    # task: split the history into multiple subsequences so that the typical history can be constructed
    #  -> i.e., split the year history into multiple months, so we get a typical January, February, ..., hence a typical year
    # start by identifying the subsequences within the histories
    self.subsequence = [] #list of start/stop pivot values for the subsequences
    startLocation = 0     #tracks the point in the history being evaluated
    n = 0                 #counts the number of the subsequence
    # in this loop we collect the similar (in time) subsequences in each history
    while True:
      subsequenceLength = self.subseqLen[n % len(self.subseqLen)]
      # if the history is longer than the subsequence we need, take the whole subsequence
      if startLocation + subsequenceLength < self.pivotValues[-1]:
        self.subsequence.append([startLocation, startLocation+subsequenceLength])
      # otherwise, take only as much as the history has, and exit
      else:
        self.subsequence.append([startLocation, self.pivotValues[-1]])
        break # TODO this could be made "while startLocation + subsequenceLength < self.pivotValues[-1]
      # iterate forward
      startLocation += subsequenceLength
      n+= 1
    numParallelSubsequences = len(self.subsequence)

    #now that the subsequences are identified, collect the data
    # for the record, defaultdict is a dict that auto-populates using the constructer given if an element isn't present
    subseqData = defaultdict(dict)  # eventually {'all':{feature:[[parallel output data]], feature:[[parallel output data]]},
    #                                    subseqIndex:{pivotParam:pivotValues[-1]},
    #                                                 feature:[[parallel data]]}
    # 'all' means all the feature data is included,
    #     while the subseqIndex dictionaries only contain the relevant subsequence data (i.e., the monthly data)
    # stack the similar histories in numpy arrays for full period (for example, by year)
    for feature in self.features:
      subseqData['all'][feature] = np.concatenate(list(inputDict['output'][h][feature] for h in inputDict['output'].keys()))

    # gather feature data by subsequence (for example, by month)
    for index in range(numParallelSubsequences):
      extractCondition = np.logical_and(self.pivotValues>=self.subsequence[index][0], self.pivotValues<self.subsequence[index][1])
      subseqData[index][self.pivotParameter] = np.extract(extractCondition, self.pivotValues)
      #get the pivot parameter entries as well, but only do it once, at the end
      if self.pivotValues[-1] == self.subsequence[index][1]:
        subseqData[index][self.pivotParameter] = np.concatenate((subseqData[index][self.pivotParameter], np.asarray([self.pivotValues[-1]])))
      #get the subsequence data for each feature, for each history
      for feature in self.features:
        subseqData[index][feature] = np.zeros(shape=(self.numHistory,len(subseqData[index][self.pivotParameter])))
        for h, historyNumber in enumerate(inputDict['output'].keys()):
          if self.pivotValues[-1] == self.subsequence[index][1]:
            #TODO this is doing the right action, but it's strange that we need to add one extra element.
            #  Maybe this should be fixed where we set the self.subsequence[index][1] for the last index, instead of patched here
            subseqData[index][feature][h,0:-1] = np.extract(extractCondition, inputDict['output'][historyNumber][feature])
            subseqData[index][feature][h,-1] = inputDict['output'][historyNumber][feature][-1]
          else:
            subseqData[index][feature][h,:] = np.extract(extractCondition, inputDict['output'][historyNumber][feature])

    # task: compare CDFs to find the nearest match to the collective time's standard CDF (see the paper ref'd in the manual)
    # start by building the CDFs in the same structure as subseqData
    # for the record, defaultdict is a dict that auto-populates using the constructer given if an element isn't present
    cdfData = defaultdict(dict) # eventually {'all':{feature:[monotonically increasing floats], feature:[monotonically increasing floats]},
    #                                    subseqIndex:{pivotParam:pivotValues[-1]},
    #                                                 feature:[monotonically increasing floats]}
    # TODO there surely is a faster way to do this than triple-for-loops
    for feature in self.features:
      #construct reasonable bins for feature
      numBins, binEdges = mathUtils.numBinsDraconis(subseqData['all'][feature])
      #get the empirical CDF by bin for entire history (e.g., full year or even multiple years)
      cdfData['all'][feature] = self.__computeECDF(subseqData['all'][feature], binEdges)
      #get the empirical CDF by bin for subsequence (e.g., for a month)
      for index in range(numParallelSubsequences):
        cdfData[index][feature] = np.zeros(shape=(self.numHistory,numBins))
        for h in range(self.numHistory):
          cdfData[index][feature][h,:] = self.__computeECDF(subseqData[index][feature][h,:], binEdges)

    # now determine which subsequences are the most typical, using the CDF
    # find the smallestDeltaCDF and its index so the typical data can be set
    # first, find and store them by history
    typicalDataHistories = {}
    for index in range(numParallelSubsequences):
      typicalDataHistories[index] = {}
      typicalDataHistories[index][self.pivotParameter] = subseqData[index][self.pivotParameter]
      smallestDeltaCDF = np.inf
      smallestDeltaIndex = numParallelSubsequences + 1 #initialized as bogus index to preserve errors
      for h in range(self.numHistory):# for h, historyNumber in enumerate(inputDict['output'].keys()):
        delta = sum(self.__computeDist(cdfData['all'][feature],cdfData[index][feature][h,:]) for feature in self.features)
        if delta < smallestDeltaCDF:
          smallestDeltaCDF = delta
          smallestDeltaIndex = h
      for feature in self.features:
        typicalDataHistories[index][feature] = subseqData[index][feature][smallestDeltaIndex,:]
    # now collapse the data into the typical history
    typicalData = {}
    typicalData[self.pivotParameter] = np.concatenate(list(typicalDataHistories[index][self.pivotParameter] for index in range(numParallelSubsequences)))
    for feature in self.features:
      typicalData[feature] = np.concatenate(list(typicalDataHistories[index][feature] for index in range(numParallelSubsequences)))
    # sanity check, should probably be skipped for efficiency, as it looks like a debugging tool
    # preserved for now in case it was important for an undiscovered reason
    #   for t in range(1,len(typicalData[self.pivotParameter])):
    #      if typicalData[self.pivotParameter][t] < typicalData[self.pivotParameter][t-1]:
    #        self.raiseAnError(RuntimeError,'Something went wrong with the TypicalHistorySet!  Expected calculated data is missing.')

    # task: collect data as expected by RAVEN
    outputDict ={'data':{}}
    # typical history
    for var in typicalData.keys():
      outputDict['data'][var] = np.zeros(1, dtype=object)
      outputDict['data'][var][0] = typicalData[var]
    # preserve input data
    for var in inputDic['inpVars']:
      outputDict['data'][var] = np.zeros(1, dtype=object)
      outputDict['data'][var][0] = inputDict[var][0]
    outputDict['dims']={}
    for var in self.features:
      outputDict['dims'][var]=[self.pivotParameter]
    return outputDict

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
