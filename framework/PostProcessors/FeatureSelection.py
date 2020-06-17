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
Created on July 10, 2013

@author: alfoa
"""
from __future__ import division, print_function , unicode_literals, absolute_import

#External Modules------------------------------------------------------------------------------------
import numpy as np
import os
from collections import OrderedDict
import copy
from sklearn.feature_selection import RFE, RFECV, mutual_info_regression,  mutual_info_classif,  VarianceThreshold
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import InputData, InputTypes
import Files
#Internal Modules End--------------------------------------------------------------------------------


class FeatureSelection(PostProcessor):
  """
    Feature Selection PostProcessor is aimed to select the most important features with different methods.
    The importance ranking is provided.
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
    ## This will replace the lines above
    inputSpecification = super(FeatureSelection, cls).getInputSpecification()
    whatInput = InputTypes.makeEnumType("whatType","whatTypeType",["RFE","RFECV", "mutualInformation"])
    inputSpecification.addSub(InputData.parameterInputFactory("what", contentType=whatInput))
    TargetsInput = InputData.parameterInputFactory("targets", contentType=InputTypes.StringType)
    inputSpecification.addSub(TargetsInput)
    PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    inputSpecification.addSub(PivotParameterInput)
    numberOfFeatures = InputData.parameterInputFactory("minimumNumberOfFeatures", contentType=InputTypes.IntegerType)
    inputSpecification.addSub(numberOfFeatures)
    step = InputData.parameterInputFactory("step", contentType=InputTypes.FloatOrIntType)
    inputSpecification.addSub(step)
    aTarg = InputData.parameterInputFactory("aggregateTargets", contentType=InputTypes.BoolType)
    inputSpecification.addSub(aTarg)        
    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.targets = [] # targets
    self.what = None  # how to perform the selection (list is in InputData specification)
    self.settings = {}
    self.dynamic  = False # is it time-dependent?
    self.printTag = 'POSTPROCESSOR FEATURE SELECTION'
    
  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs
      @ In, xmlNode, xml.etree.ElementTree Element Objects, the xml element node that will be checked against the available options specific to this Sampler
      @ Out, None
    """
    paramInput = FeatureSelection.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == 'what': 
        self.what = child.value.strip()
      elif child.getName() == 'targets':
        self.targets = list(inp.strip() for inp in child.value.strip().split(','))
      elif child.getName() == 'step':
        self.settings[child.getName()] = child.value
      elif child.getName() == 'minimumNumberOfFeatures':
        self.settings[child.getName()] = child.value
      elif child.getName() == 'aggregateTargets':
        self.settings[child.getName()] = child.value        
      else:
        self.raiseAnError(IOError, 'Unrecognized xml node name: ' + child.getName() + '!')

  def collectOutput(self,finishedJob, output):
    """
      Function to place all of the computed data into the output object, (Files or DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict = evaluation[1]
    # Output to DataObjects
    if output.type in ['PointSet','HistorySet']:
      self.raiseADebug('Dumping output in data object named ' + output.name)
      output.load(outputDict, style="dict")
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)
    
  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputList, list, list of input dictionaries
    """
    currentInput = currentInp[-1]  if type(currentInp) == list else currentInp
    if type(currentInput) == dict:
      if 'targets' not in currentInput.keys() and 'timeDepData' not in currentInput.keys():
        self.raiseAnError(IOError, 'Did not find targets or timeDepData in input dictionary')
      return currentInput

    if not hasattr(currentInput,'type'):
      self.raiseAnError(IOError, self, 'FeatureSelection postprocessor accepts DataObject(s) only! Got ' + str(type(currentInput)))
    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'FeatureSelection postprocessor accepts DataObject(s) only! Got ' + str(currentInput.type) + '!!!!')
    # get input from PointSet DataObject
    if currentInput.type in ['PointSet']:
      dataSet = currentInput.asDataset()
      inputDict = {'targets':{}, 'metadata':{}, 'features':{}}
      for feat in list(set(list(dataSet.keys())) - set(list(self.targets))):
        inputDict['features'][feat] = copy.copy(dataSet[feat].values)
      for targetP in self.targets:
        if targetP in currentInput.getVars('output'):
          inputDict['targets'][targetP] = copy.copy(dataSet[targetP].values)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(targetP) + ' is listed FeatureSelection postprocessor targets, but not found in the provided output!')
      inputDict['metadata'] = currentInput.getMeta(pointwise=True)
      inputList = [inputDict]
    # get input from HistorySet DataObject
    if currentInput.type in ['HistorySet']:
      dataSet = currentInput.asDataset()
      if self.pivotParameter is None:
        self.raiseAnError(IOError, self, 'Time-dependent FeatureSelection is requested (HistorySet) but no pivotParameter got inputted!')
      self.dynamic = True
      self.pivotValue = dataSet[self.pivotParameter].values
      if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
        self.raiseAnError(IOError, "In data object ", currentInput.name, ", the realization' pivot parameters have unsynchronized pivot values!"
                + "Please use the internal postprocessor 'HistorySetSync' to synchronize the data.")
      slices = currentInput.sliceByIndex(self.pivotParameter)
      metadata = currentInput.getMeta(pointwise=True)
      inputList = []
      for sliceData in slices:
        inputDict = {}
        inputDict['metadata'] = metadata
        inputDict['targets'] = dict((target, sliceData[target].values) for target in self.targets)
        inputDict['features'] = dict((feature, sliceData[feature].values) for feature in list(set(list(dataSet.keys())) - set(list(self.targets))))
        inputList.append(inputDict)

    return inputList

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputList = self.inputToInternal(inputIn)
    if not self.dynamic:
      outputDict = self.__runLocal(inputList[0])
    else:
      outputList = []
      for inputDict in inputList:
        outputList.append(self.__runLocal(inputDict))
      outputDict = dict((var, list()) for var in outputList[0].keys())
      for output in outputList:
        for var, value in output.items():
          outputDict[var] = np.append(outputDict[var], value)
      # add the pivot parameter and its values
      outputDict[self.pivotParameter] = np.atleast_1d(self.pivotValue)

    return outputDict

  def __runLocal(self, inputDict):
    """
      This method executes the postprocessor action.
      @ In, inputDict, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, dictionary containing the evaluated data
    """
    # lr = LinearRegression().fit(sampledFeatMatrix, inputDict['targets'][target])
    # compute sensitivities of targets with respect to features
    step = self.settings.get('step', 1)
    nFeatures = self.settings.get('minimumNumberOfFeatures', None)
    aggregateTargets = self.settings.get("aggregateTargets", False)
    if aggregateTargets:
      # perform a PCA and analyze the first principal component
      kpca = KernelPCA(n_components=1, kernel = "rbf", random_state=0)
      newTarget =  kpca.fit_transform(np.atleast_2d(list(inputDict['targets'].values())).T)
      # print(kpca.explained_variance_ratio_)
    newTarget if aggregateTargets else inputDict['targets'][targ]  
    # compute importance rank
    outputDict = {}
    # transformer = FactorAnalysis(n_components=10, random_state=0)
    # tt = transformer.fit(np.atleast_2d(list(inputDict['features'].values())).T)
    if self.what == "RFE":
      selectors = [RFE(LinearRegression(), n_features_to_select=nFeatures, step=step) for _ in range(len(self.targets))]
      for i, targ in enumerate(self.targets):
        selectors[i] = selectors[i].fit(np.atleast_2d(list(inputDict['features'].values())).T, newTarget if aggregateTargets else inputDict['targets'][targ])
        self.raiseAMessage("Features downselected to "+str( selectors[i].n_features_) +" for target "+targ)
        outputDict[self.name+"_"+targ] = np.atleast_1d(np.array(list(inputDict['features'].keys()))[selectors[i].support_])
    elif self.what == 'RFECV':
      selectors = [RFECV(LinearRegression(), step=step, min_features_to_select=nFeatures, n_jobs=-1) for _ in range(len(self.targets))]
      minFeaturesSelected = int(1e6)
      for i, targ in enumerate(self.targets):
        selectors[i] = selectors[i].fit(np.atleast_2d(list(inputDict['features'].values())).T, newTarget if aggregateTargets else inputDict['targets'][targ])
        self.raiseAMessage("Features downselected to "+str( selectors[i].n_features_) +" for target "+targ)
        minFeaturesSelected = min(minFeaturesSelected, selectors[i].n_features_)
      for i, targ in enumerate(self.targets):
        outputDict[self.name+"_"+targ] = np.atleast_1d(np.array(list(inputDict['features'].keys()))[selectors[i].support_])[:minFeaturesSelected]
    elif self.what == 'mutualInformation':
      selectors = []
      for i, targ in enumerate(self.targets):
        sortedFeatures = mutual_info_regression(np.atleast_2d(list(inputDict['features'].values())).T, newTarget if aggregateTargets else inputDict['targets'][targ]).argsort()
        outputDict[self.name+"_"+targ] = np.atleast_1d(np.array(list(inputDict['features'].keys()))[sortedFeatures][-nFeatures:])
    elif self.what == 'PCA':
      transformer = KernelPCA(n_components=nFeatures, kernel = "rbf", random_state=0).fit(np.atleast_2d(list(inputDict['features'].values()) + list(inputDict['targets'].values())).T)
       

    return outputDict

