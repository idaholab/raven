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
#External Modules------------------------------------------------------------------------------------
import numpy as np
import os
from collections import OrderedDict
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import InputData, InputTypes
from ... import Files
#Internal Modules End--------------------------------------------------------------------------------

class ImportanceRank(PostProcessorInterface):
  """
    ImportantRank class. It computes the important rank for given input parameters
    1. The importance of input parameters can be ranked via their sensitivies (SI: sensitivity index)
    2. The importance of input parameters can be ranked via their sensitivies and covariances (II: importance index)
    3. The importance of input directions based principal component analysis of inputs covariances (PCA index)
    3. CSI: Cumulative sensitive index (added in the future)
    4. CII: Cumulative importance index (added in the future)
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
    inputSpecification = super(ImportanceRank, cls).getInputSpecification()

    WhatInput = InputData.parameterInputFactory("what", contentType=InputTypes.StringType)
    inputSpecification.addSub(WhatInput)

    VariablesInput = InputData.parameterInputFactory("variables", contentType=InputTypes.StringType)
    DimensionsInput = InputData.parameterInputFactory("dimensions", contentType=InputTypes.StringType)
    ManifestInput = InputData.parameterInputFactory("manifest", contentType=InputTypes.StringType)
    ManifestInput.addSub(VariablesInput)
    ManifestInput.addSub(DimensionsInput)
    LatentInput = InputData.parameterInputFactory("latent", contentType=InputTypes.StringType)
    LatentInput.addSub(VariablesInput)
    LatentInput.addSub(DimensionsInput)
    FeaturesInput = InputData.parameterInputFactory("features", contentType=InputTypes.StringType)
    FeaturesInput.addSub(ManifestInput)
    FeaturesInput.addSub(LatentInput)
    inputSpecification.addSub(FeaturesInput)

    TargetsInput = InputData.parameterInputFactory("targets", contentType=InputTypes.StringType)
    inputSpecification.addSub(TargetsInput)

    #DimensionsInput = InputData.parameterInputFactory("dimensions", contentType=InputTypes.StringType)
    #inputSpecification.addSub(DimensionsInput)

    MVNDistributionInput = InputData.parameterInputFactory("mvnDistribution", contentType=InputTypes.StringType)
    MVNDistributionInput.addParam("class", InputTypes.StringType, True)
    MVNDistributionInput.addParam("type", InputTypes.StringType, True)
    inputSpecification.addSub(MVNDistributionInput)

    PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    inputSpecification.addSub(PivotParameterInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.targets = []
    self.features = []
    self.latent = []
    self.latentDim = []
    self.manifest = []
    self.manifestDim = []
    self.dimensions = []
    self.mvnDistribution = None
    self.acceptedMetric = ['sensitivityindex','importanceindex','pcaindex','transformation','inversetransformation','manifestsensitivity']
    self.all = ['sensitivityindex','importanceindex','pcaindex']
    self.statAcceptedMetric = ['pcaindex','transformation','inversetransformation']
    self.what = self.acceptedMetric # what needs to be computed, default is all
    self.printTag = 'POSTPROCESSOR IMPORTANTANCE RANK'
    self.transformation = False
    self.latentSen = False
    self.reconstructSen = False
    self.pivotParameter = None # time-dependent pivot parameter
    self.dynamic        = False # is it time-dependent?
    # assembler objects to be requested
    self.addAssemblerObject('mvnDistribution', InputData.Quantity.one)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'what':
        what = child.value.strip()
        if what.lower() == 'all':
          self.what = self.all
        else:
          requestMetric = list(var.strip() for var in what.split(','))
          toCalculate = []
          for metric in requestMetric:
            if metric.lower() == 'all':
              toCalculate.extend(self.all)
            elif metric.lower() in self.acceptedMetric:
              if metric.lower() not in toCalculate:
                toCalculate.append(metric.lower())
              else:
                self.raiseAWarning('Duplicate calculations',metric,'are removed from XML node <what> in',self.printTag)
            else:
              self.raiseAnError(IOError, self.printTag,'asked unknown operation', metric, '. Available',str(self.acceptedMetric))
          self.what = toCalculate
      elif child.getName() == 'targets':
        self.targets = list(inp.strip() for inp in child.value.strip().split(','))
      elif child.getName() == 'features':
        for subNode in child.subparts:
          if subNode.getName() == 'manifest':
            for subSubNode in subNode.subparts:
              if subSubNode.getName() == 'variables':
                self.manifest = list(inp.strip() for inp in subSubNode.value.strip().split(','))
                self.features.extend(self.manifest)
              elif subSubNode.getName() == 'dimensions':
                self.manifestDim = list(int(inp.strip()) for inp in subSubNode.value.strip().split(','))
              else:
                self.raiseAnError(IOError, 'Unrecognized xml node name:',subSubNode.getName(),'in',self.printTag)
          if subNode.getName() == 'latent':
            self.latentSen = True
            for subSubNode in subNode.subparts:
              if subSubNode.getName() == 'variables':
                self.latent = list(inp.strip() for inp in subSubNode.value.strip().split(','))
                self.features.extend(self.latent)
              elif subSubNode.getName() == 'dimensions':
                self.latentDim = list(int(inp.strip()) for inp in subSubNode.value.strip().split(','))
              else:
                self.raiseAnError(IOError, 'Unrecognized xml node name:',subSubNode.getName(),'in',self.printTag)
      elif child.getName() == 'mvnDistribution':
        self.mvnDistribution = child.value.strip()
      elif child.getName() == "pivotParameter":
        self.pivotParameter = child.value
      else:
        self.raiseAnError(IOError, 'Unrecognized xml node name: ' + child.getName() + '!')
    if not self.latentDim and len(self.latent) != 0:
      self.latentDim = range(1,len(self.latent)+1)
      self.raiseAWarning('The dimensions for given latent variables: ' + str(self.latent) + ' is not provided! Default dimensions will be used: ' + str(self.latentDim) + ' in ' + self.printTag)
    if not self.manifestDim and len(self.manifest) !=0:
      self.manifestDim = range(1,len(self.manifest)+1)
      self.raiseAWarning('The dimensions for given latent variables: ' + str(self.manifest) + ' is not provided! Default dimensions will be used: ' + str(self.manifestDim) + ' in ' + self.printTag)
    if not self.features:
      self.raiseAnError(IOError, 'No variables provided for XML node: features in',self.printTag)
    if not self.targets:
      self.raiseAnError(IOError, 'No variables provided for XML node: targets in', self.printTag)
    if len(self.latent) !=0 and len(self.manifest) !=0:
      self.reconstructSen = True
      self.transformation = True

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
      output.addRealization(outputDict)
    elif output.type == 'HDF5':
      self.raiseAWarning('Output type ' + str(output.type) + ' not yet implemented. Skip it !!!!!')
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    super().initialize(runInfo, inputs, initDict)
    self.mvnDistribution = self.retrieveObjectFromAssemblerDict('mvnDistribution', self.mvnDistribution)
  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputList, list, list of input dictionaries
    """
    if type(currentInp) == list:
      currentInput = currentInp[-1]
    else:
      currentInput = currentInp
    if type(currentInput) == dict:
      if 'targets' not in currentInput.keys() and 'timeDepData' not in currentInput.keys():
        self.raiseAnError(IOError, 'Did not find targets or timeDepData in input dictionary')
      return currentInput

    if hasattr(currentInput,'type'):
      inType = currentInput.type
    else:
      if type(currentInput).__name__ == 'list':
        inType = 'list'
      else:
        self.raiseAnError(IOError, self, 'ImportanceRank postprocessor accepts Files, HDF5, PointSet, DataObject(s) only! Got ' + str(type(currentInput)))
    if inType not in ['HDF5', 'PointSet','HistorySet', 'list'] and not isinstance(inType,Files.File):
      self.raiseAnError(IOError, self, 'ImportanceRank postprocessor accepts Files, HDF5, HistorySet, PointSet, DataObject(s) only! Got ' + str(inType) + '!!!!')
    # get input from the external csv file
    if isinstance(inType,Files.File):
      if currentInput.subtype == 'csv':
        pass # to be implemented
    # get input from PointSet DataObject
    if inType in ['PointSet']:
      dataSet = currentInput.asDataset()
      inputDict = {'targets':{}, 'metadata':{}, 'features':{}}
      for feat in self.features:
        if feat in currentInput.getVars('input'):
          inputDict['features'][feat] = copy.copy(dataSet[feat].values)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(feat) + ' is listed ImportanceRank postprocessor features, but not found in the provided input!')
      for targetP in self.targets:
        if targetP in currentInput.getVars('output'):
          inputDict['targets'][targetP] = copy.copy(dataSet[targetP].values)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(targetP) + ' is listed ImportanceRank postprocessor targets, but not found in the provided output!')
      inputDict['metadata'] = currentInput.getMeta(pointwise=True)
      inputList = [inputDict]
    # get input from HistorySet DataObject
    if inType in ['HistorySet']:
      dataSet = currentInput.asDataset()
      if self.pivotParameter is None:
        self.raiseAnError(IOError, self, 'Time-dependent importance ranking is requested (HistorySet) but no pivotParameter got inputted!')
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
        inputDict['features'] = dict((feature, sliceData[feature].values) for feature in self.features)
        inputList.append(inputDict)
    # get input from HDF5 Database
    if inType == 'HDF5':
      pass  # to be implemented

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
    from sklearn.linear_model import LinearRegression

    outputDict = {}
    senCoeffDict = {}
    senWeightDict = {}
    # compute sensitivities of targets with respect to features
    featValues = []
    # compute importance rank
    if self.latentSen:
      for feat in self.latent:
        featValues.append(inputDict['features'][feat])
      feats = self.latent
      self.dimensions = self.latentDim
    else:
      for feat in self.manifest:
        featValues.append(inputDict['features'][feat])
      feats = self.manifest
      self.dimensions = self.manifestDim
    sampledFeatMatrix = np.atleast_2d(np.asarray(featValues)).T
    for target in self.targets:
      featCoeffs = LinearRegression().fit(sampledFeatMatrix, inputDict['targets'][target]).coef_
      featWeights = abs(featCoeffs)/np.sum(abs(featCoeffs))
      senWeightDict[target] = featWeights
      senCoeffDict[target] = featCoeffs
    for what in self.what:
      if what.lower() == 'sensitivityindex':
        what = 'sensitivityIndex'
        for target in self.targets:
          for i, feat in enumerate(feats):
            varName = '_'.join([what, target, feat])
            outputDict[varName] = np.atleast_1d(senWeightDict[target][i])
      if what.lower() == 'importanceindex':
        what = 'importanceIndex'
        for target in self.targets:
          featCoeffs = senCoeffDict[target]
          featWeights = []
          if not self.latentSen:
            for index,feat in enumerate(feats):
              totDim = self.mvnDistribution.dimension
              covIndex = totDim * (self.dimensions[index] - 1) + self.dimensions[index] - 1
              if self.mvnDistribution.covarianceType == 'abs':
                covTarget = featCoeffs[index] * self.mvnDistribution.covariance[covIndex] * featCoeffs[index]
              else:
                covFeature = self.mvnDistribution.covariance[covIndex]*self.mvnDistribution.mu[self.dimensions[index]-1]**2
                covTarget = featCoeffs[index] * covFeature * featCoeffs[index]
              featWeights.append(covTarget)
            featWeights = featWeights/np.sum(featWeights)
            for i, feat in enumerate(feats):
              varName = '_'.join([what, target, feat])
              outputDict[varName] = np.atleast_1d(featWeights[i])
          # if the features type is 'latent', since latentVariables are used to compute the sensitivities
          # the covariance for latentVariances are identity matrix
          else:
            for i, feat in enumerate(feats):
              varName = '_'.join([what, target, feat])
              outputDict[varName] = np.atleast_1d(senWeightDict[target][i])
      if what.lower() == 'manifestsensitivity':
        if self.reconstructSen:
          what = 'manifestSensitivity'
          # compute the inverse transformation matrix
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          inverseTransformationMatrix = self.mvnDistribution.inverseTransformationMatrix(manifestIndex)
          inverseTransformationMatrix = inverseTransformationMatrix[index]
          # recompute the sensitivities for manifest variables
          for target in self.targets:
            latentSen = np.asarray(senCoeffDict[target])
            if self.mvnDistribution.covarianceType == 'abs':
              manifestSen = list(np.dot(latentSen,inverseTransformationMatrix))
            else:
              manifestSen = list(np.dot(latentSen,inverseTransformationMatrix)/inputDict['targets'][target])
            for i, feat in enumerate(self.manifest):
              varName = '_'.join([what, target, feat])
              outputDict[varName] = np.atleast_1d(manifestSen[i])
        elif self.latentSen:
          self.raiseAnError(IOError, 'Unable to reconstruct the sensitivities for manifest variables, this is because no manifest variable is provided in',self.printTag)
        else:
          self.raiseAWarning('No latent variables, and there is no need to reconstruct the sensitivities for manifest variables!')
      #calculate PCA index
      if what.lower() == 'pcaindex':
        if not self.latentSen:
          self.raiseAWarning('pcaIndex can be not requested because no latent variable is provided!')
        else:
          what = 'pcaIndex'
          index = [dim-1 for dim in self.dimensions]
          singularValues = self.mvnDistribution.returnSingularValues(index)
          singularValues = list(singularValues/np.sum(singularValues))
          for i, feat in enumerate(feats):
            varName = '_'.join([what, feat])
            outputDict[varName] = np.atleast_1d(singularValues[i])
      if what.lower() == 'transformation':
        if self.transformation:
          what = 'transformation'
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          transformMatrix = self.mvnDistribution.transformationMatrix(index)
          for ind,var in enumerate(self.manifest):
            for i, feat in enumerate(feats):
              varName = '_'.join([what, var, feat])
              outputDict[varName] = np.atleast_1d(transformMatrix[manifestIndex[ind]][i])
        else:
          self.raiseAnError(IOError,'Unable to output the transformation matrix, please provide both "manifest" and "latent" variables in XML node "features" in',self.printTag)
      if what.lower() == 'inversetransformation':
        if self.transformation:
          what = 'inverseTransformation'
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          inverseTransformationMatrix = self.mvnDistribution.inverseTransformationMatrix(manifestIndex)
          for ind,var in enumerate(self.latent):
            for i, mVar in enumerate(self.manifest):
              varName = what + '_' + var + '_' + mVar
              varName = '_'.join([what, var, mVar])
              outputDict[varName] = np.atleast_1d(inverseTransformationMatrix[index[ind]][i])
        else:
          self.raiseAnError(IOError,'Unable to output the inverse transformation matrix, please provide both "manifest" and "latent" variables in XML node "features" in', self.printTag)

      # To be implemented
      #if what == 'CumulativeSenitivityIndex':
      #  self.raiseAnError(NotImplementedError,'CumulativeSensitivityIndex is not yet implemented for ' + self.printTag)
      #if what == 'CumulativeImportanceIndex':
      #  self.raiseAnError(NotImplementedError,'CumulativeImportanceIndex is not yet implemented for ' + self.printTag)

    return outputDict
