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
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import numpy as np
import os
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import InputData
import Files
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class ImportanceRank(PostProcessor):
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

    WhatInput = InputData.parameterInputFactory("what", contentType=InputData.StringType)
    inputSpecification.addSub(WhatInput)

    VariablesInput = InputData.parameterInputFactory("variables", contentType=InputData.StringType)
    DimensionsInput = InputData.parameterInputFactory("dimensions", contentType=InputData.StringType)
    ManifestInput = InputData.parameterInputFactory("manifest", contentType=InputData.StringType)
    ManifestInput.addSub(VariablesInput)
    ManifestInput.addSub(DimensionsInput)
    LatentInput = InputData.parameterInputFactory("latent", contentType=InputData.StringType)
    LatentInput.addSub(VariablesInput)
    LatentInput.addSub(DimensionsInput)
    FeaturesInput = InputData.parameterInputFactory("features", contentType=InputData.StringType)
    FeaturesInput.addSub(ManifestInput)
    FeaturesInput.addSub(LatentInput)
    inputSpecification.addSub(FeaturesInput)

    TargetsInput = InputData.parameterInputFactory("targets", contentType=InputData.StringType)
    inputSpecification.addSub(TargetsInput)

    #DimensionsInput = InputData.parameterInputFactory("dimensions", contentType=InputData.StringType)
    #inputSpecification.addSub(DimensionsInput)

    MVNDistributionInput = InputData.parameterInputFactory("mvnDistribution", contentType=InputData.StringType)
    inputSpecification.addSub(MVNDistributionInput)

    PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
    inputSpecification.addSub(PivotParameterInput)

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
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
    self.requiredAssObject = (True,(['Distributions'],[-1]))
    self.transformation = False
    self.latentSen = False
    self.reconstructSen = False
    self.pivotParameter = None # time-dependent pivot parameter
    self.dynamic        = False # is it time-dependent?

  def _localWhatDoINeed(self):
    """
      This method is local mirror of the general whatDoINeed method
      It is implemented by this postprocessor that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {'Distributions':[]}
    needDict['Distributions'].append((None,self.mvnDistribution))
    return needDict

  def _localGenerateAssembler(self,initDict):
    """
      see generateAssembler method in Assembler
      @ In, initDict, dict, dictionary ({'mainClassName':{'specializedObjectName':ObjectInstance}})
      @ Out, None
    """
    distName = self.mvnDistribution
    self.mvnDistribution = initDict['Distributions'][distName]

  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs
      @ In, xmlNode, xml.etree.ElementTree Element Objects, the xml element node that will be checked against the available options specific to this Sampler
      @ Out, None
    """
    paramInput = ImportanceRank.getInputSpecification()()
    paramInput.parseNode(xmlNode)
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

  def _localPrintXML(self,outFile,options=None,pivotVal=None):
    """
      Adds requested entries to XML node.
      @ In, outFile, Files.StaticXMLOutput, file to which entries will be printed
      @ In, options, dict, optional, list of requests and options
        May include: 'what': comma-separated string list, the qualities to print out
      @ In, pivotVal, float, value of the pivot parameter, i.e. time, burnup, ...
      @ Out, None
    """
    #build tree
    for what in options.keys():
      if what.lower() in self.statAcceptedMetric:
        continue
      if what == 'manifestSensitivity':
        continue
      for target in options[what].keys():
        valueDict = OrderedDict()
        for var,index,dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(target,what,valueDict,pivotVal=pivotVal)
    if 'manifestSensitivity' in options.keys():
      what = 'manifestSensitivity'
      for target in options[what].keys():
        outFile.addScalar(target,'type',self.mvnDistribution.covarianceType,pivotVal=pivotVal)
        valueDict = OrderedDict()
        for var,index,dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(target,what,valueDict,pivotVal=pivotVal)

  def _localPrintPCAInformation(self,outFile,options=None,pivotVal=None):
    """
      Adds requested entries to XML node.
      @ In, outFile, Files.StaticXMLOutput, file to which entries will be printed
      @ In, options, dict, optional, list of requests and options
        May include: 'what': comma-separated string list, the qualities to print out
      @ In, pivotVal, float, value of the pivot parameter, i.e. time, burnup, ...
      @ Out, None
    """
    # output variables and dimensions
    latentDict = OrderedDict()
    if self.latentSen:
      for index,var in enumerate(self.latent):
        latentDict[var] = self.latentDim[index]
      outFile.addVector('dimensions','latent',latentDict,pivotVal=pivotVal)
    manifestDict = OrderedDict()
    if len(self.manifest) > 0:
      for index,var in enumerate(self.manifest):
        manifestDict[var] = self.manifestDim[index]
      outFile.addVector('dimensions','manifest',manifestDict,pivotVal=pivotVal)
    #pca index is a feature only of target, not with respect to anything else
    if 'pcaIndex' in options.keys():
      pca = options['pcaIndex']
      for var,index,dim in pca:
        outFile.addScalar('pcaIndex',var,index,pivotVal=pivotVal)
    if 'transformation' in options.keys():
      what = 'transformation'
      outFile.addScalar(what,'type',self.mvnDistribution.covarianceType,pivotVal=pivotVal)
      for target in options[what].keys():
        valueDict = OrderedDict()
        for var, index, dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(what,target,valueDict,pivotVal=pivotVal)
    if 'inverseTransformation' in options.keys():
      what = 'inverseTransformation'
      outFile.addScalar(what,'type',self.mvnDistribution.covarianceType,pivotVal=pivotVal)
      for target in options[what].keys():
        valueDict = OrderedDict()
        for var, index, dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(what,target,valueDict,pivotVal=pivotVal)

  def collectOutput(self,finishedJob, output):
    """
      Function to place all of the computed data into the output object, (Files or DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, ' No available output to collect (Run probably is not finished yet) via',self.printTag)
    outputDict = evaluation[1]
    # Output to file
    if isinstance(output, Files.File):
      availExtens = ['xml','csv']
      outputExtension = output.getExt().lower()
      if outputExtension not in availExtens:
        self.raiseAWarning('Output extension you input is ' + outputExtension)
        self.raiseAWarning('Available are ' + str(availExtens) + '. Converting extension to ' + str(availExtens[0]) + '!')
        outputExtensions = availExtens[0]
        output.setExtension(outputExtensions)
      output.setPath(self._workingDir)
      self.raiseADebug('Dumping output in file named ' + output.getAbsFile())
      output.open('w')
      if outputExtension == 'csv':
        self._writeCSV(output,outputDict)
      else:
        self._writeXML(output,outputDict)
    # Output to DataObjects
    elif output.type in ['PointSet','HistorySet']:
      self.raiseADebug('Dumping output in data object named ' + output.name)
      self._writeDataObject(output,outputDict)
    elif output.type == 'HDF5':
      self.raiseAWarning('Output type ' + str(output.type) + ' not yet implemented. Skip it !!!!!')
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def _writeCSV(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, None
    """
    separator = ','
    if self.dynamic:
      output.write('Importance Rank' + separator + 'Pivot Parameter' + separator + self.pivotParameter + os.linesep)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for step, outputDict in enumerate(outputResults):
      if self.dynamic:
        output.write('Pivot Value'+separator+str(outputDictionary.keys()[step])+os.linesep)
      #only output 'pcaindex','transformation','inversetransformation' for the first step.
      if step == 0:
        for what in outputDict.keys():
          if what.lower() in self.statAcceptedMetric:
            self.raiseADebug('Writing parameter rank for metric ' + what)
            if what.lower() == 'pcaindex':
              output.write('pcaIndex,' + '\n')
              output.write('Parameters'  + ''.join([separator + str(item[0])  for item in outputDict[what]]) + os.linesep)
              output.write(what + ''.join([separator + '%.8E' % item[1] for item in outputDict[what]]) + os.linesep)
              output.write(os.linesep)
            else:
              for target in outputDict[what].keys():
                output.write('Target,' + target + '\n')
                output.write('Parameters'  + ''.join([separator + str(item[0])  for item in outputDict[what][target]]) + os.linesep)
                output.write(what + ''.join([separator + '%.8E' % item[1] for item in outputDict[what][target]]) + os.linesep)
              output.write(os.linesep)
      for what in outputDict.keys():
        if what.lower() in self.statAcceptedMetric:
          continue
        if what.lower() in self.acceptedMetric:
          self.raiseADebug('Writing parameter rank for metric ' + what)
          for target in outputDict[what].keys():
            output.write('Target,' + target + '\n')
            output.write('Parameters'  + ''.join([separator + str(item[0])  for item in outputDict[what][target]]) + os.linesep)
            output.write(what + ''.join([separator + '%.8E' % item[1] for item in outputDict[what][target]]) + os.linesep)
          output.write(os.linesep)
    output.close()

  def _writeXML(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, None
    """
    if output.isOpen():
      output.close()
    if self.dynamic:
      outFile = Files.returnInstance('DynamicXMLOutput',self)
    else:
      outFile = Files.returnInstance('StaticXMLOutput',self)
    outFile.initialize(output.getFilename(),self.messageHandler,path=output.getPath())
    outFile.newTree('ImportanceRankPP',pivotParam=self.pivotParameter)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for step, outputDict in enumerate(outputResults):
      pivotVal = outputDictionary.keys()[step]
      self._localPrintXML(outFile,outputDict,pivotVal)
      if step == 0:
        self._localPrintPCAInformation(outFile,outputDict,pivotVal)
    outFile.writeFile()
    self.raiseAMessage('ImportanceRank XML printed to "'+output.getFilename()+'"!')

  def _writeDataObject(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, None
    """
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for step, outputDict in enumerate(outputResults):
      if step == 0:
        for what in outputDict.keys():
          if what.lower() not in self.statAcceptedMetric:
            continue
          elif what.lower() == 'pcaindex':
            self.raiseADebug('Dumping ' + what + '. Metadata name = ' + what)
            output.updateMetadata(what,outputDict[what])
          else:
            for target in outputDict[what].keys():
              self.raiseADebug('Dumping ' + target + '-' + what + '. Metadata name = ' + target + '-' + what + '. Targets stored in ' +  target + '-'  + what)
              output.updateMetadata(target + '-' + what, outputDict[what][target])
      appendix = '-'+self.pivotParameter+'-'+str(outputDictionary.keys()[step]) if self.dynamic else ''
      for what in outputDict.keys():
        if what.lower() in self.statAcceptedMetric:
          continue
        if what.lower() in self.acceptedMetric:
          for target in outputDict[what].keys():
            self.raiseADebug('Dumping ' + target + '-' + what + '. Metadata name = ' + target + '-' + what + '. Targets stored in ' +  target + '-'  + what)
            output.updateMetadata(target + '-'  + what+appendix, outputDict[what][target])

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
      @ Out, inputDict, dictionary of the converted data
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
      inputDict = {'targets':{}, 'metadata':{}, 'features':{}}
      for feat in self.features:
        if feat in currentInput.getParaKeys('input'):
          inputDict['features'][feat] = currentInput.getParam('input', feat)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(feat) + ' is listed ImportanceRank postprocessor features, but not found in the provided input!')
      for targetP in self.targets:
        if targetP in currentInput.getParaKeys('output'):
          inputDict['targets'][targetP] = currentInput.getParam('output', targetP)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(targetP) + ' is listed ImportanceRank postprocessor targets, but not found in the provided output!')
      inputDict['metadata'] = currentInput.getAllMetadata()
    # get input from HistorySet DataObject
    if inType in ['HistorySet']:
      if self.pivotParameter is None:
        self.raiseAnError(IOError, self, 'Time-dependent importance ranking is requested (HistorySet) but no pivotParameter got inputted!')
      inputs  = currentInput.getParametersValues('inputs',nodeId = 'ending')
      outputs = currentInput.getParametersValues('outputs',nodeId = 'ending')
      numSteps = len(outputs.values()[0].values()[0])
      self.dynamic = True
      if self.pivotParameter not in currentInput.getParaKeys('output'):
        self.raiseAnError(IOError, self, 'Pivot parameter ' + self.pivotParameter + ' has not been found in output space of data object '+currentInput.name)
      pivotParameter = []
      for step in range(len(outputs.values()[0][self.pivotParameter])):
        currentSnapShot = [outputs[i][self.pivotParameter][step] for i in outputs.keys()]
        if len(set(currentSnapShot)) > 1:
          self.raiseAnError(IOError, self, 'Histories are not syncronized! Please, pre-process the data using Interfaced PostProcessor HistorySetSync!')
        pivotParameter.append(currentSnapShot[-1])
      inputDict = {'timeDepData':OrderedDict.fromkeys(pivotParameter,None)}
      for step in range(numSteps):
        inputDict['timeDepData'][pivotParameter[step]] = {'targets':{},'features':{}}
        for targetP in self.targets:
          if targetP in currentInput.getParaKeys('output') :
            inputDict['timeDepData'][pivotParameter[step]]['targets'][targetP] = np.asarray([outputs[i][targetP][step] for i in outputs.keys()])
          else:
            self.raiseAnError(IOError, self, 'Target ' + targetP + ' has not been found in data object '+currentInput.name)
        for feat in self.features:
          if feat in currentInput.getParaKeys('input'):
            inputDict['timeDepData'][pivotParameter[step]]['features'][feat] = np.asarray([inputs[i][feat][-1] for i in inputs.keys()])
          else:
            self.raiseAnError(IOError, self, 'Feature ' + feat + ' has not been found in data object '+currentInput.name)
        inputDict['timeDepData'][pivotParameter[step]]['metadata'] = currentInput.getAllMetadata()

    # get input from HDF5 Database
    if inType == 'HDF5':
      pass  # to be implemented

    return inputDict

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputDict = self.inputToInternal(inputIn)
    if not self.dynamic:
      outputDict = self.__runLocal(inputDict)
    else:
      # time dependent (actually pivot-dependent)
      outputDict = OrderedDict()
      for pivotParamValue in inputDict['timeDepData'].keys():
        outputDict[pivotParamValue] = self.__runLocal(inputDict['timeDepData'][pivotParamValue])
    return outputDict

  def __runLocal(self, inputDict):
    """
      This method executes the postprocessor action.
      @ In, inputDict, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, dictionary containing the evaluated data
    """
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
      senWeightDict[target] = list(zip(feats,featWeights,self.dimensions))
      senCoeffDict[target] = featCoeffs
    for what in self.what:
      if what.lower() == 'sensitivityindex':
        what = 'sensitivityIndex'
        if what not in outputDict.keys():
          outputDict[what] = {}
        for target in self.targets:
          entries = senWeightDict[target]
          entries.sort(key=lambda x: x[1],reverse=True)
          outputDict[what][target] = entries
      if what.lower() == 'importanceindex':
        what = 'importanceIndex'
        if what not in outputDict.keys():
          outputDict[what] = {}
        for target in self.targets:
          featCoeffs = senCoeffDict[target]
          featWeights = []
          if not self.latentSen:
            for index,feat in enumerate(self.manifest):
              totDim = self.mvnDistribution.dimension
              covIndex = totDim * (self.dimensions[index] - 1) + self.dimensions[index] - 1
              if self.mvnDistribution.covarianceType == 'abs':
                covTarget = featCoeffs[index] * self.mvnDistribution.covariance[covIndex] * featCoeffs[index]
              else:
                covFeature = self.mvnDistribution.covariance[covIndex]*self.mvnDistribution.mu[self.dimensions[index]-1]**2
                covTarget = featCoeffs[index] * covFeature * featCoeffs[index]
              featWeights.append(covTarget)
            featWeights = featWeights/np.sum(featWeights)
            entries = list(zip(self.manifest,featWeights,self.dimensions))
            entries.sort(key=lambda x: x[1],reverse=True)
            outputDict[what][target] = entries
          # if the features type is 'latent', since latentVariables are used to compute the sensitivities
          # the covariance for latentVariances are identity matrix
          else:
            entries = senWeightDict[target]
            entries.sort(key=lambda x: x[1],reverse=True)
            outputDict[what][target] = entries
      #calculate PCA index
      if what.lower() == 'pcaindex':
        if not self.latentSen:
          self.raiseAWarning('pcaIndex can be not requested because no latent variable is provided!')
        else:
          what = 'pcaIndex'
          if what not in outputDict.keys():
            outputDict[what] = {}
          index = [dim-1 for dim in self.dimensions]
          singularValues = self.mvnDistribution.returnSingularValues(index)
          singularValues = list(singularValues/np.sum(singularValues))
          entries = list(zip(self.latent,singularValues,self.dimensions))
          entries.sort(key=lambda x: x[1],reverse=True)
          outputDict[what] = entries

      if what.lower() == 'transformation':
        if self.transformation:
          what = 'transformation'
          if what not in outputDict.keys():
            outputDict[what] = {}
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          transformMatrix = self.mvnDistribution.transformationMatrix(index)
          for ind,var in enumerate(self.manifest):
            entries = list(zip(self.latent,transformMatrix[manifestIndex[ind]],self.latentDim))
            outputDict[what][var] = entries
        else:
          self.raiseAnError(IOError,'Unable to output the transformation matrix, please provide both "manifest" and "latent" variables in XML node "features" in',self.printTag)
      if what.lower() == 'inversetransformation':
        if self.transformation:
          what = 'inverseTransformation'
          if what not in outputDict.keys():
            outputDict[what] = {}
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          inverseTransformationMatrix = self.mvnDistribution.inverseTransformationMatrix(manifestIndex)
          for ind,var in enumerate(self.latent):
            entries = list(zip(self.manifest,inverseTransformationMatrix[index[ind]],self.manifestDim))
            outputDict[what][var] = entries
        else:
          self.raiseAnError(IOError,'Unable to output the inverse transformation matrix, please provide both "manifest" and "latent" variables in XML node "features" in', self.printTag)

      if what.lower() == 'manifestsensitivity':
        if self.reconstructSen:
          what = 'manifestSensitivity'
          if what not in outputDict.keys():
            outputDict[what] = {}
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
            entries = list(zip(self.manifest,manifestSen,self.manifestDim))
            entries.sort(key=lambda x: abs(x[1]),reverse=True)
            outputDict[what][target] = entries
        elif self.latentSen:
          self.raiseAnError(IOError, 'Unable to reconstruct the sensitivities for manifest variables, this is because no manifest variable is provided in',self.printTag)
        else:
          self.raiseAWarning('No latent variables, and there is no need to reconstruct the sensitivities for manifest variables!')

      # To be implemented
      #if what == 'CumulativeSenitivityIndex':
      #  self.raiseAnError(NotImplementedError,'CumulativeSensitivityIndex is not yet implemented for ' + self.printTag)
      #if what == 'CumulativeImportanceIndex':
      #  self.raiseAnError(NotImplementedError,'CumulativeImportanceIndex is not yet implemented for ' + self.printTag)

    return outputDict
