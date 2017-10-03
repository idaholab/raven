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

#External Modules---------------------------------------------------------------
import numpy as np
import os
import copy
from collections import OrderedDict, defaultdict
from sklearn.linear_model import LinearRegression
import six
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData
import Files
import Runners
#Internal Modules End-----------------------------------------------------------


#class BasicStatisticsInput(InputData.ParameterInput):
#  """
#    Class for reading the Basic Statistics block
#  """

#BasicStatisticsInput.createClass("PostProcessor", False, baseNode=ModelInput)
#BasicStatisticsInput.addSub(WhatInput)
#BiasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType) #bool
#BasicStatisticsInput.addSub(BiasedInput)
#ParameterInput = InputData.parameterInputFactory("parameters", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(ParameterInput)
#MethodsToRunInput = InputData.parameterInputFactory("methodsToRun", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(MethodsToRunInput)
#FunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(FunctionInput)
#PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(PivotParameterInput)

#
class BasicStatistics(PostProcessor):
  """
    BasicStatistics filter class. It computes all the most popular statistics
  """

  scalarVals = ['expectedValue',
                'minimum',
                'maximum',
                'median',
                'variance',
                'sigma',
                'percentile',
                'variationCoefficient',
                'skewness',
                'kurtosis',
                'samples']
  vectorVals = ['sensitivity',
                'covariance',
                'pearson',
                'NormalizedSensitivity',
                'VarianceDependentSensitivity']

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
    inputSpecification = super(BasicStatistics, cls).getInputSpecification()

    for scalar in cls.scalarVals:
      scalarSpecification = InputData.parameterInputFactory(scalar, contentType=InputData.StringListType)
      if scalar == 'percentile':
        #percent is a string type because otherwise we can't tell 95.0 from 95
        # which matters because the number is used in output.
        scalarSpecification.addParam("percent", InputData.StringType)
      inputSpecification.addSub(scalarSpecification)

    for vector in cls.vectorVals:
      vectorSpecification = InputData.parameterInputFactory(vector)
      features = InputData.parameterInputFactory('features',
                                contentType=InputData.StringListType)
      vectorSpecification.addSub(features)
      targets = InputData.parameterInputFactory('targets',
                                contentType=InputData.StringListType)
      vectorSpecification.addSub(targets)
      inputSpecification.addSub(vectorSpecification)

    allInput = InputData.parameterInputFactory('all', contentType=InputData.StringType)
    allInput.addSub(features)
    allInput.addSub(targets)
    inputSpecification.addSub(allInput)

    pivotParameterInput = InputData.parameterInputFactory('pivotParameter', contentType=InputData.StringType)
    inputSpecification.addSub(pivotParameterInput)

    methodsToRunInput = InputData.parameterInputFactory("methodsToRun", contentType=InputData.StringType)
    inputSpecification.addSub(methodsToRunInput)

    biasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType) #bool
    inputSpecification.addSub(biasedInput)

    ## TODO: Fill this in with the appropriate tags

    # inputSpecification.addSub(WhatInput)
    # BiasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType) #bool
    # inputSpecification.addSub(BiasedInput)
    # ParameterInput = InputData.parameterInputFactory("parameters", contentType=InputData.StringType)
    # inputSpecification.addSub(ParameterInput)
    # FunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
    # inputSpecification.addSub(FunctionInput)
    # PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
    # inputSpecification.addSub(PivotParameterInput)

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.parameters = {}  # parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.acceptedCalcParam = self.scalarVals + self.vectorVals
    self.what = self.acceptedCalcParam  # what needs to be computed... default...all
    self.methodsToRun = []  # if a function is present, its outcome name is here stored... if it matches one of the known outcomes, the pp is going to use the function to compute it
    self.externalFunction = []
    self.printTag = 'POSTPROCESSOR BASIC STATISTIC'
    self.addAssemblerObject('Function','-1', True)
    self.biased = False # biased statistics?
    self.pivotParameter = None # time-dependent statistics pivot parameter
    self.dynamic        = False # is it time-dependent?

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, inputDict, dict, dictionary of the converted data
    """
    # each post processor knows how to handle the coming inputs. The BasicStatistics postprocessor accept all the input type (files (csv only), hdf5 and datas
    self.dynamic = False
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0:
      self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")

    if type(currentInput).__name__ =='dict':
      if 'targets' not in currentInput.keys() and 'timeDepData' not in currentInput.keys():
        self.raiseAnError(IOError, 'Did not find targets or timeDepData in input dictionary')
      return currentInput
    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'BasicStatistics postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)
    if currentInput.type in ['PointSet']:
      inputDict = {'targets':{},'metadata':currentInput.getAllMetadata()}
      for targetP in self.parameters['targets']:
        if   targetP in currentInput.getParaKeys('input'):
          inputDict['targets'][targetP] = currentInput.getParam('input' , targetP, nodeId = 'ending')
        elif targetP in currentInput.getParaKeys('output'):
          inputDict['targets'][targetP] = currentInput.getParam('output', targetP, nodeId = 'ending')
        else:
          self.raiseAnError(IOError, self, 'Target ' + targetP + ' has not been found in data object '+currentInput.name)
    else:
      if self.pivotParameter is None:
        self.raiseAnError(IOError, self, 'Time-dependent statistics is requested (HistorySet) but no pivotParameter got inputted!')
      inputs, outputs  = currentInput.getParametersValues('inputs',nodeId = 'ending'), currentInput.getParametersValues('outputs',nodeId = 'ending')
      nTs, self.dynamic = len(outputs.values()[0].values()[0]), True
      if self.pivotParameter not in currentInput.getParaKeys('output'):
        self.raiseAnError(IOError, self, 'Pivot parameter ' + self.pivotParameter + ' has not been found in output space of data object '+currentInput.name)
      pivotParameter =  six.next(six.itervalues(outputs))[self.pivotParameter]
      self.raiseAMessage("Starting recasting data for time-dependent statistics")
      targetInput  = []
      targetOutput = []
      for targetP in self.parameters['targets']:
        if targetP in currentInput.getParaKeys('output'):
          targetOutput.append(targetP)
        elif targetP in currentInput.getParaKeys('input'):
          targetInput.append(targetP)
        else:
          self.raiseAnError(IOError, self, 'Target ' + targetP + ' has not been found in data object '+currentInput.name)
      inputDict = {}
      inputDict['timeDepData'] = OrderedDict((el,defaultdict(dict)) for el in pivotParameter)
      for targetP in targetInput:
        inputValues = np.asarray([val[targetP][-1] for val in inputs.values()])
        for ts in range(nTs):
          inputDict['timeDepData'][pivotParameter[ts]]['targets'][targetP] = inputValues
      metadata = currentInput.getAllMetadata()
      for cnt, targetP in enumerate(targetOutput):
        outputValues = np.asarray([val[targetP] for val in outputs.values()])
        if len(outputValues.shape) != 2:
          self.raiseAnError(IOError, 'Histories are not syncronized! Please, pre-process the data using Interfaced PostProcessor HistorySetSync!')
        for ts in range(nTs):
          inputDict['timeDepData'][pivotParameter[ts]]['targets'][targetP] = outputValues[:,ts]
          if cnt == 0:
            inputDict['timeDepData'][pivotParameter[ts]]['metadata'] = metadata
    self.raiseAMessage("Recasting performed")
    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the BasicStatistic pp. In here the working dir is
      grepped.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    #construct a list of all the parameters that have requested values into self.allUsedParams
    self.allUsedParams = set()
    #first collect parameters for which scalar values were requested
    for scalar in self.scalarVals:
      if scalar in self.toDo.keys():
        #special treatment of percentile since the user can specify the percents directly
        if scalar == 'percentile':
          for pct,targs in self.toDo[scalar].items():
            self.allUsedParams.update(targs)
        else:
          self.allUsedParams.update(self.toDo[scalar])
    #second collect parameters for which matrix values were requested, either as targets or features
    for vector in self.vectorVals:
      if vector in self.toDo.keys():
        for entry in self.toDo[vector]:
          self.allUsedParams.update(entry['targets'])
          self.allUsedParams.update(entry['features'])
    #for backward compatibility, compile the full list of parameters used in Basic Statistics calculations
    self.parameters['targets'] = list(self.allUsedParams)
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    self.toDo = {}
    for child in paramInput.subparts:
      tag = child.getName()
      #because percentile is strange (has an attached parameter), we address it first
      if tag == 'percentile':
        #get targets
        targets = set(child.value)
        #what if user didn't give any targets?
        if len(targets)<1:
          self.raiseAWarning('No targets were specified in text of <'+tag+'>!  Skipping metric...')
          continue
        #prepare storage dictionary, keys are percentiles, values are set(targets)
        if 'percentile' not in self.toDo.keys():
          self.toDo['percentile']={}
          self.parameters['percentile_map'] = {}
        if 'percent' not in child.parameterValues:
          floatPercentile = [float(5),float(95)]
          self.parameters['percentile_map'][floatPercentile[0]] = '5'
          self.parameters['percentile_map'][floatPercentile[1]] = '95'
        else:
          #user specified a percentage!
          strPercent = child.parameterValues['percent']
          floatPercentile = [float(strPercent)]
          self.parameters['percentile_map'][floatPercentile[0]] = strPercent
        for reqPercent in floatPercentile:
          if reqPercent in self.toDo['percentile'].keys():
            self.toDo['percentile'][reqPercent].update(targets)
          else:
            self.toDo['percentile'][reqPercent] = set(targets)
      elif tag in self.scalarVals:
        if tag in self.toDo.keys():
          self.toDo[tag].update(set(child.value))
        else:
          self.toDo[tag] = set(child.value)
      elif tag in self.vectorVals:
        self.toDo[tag] = [] #'inputs':[],'outputs':[]}
        tnode = child.findFirst('targets')
        if tnode is None:
          self.raiseAnError('Request for vector value <'+tag+'> requires a "targets" node, and none was found!')
        fnode = child.findFirst('features')
        if fnode is None:
          self.raiseAnError('Request for vector value <'+tag+'> requires a "features" node, and none was found!')
        if tag in self.toDo.keys():
          # we're storing toDo[tag] as a list of dictionaries.  This is because the user might specify multiple
          #   nodes with the same metric (tag), but with different targets and features.  For instance, the user might
          #   want the sensitivity of A and B to X and Y, and the sensitivity of C to W and Z, but not the sensitivity
          #   of A to W.  If we didn't keep them separate, we could potentially waste a fair number of calculations.
          self.toDo[tag].append({'targets':set(fnode.value),
                            'features':set(tnode.value)})
        else:
          self.toDo[tag] = [{'targets':set(fnode.value),
                            'features':set(tnode.value)}]
      elif tag == 'all':
        #do all the metrics
        #establish targets and features
        # - as currently done, we only do the scalar metrics for the targets
        #   and features are for the matrix operations
        tnode = child.findFirst('targets')
        if tnode is None:
          self.raiseAnError(IOError,'When using "all" node, you must specify a "targets" and a "features" node!  "targets" is missing.')
        fnode = child.findFirst('features')
        if fnode is None:
          self.raiseAnError(IOError,'When using "all" node, you must specify a "targets" and a "features" node!  "features" is missing.')
        targets = set(tnode.value)
        features = set(fnode.value)
        for scalar in self.scalarVals:
          #percentile is a little different
          if scalar == 'percentile':
            if scalar not in self.toDo.keys():
              self.toDo[scalar] = {}
              self.parameters[scalar+'_map'] = {}
            for pct in [float(5),float(95)]:
              self.parameters['percentile_map'][pct] = str(int(pct))
              if pct in self.toDo[scalar].keys():
                self.toDo[scalar][pct].update(targets)
              else:
                self.toDo[scalar][pct] = set(targets)
          #other scalars are simple
          else:
            if scalar not in self.toDo.keys():
              self.toDo[scalar] = set()
            self.toDo[scalar].update(set(tnode.value))
        for vector in self.vectorVals:
          if vector not in self.toDo.keys():
            self.toDo[vector] = []
          self.toDo[vector].append({'targets':set(fnode.value),
                                 'features':set(tnode.value)})
      elif tag == "biased":
        if child.value.lower() in utils.stringsThatMeanTrue():
          self.biased = True
      elif tag == "pivotParameter":
        self.pivotParameter = child.value
      else:
        self.raiseAWarning('Unrecognized node in BasicStatistics "',tag,'" has been ignored!')
    assert (len(self.toDo)>0), self.raiseAnError(IOError, 'BasicStatistics needs parameters to work on! Please check input for PP: ' + self.name)

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")

    outputDictionary = evaluation[1]

    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam:
        methodToTest.append(key)
    if isinstance(output,Files.File):
      availExtens = ['xml','csv']
      outputExtension = output.getExt().lower()
      if outputExtension not in availExtens:
        self.raiseAMessage('BasicStatistics did not recognize extension ".'+str(outputExtension)+'" as ".xml", so writing text output...')
      output.setPath(self._workingDir)
      self.raiseADebug('Writing statistics output in file named ' + output.getAbsFile())
      output.open('w')
      if outputExtension == 'xml':
        self._writeXML(output,outputDictionary,methodToTest)
      else:
        separator = '   ' if outputExtension != 'csv' else ','
        self._writeText(output,outputDictionary,methodToTest,separator)
    elif output.type in ['PointSet','HistorySet']:
      self.raiseADebug('Dumping output in data object named ' + output.name)
      outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
      for ts, outputDict in enumerate(outputResults):
        appendix = '-'+self.pivotParameter+'-'+str(outputDictionary.keys()[ts]) if self.dynamic else ''
        for what in outputDict.keys():
          if what not in self.vectorVals + methodToTest:
            for targetP in outputDict[what].keys():
              self.raiseADebug('Dumping variable ' + targetP + '. Parameter: ' + what + '. Metadata name = ' + targetP + '-' + what)
              output.updateMetadata(targetP + '-' + what + appendix, outputDict[what][targetP])
          else:
            if what not in methodToTest and len(self.allUsedParams) > 1:
              self.raiseADebug('Dumping vector metric',what)
              output.updateMetadata(what.replace("|","-") + appendix, outputDict[what])
        if self.externalFunction:
          self.raiseADebug('Dumping External Function results')
          for what in self.methodsToRun:
            if what not in self.acceptedCalcParam:
              output.updateMetadata(what + appendix, outputDict[what])
              self.raiseADebug('Dumping External Function parameter ' + what)
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def _writeText(self,output,outputDictionary,methodToTest,separator='  '):
    """
      Defines the method for writing the basic statistics to a text file (space and newline delimited)
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary of statistics values (or list of the same if self.dynamic)
      @ In, methodToTest, list, strings of methods to test
      @ In, separator, string, optional, separator string (e.g. for csv use ",")
      @ Out, None
    """
    if self.dynamic:
      output.write('Dynamic BasicStatistics'+ separator+ 'Pivot Parameter' + separator + self.pivotParameter + separator + os.linesep)
    quantitiesToWrite = {}
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    longestParam = max(list(len(param) for param in self.allUsedParams)+[9]) #9 is for 'Metric:'
    # use format functions to make writing matrices easier
    paramFormat = ('{:>'+str(longestParam)+'.'+str(longestParam)+'}').format
    for ts, outputDict in enumerate(outputResults):
      if self.dynamic:
        output.write('Pivot Value' +separator+ str(outputDictionary.keys()[ts]) + os.linesep)
      # do scalars metrics first
      #header
      haveScalars = list(scalar for scalar in self.scalarVals if scalar in outputDict.keys())
      if 'percentile_map' in self.parameters and len(self.parameters['percentile_map']) >0 :
        haveScalars = haveScalars + ['percentile_'+val for val in self.parameters['percentile_map'].values()]
      valueStrFormat = ('{:^22.22}').format
      valueFormat    = '{:+.15e}'.format
      if len(haveScalars) > 0:
        longestScalar = max(18,max(len(scalar) for scalar in haveScalars))
        output.write(paramFormat('Metric:') + separator)
        output.write(separator.join(valueStrFormat(scalar) for scalar in haveScalars) + os.linesep)
        #body
        for param in self.allUsedParams:
          output.write(paramFormat(param) + separator)
          values = [None]*len(haveScalars)
          for s,scalar in enumerate(haveScalars):
            if param in outputDict.get(scalar,{}).keys():
              values[s] = valueFormat(outputDict[scalar][param])
            else:
              values[s] = valueStrFormat('---')
          output.write(separator.join(values) + os.linesep)
      # then do vector metrics (matrix style)
      haveVectors = list(vector for vector in self.vectorVals if vector in outputDict.keys())
      for vector in haveVectors:
        #label
        output.write(os.linesep + os.linesep)
        output.write(vector+':'+os.linesep)
        #header
        vecTargets = sorted(outputDict[vector].keys())
        output.write(separator.join(valueStrFormat(v) for v in [' ']+vecTargets)+os.linesep)
        #populate feature list
        vecFeatures = set()
        list(vecFeatures.update(set(outputDict[vector][t].keys())) for t in vecTargets)
        vecFeatures = sorted(list(vecFeatures))
        #body
        for feature in vecFeatures:
          output.write(valueStrFormat(feature)+separator)
          values = [valueStrFormat('---')]*len(vecTargets)
          for t,target in enumerate(vecTargets):
            if feature in outputDict[vector][target].keys():
              values[t] = valueFormat(outputDict[vector][target][feature])
          output.write(separator.join(values)+os.linesep)

  def _writeXML(self,origOutput,outputDictionary,methodToTest):
    """
      Defines the method for writing the basic statistics to a .xml file.
      @ In, origOutput, File object, file to write
      @ In, outputDictionary, dict, dictionary of statistics values
      @ In, methodToTest, list, strings of methods to test
      @ Out, None
    """
    #create XML output with same path as original output
    if origOutput.isOpen():
      origOutput.close()
    if self.dynamic:
      output = Files.returnInstance('DynamicXMLOutput',self)
    else:
      output = Files.returnInstance('StaticXMLOutput',self)
    output.initialize(origOutput.getFilename(),self.messageHandler,path=origOutput.getPath())
    output.newTree('BasicStatisticsPP',pivotParam=self.pivotParameter)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for ts, outputDict in enumerate(outputResults):
      pivotVal = outputDictionary.keys()[ts]
      for t,target in enumerate(self.allUsedParams):
        #do scalars first
        for metric in self.scalarVals:
          #TODO percentile
          if metric == 'percentile':
            for key in outputDict.keys():
              if key.startswith(metric) and target in outputDict[key].keys():
                output.addScalar(target,key,outputDict[key][target],pivotVal=pivotVal)
          elif metric in outputDict.keys() and target in outputDict[metric]:
            output.addScalar(target,metric,outputDict[metric][target],pivotVal=pivotVal)
        #do matrix values
        for metric in self.vectorVals:
          if metric in outputDict.keys() and target in outputDict[metric]:
            output.addVector(target,metric,outputDict[metric][target],pivotVal=pivotVal)

    output.writeFile()

  def __computeVp(self,p,weights):
    """
      Compute the sum of p-th power of weights
      @ In, p, int, the power
      @ In, weights, list or numpy.array, weights
      @ Out, vp, float, the sum of p-th power of weights
    """
    vp = np.sum(np.power(weights,p))
    return vp

  def __computeUnbiasedCorrection(self,order,weightsOrN):
    """
      Compute unbiased correction given weights and momement order
      Reference paper:
      Lorenzo Rimoldini, "Weighted skewness and kurtosis unbiased by sample size", http://arxiv.org/pdf/1304.6564.pdf
      @ In, order, int, moment order
      @ In, weightsOrN, list/numpy.array or int, if list/numpy.array -> weights else -> number of samples
      @ Out, corrFactor, float (order <=3) or tuple of floats (order ==4), the unbiased correction factor
    """
    if order > 4:
      self.raiseAnError(RuntimeError,"computeUnbiasedCorrection is implemented for order <=4 only!")
    if type(weightsOrN).__name__ not in ['int','int8','int16','int64','int32']:
      if order == 2:
        V1, v1Square, V2 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**2.0, self.__computeVp(2, weightsOrN)
        corrFactor   = v1Square/(v1Square-V2)
      elif order == 3:
        V1, v1Cubic, V2, V3 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**3.0, self.__computeVp(2, weightsOrN), self.__computeVp(3, weightsOrN)
        corrFactor   =  v1Cubic/(v1Cubic-3.0*V2*V1+2.0*V3)
      elif order == 4:
        V1, v1Square, V2, V3, V4 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**2.0, self.__computeVp(2, weightsOrN), self.__computeVp(3, weightsOrN), self.__computeVp(4, weightsOrN)
        numer1 = v1Square*(v1Square**2.0-3.0*v1Square*V2+2.0*V1*V3+3.0*V2**2.0-3.0*V4)
        numer2 = 3.0*v1Square*(2.0*v1Square*V2-2.0*V1*V3-3.0*V2**2.0+3.0*V4)
        denom = (v1Square-V2)*(v1Square**2.0-6.0*v1Square*V2+8.0*V1*V3+3.0*V2**2.0-6.0*V4)
        corrFactor = numer1/denom ,numer2/denom
    else:
      if   order == 2:
        corrFactor   = float(weightsOrN)/(float(weightsOrN)-1.0)
      elif order == 3:
        corrFactor   = (float(weightsOrN)**2.0)/((float(weightsOrN)-1)*(float(weightsOrN)-2))
      elif order == 4:
        corrFactor = (float(weightsOrN)*(float(weightsOrN)**2.0-2.0*float(weightsOrN)+3.0))/((float(weightsOrN)-1)*(float(weightsOrN)-2)*(float(weightsOrN)-3)),(3.0*float(weightsOrN)*(2.0*float(weightsOrN)-3.0))/((float(weightsOrN)-1)*(float(weightsOrN)-2)*(float(weightsOrN)-3))
    return corrFactor

  def _computeKurtosis(self,arrayIn,expValue,variance,pbWeight=None):
    """
      Method to compute the Kurtosis (fisher) of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the Kurtosis needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Kurtosis of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(4,pbWeight) if not self.biased else 1.0
      if not self.biased:
        result = -3.0 + ((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,4.0),pbWeight))*unbiasCorr[0]-unbiasCorr[1]*np.power(((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,2.0),pbWeight))),2.0))/np.power(variance,2.0)
      else:
        result = -3.0 + ((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,4.0),pbWeight))*unbiasCorr)/np.power(variance,2.0)
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(4,len(arrayIn)) if not self.biased else 1.0
      if not self.biased:
        result = -3.0 + ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**4)*unbiasCorr[0]-unbiasCorr[1]*(np.average((arrayIn - expValue)**2))**2.0)/(variance)**2.0
      else:
        result = -3.0 + ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**4)*unbiasCorr)/(variance)**2.0
    return result

  def _computeSkewness(self,arrayIn,expValue,variance,pbWeight=None):
    """
      Method to compute the skewness of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the skewness needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the skewness of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(3,pbWeight) if not self.biased else 1.0
      result = (1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,3.0),pbWeight))*unbiasCorr/np.power(variance,1.5)
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(3,len(arrayIn)) if not self.biased else 1.0
      result = ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**3)*unbiasCorr)/np.power(variance,1.5)
    return result

  def _computeVariance(self,arrayIn,expValue,pbWeight=None):
    """
      Method to compute the Variance (fisher) of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the Variance needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Variance of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(2,pbWeight) if not self.biased else 1.0
      result = (1.0/self.__computeVp(1,pbWeight))*np.average((arrayIn - expValue)**2,weights= pbWeight)*unbiasCorr
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(2,len(arrayIn)) if not self.biased else 1.0
      result = np.average((arrayIn - expValue)**2)*unbiasCorr
    return result

  def _computeSigma(self,arrayIn,variance,pbWeight=None):
    """
      Method to compute the sigma of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the sigma needs to be estimated
      @ In, variance, float, variance of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, sigma, float, the sigma of the array of data
    """
    return np.sqrt(variance)

  def _computeWeightedPercentile(self,arrayIn,pbWeight,percent=0.5):
    """
      Method to compute the weighted percentile in a array of data
      @ In, arrayIn, list/numpy.array, the array of values from which the percentile needs to be estimated
      @ In, pbWeight, list/numpy.array, the reliability weights that correspond to the values in 'array'
      @ In, percent, float, the percentile that needs to be computed (between 0.01 and 1.0)
      @ Out, result, float, the percentile
    """
    idxs                   = np.argsort(np.asarray(zip(pbWeight,arrayIn))[:,1])
    # Inserting [0.0,arrayIn[idxs[0]]] is needed when few samples are generated and
    # a percentile that is < that the first pb weight is requested. Otherwise the median
    # is returned (that is wrong).
    sortedWeightsAndPoints = np.insert(np.asarray(zip(pbWeight[idxs],arrayIn[idxs])),0,[0.0,arrayIn[idxs[0]]],axis=0)
    weightsCDF             = np.cumsum(sortedWeightsAndPoints[:,0])
    try:
      index = utils.find_le_index(weightsCDF,percent)
      result = sortedWeightsAndPoints[index,1]
    except ValueError:
      result = np.median(arrayIn)
    return result

  def __runLocal(self, input):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  input, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    pbWeights, pbPresent  = {'realization':None}, False
    if self.externalFunction:
      # there is an external function
      for what in self.methodsToRun:
        outputDict[what] = self.externalFunction.evaluate(what, input['targets'])
        # check if "what" corresponds to an internal method
        if what in self.acceptedCalcParam:
          if what not in ['pearson', 'covariance', 'NormalizedSensitivity', 'VarianceDependentSensitivity', 'sensitivity']:
            if type(outputDict[what]) != dict:
              self.raiseAnError(IOError, 'BasicStatistics postprocessor: You have overwritten the "' + what + '" method through an external function, it must be a dictionary!!')
          else:
            if type(outputDict[what]) != np.ndarray:
              self.raiseAnError(IOError, 'BasicStatistics postprocessor: You have overwritten the "' + what + '" method through an external function, it must be a numpy.ndarray!!')
            if len(outputDict[what].shape) != 2:
              self.raiseAnError(IOError, 'BasicStatistics postprocessor: You have overwritten the "' + what + '" method through an external function, it must be a 2D numpy.ndarray!!')
    # setting some convenience values
    parameterSet = list(self.allUsedParams)
    if 'metadata' in input.keys():
      pbPresent = 'ProbabilityWeight' in input['metadata'].keys() if 'metadata' in input.keys() else False
    if not pbPresent:
      pbWeights['realization'] = None
      if 'metadata' in input.keys():
        if 'SamplerType' in input['metadata'].keys():
          if input['metadata']['SamplerType'][0] != 'MonteCarlo' :
            self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
        else:
          self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights. Assuming unit weights instead...')
    else:
      pbWeights['realization'] = input['metadata']['ProbabilityWeight']/np.sum(input['metadata']['ProbabilityWeight'])
    #This section should take the probability weight for each sampling variable
    pbWeights['SampledVarsPbWeight'] = {'SampledVarsPbWeight':{}}
    if 'metadata' in input.keys():
      for target in parameterSet:
        if 'ProbabilityWeight-'+target in input['metadata'].keys():
          pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target] = np.asarray(input['metadata']['ProbabilityWeight-'+target])
          pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target][:] = pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target][:]/np.sum(pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target])

    #establish a dict of indices to parameters and vice versa
    parameter2index = dict((param,p) for p,param in enumerate(input['targets'].keys()))
    for p,param in enumerate(input['targets'].keys()):
      parameter2index[param] = p

    #storage dictionary for skipped metrics
    self.skipped = {}

    #construct a dict of required computations
    needed = dict((metric,set()) for metric in self.scalarVals) #for each metric (keys), the list of parameters we need that value for
    needed.update(dict((metric,{'targets':set(),'features':set()}) for metric in self.vectorVals))
    #percentile is a special exception
    if 'percentile' in needed.keys():
      needed['percentile'] = {}
    #add things requested by the user
    #start by adding the exact request by the user, then add the dependencies
    for metric,params in self.toDo.items():
      #percentile is a special case, and it neither relies on anything nor is relied upon by anything
      if metric == 'percentile':
        for pct,targets in params.items():
          needed[metric][pct] = targets
      elif type(params) == set:
        #scalar parameter
        needed[metric].update(params)
      elif type(params) == list and type(params[0]) == dict:
        # vector parameter
        needed[metric] = {'targets':set(),'features':set()}
        for entry in params:
          needed[metric]['targets'].update(entry['targets'])
          needed[metric]['features'].update(entry['features'])
      else:
        self.raiseAWarning('Unrecognized format for metric "'+metric+'!  Expected "set" or "dict" but got',type(params))
    # variable                     | needs                  | needed for
    # --------------------------------------------------------------------
    # skewness needs               | expectedValue,variance |
    # kurtosis needs               | expectedValue,variance |
    # median needs                 |                        |
    # percentile needs             |                        |
    # maximum needs                |                        |
    # minimum needs                |                        |
    # covariance needs             |                        | pearson,VarianceDependentSensitivity,NormalizedSensitivity
    # NormalizedSensitivity        | covariance,VarDepSens  |
    # VarianceDependentSensitivity | covariance             | NormalizedSensitivity
    # sensitivity needs            |                        |
    # pearson needs                | covariance             |
    # sigma needs                  | variance               | variationCoefficient
    # variance                     | expectedValue          | sigma, skewness, kurtosis
    # expectedValue                |                        | variance, variationCoefficient, skewness, kurtosis
    needed['sigma'].update(needed.get('variationCoefficient'))
    needed['variance'].update(needed.get('sigma',set()))
    needed['expectedValue'].update(needed.get('sigma',set()))
    needed['expectedValue'].update(needed.get('variationCoefficient',set()))
    needed['expectedValue'].update(needed.get('variance',set()))
    needed['expectedValue'].update(needed.get('skewness',set()))
    needed['expectedValue'].update(needed.get('kurtosis',set()))
    if 'NormalizedSensitivity' in needed.keys():
      needed['expectedValue'].update(needed['NormalizedSensitivity']['targets'])
      needed['expectedValue'].update(needed['NormalizedSensitivity']['features'])
      needed['covariance']['targets'].update(needed['NormalizedSensitivity']['targets'])
      needed['covariance']['features'].update(needed['NormalizedSensitivity']['features'])
      needed['VarianceDependentSensitivity']['targets'].update(needed['NormalizedSensitivity']['targets'])
      needed['VarianceDependentSensitivity']['features'].update(needed['NormalizedSensitivity']['features'])
    if 'pearson' in needed.keys():
      needed['covariance']['targets'].update(needed['pearson']['targets'])
      needed['covariance']['features'].update(needed['pearson']['features'])
    if 'VarianceDependentSensitivity' in needed.keys():
      needed['covariance']['targets'].update(needed['VarianceDependentSensitivity']['targets'])
      needed['covariance']['features'].update(needed['VarianceDependentSensitivity']['features'])
    #
    # BEGIN actual calculations
    #
    calculations = {}
    # do things in order to preserve prereqs
    # TODO many of these could be sped up through vectorization
    # TODO additionally, this could be done with less code duplication, probably
    #################
    # SCALAR VALUES #
    #################
    def startMetric(metric):
      """
        Common starting for each metric calculation.
        @ In, metric, string, name of metric
        @ Out, None
      """
      if len(needed[metric])>0:
        self.raiseADebug('Starting "'+metric+'"...')
        calculations[metric]={}
    #
    # samples
    #
    metric = 'samples'
    startMetric(metric)
    for targetP in needed[metric]:
      calculations[metric][targetP] = len(utils.first(input['targets'].values()))
    #
    # expected value
    #
    metric = 'expectedValue'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[metric][targetP] = np.average(input['targets'][targetP], weights = relWeight)
      else:
        relWeight  = None
        calculations[metric][targetP] = np.mean(input['targets'][targetP])
    #
    # variance
    #
    metric = 'variance'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeVariance(input['targets'][targetP],calculations['expectedValue'][targetP],pbWeight=relWeight)
      #sanity check
      if (calculations[metric][targetP] == 0):
        self.raiseAWarning('The variable: ' + targetP + ' has zero variance! Please check your input in PP: ' + self.name)
    #
    # sigma
    #
    metric = 'sigma'
    startMetric(metric)
    for targetP in needed[metric]:
      if calculations['variance'][targetP] == 0:
        #np.Infinity:
        self.raiseAWarning('The variable: ' + targetP + ' has zero sigma! Please check your input in PP: ' + self.name)
        calculations[metric][targetP] = 0.0
      else:
        calculations[metric][targetP] = self._computeSigma(input['targets'][targetP],calculations['variance'][targetP])
    #
    # coeff of variation (sigma/mu)
    #
    metric = 'variationCoefficient'
    startMetric(metric)
    for targetP in needed[metric]:
      if calculations['expectedValue'][targetP] == 0:
        self.raiseAWarning('Expected Value for ' + targetP + ' is zero! Variation Coefficient cannot be calculated, so setting as infinite.')
        calculations[metric][targetP] = np.Infinity
      else:
        calculations[metric][targetP] = calculations['sigma'][targetP]/calculations['expectedValue'][targetP]
    #
    # skewness
    #
    metric = 'skewness'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeSkewness(input['targets'][targetP],calculations['expectedValue'][targetP],calculations['variance'][targetP],pbWeight=relWeight)
    #
    # kurtosis
    #
    metric = 'kurtosis'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeKurtosis(input['targets'][targetP],calculations['expectedValue'][targetP],calculations['variance'][targetP],pbWeight=relWeight)
    #
    # median
    #
    metric = 'median'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[metric][targetP] = self._computeWeightedPercentile(input['targets'][targetP],relWeight,percent=0.5)
      else:
        calculations[metric][targetP] = np.median(input['targets'][targetP])
    #
    # maximum
    #
    metric = 'maximum'
    startMetric(metric)
    for targetP in needed[metric]:
      calculations[metric][targetP] = np.amax(input['targets'][targetP])
    #
    # minimum
    #
    metric = 'minimum'
    startMetric(metric)
    for targetP in needed[metric]:
      calculations[metric][targetP] = np.amin(input['targets'][targetP])
    #
    # percentile
    #
    metric = 'percentile'
    self.raiseADebug('Starting "'+metric+'"...')
    for percent,targets in needed[metric].items():
      self.raiseADebug('...',str(percent),'...')
      label = metric+'_'+self.parameters['percentile_map'][percent]
      calculations[label] = {}
      for targetP in targets:
        if pbPresent:
          relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[label][targetP] = np.percentile(input['targets'][targetP], percent) if not pbPresent else self._computeWeightedPercentile(input['targets'][targetP],relWeight,percent=float(percent)/100.0)
    #################
    # VECTOR VALUES #
    #################
    #
    # sensitivity matrix
    #
    def startVector(metric):
      """
        Common method among all metrics for establishing parameters
        @ In, metric, string, the name of the statistics metric to calculate
        @ Out, targets, list(str), list of target parameter names (evaluate metrics for these)
        @ Out, features, list(str), list of feature parameter names (evaluate with respect to these)
        @ Out, skip, bool, if True it means either features or parameters were missing, so don't calculate anything
      """
      # default to skipping, change that if we find criteria
      targets = []
      features = []
      skip = True
      allParams = set(needed[metric]['targets'])
      allParams.update(set(needed[metric]['features']))
      if len(needed[metric]['targets'])>0 and len(allParams)>=2:
        self.raiseADebug('Starting "'+metric+'"...')
        calculations[metric]={}
        targets = needed[metric]['targets']
        features = needed[metric]['features']
        skip = False #True only if we don't have targets and features
        if len(features)<1:
          self.raiseAWarning('No features specified for <'+metric+'>!  Please specify features in a <features> node (see the manual).  Skipping...')
          skip = True
      elif len(needed[metric]['targets']) == 0:
        #unrequested, no message needed
        pass
      elif len(allParams) < 2:
        #insufficient target/feature combinations (usually when only 1 target and 1 feature, and they are the same)
        self.raiseAWarning('A total of',len(allParams),'were provided for metric',metric,'but at least 2 are required!  Skipping...')
      if skip:
        if metric not in self.skipped.keys():
          self.skipped[metric] = {}
        self.skipped[metric].update(needed[metric])
      return targets,features,skip

    metric = 'sensitivity'
    targets,features,skip = startVector(metric)
    #NOTE sklearn expects the transpose of what we usually do in RAVEN, so #samples by #features
    if not skip:
      #for sensitivity matrix, we don't use numpy/scipy methods to calculate matrix operations,
      #so we loop over targets and features
      for t,target in enumerate(targets):
        calculations[metric][target] = {}
        targetVals = input['targets'][target]
        #don't do self-sensitivity
        inpSamples = np.atleast_2d(np.asarray(list(input['targets'][f] for f in features if f!=target))).T
        useFeatures = list(f for f in features if f != target)
        #use regressor coefficients as sensitivity
        regressDict = dict(zip(useFeatures, LinearRegression().fit(inpSamples,targetVals).coef_))
        for f,feature in enumerate(features):
          calculations[metric][target][feature] = 1.0 if feature==target else regressDict[feature]
    #
    # covariance matrix
    #
    metric = 'covariance'
    targets,features,skip = startVector(metric)
    if not skip:
      # because the C implementation is much faster than picking out individual values,
      #   we do the full covariance matrix with all the targets and features.
      # FIXME adding an alternative for users to choose pick OR do all, defaulting to something smart
      #   dependent on the percentage of the full matrix desired, would be better.
      # IF this is fixed, make sure all the features and targets are requested for all the metrics
      #   dependent on this metric
      params = list(set(targets).union(set(features)))
      paramSamples = np.zeros((len(params), utils.first(input['targets'].values()).size))
      pbWeightsList = [None]*len(input['targets'].keys())
      for p,param in enumerate(params):
        dataIndex = parameter2index[param]
        paramSamples[p,:] = input['targets'][param][:]
        pbWeightsList[p] = pbWeights['realization'] if param not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][param]
      pbWeightsList.append(pbWeights['realization'])
      #Note: this is basically "None in pbWeightsList", but
      # using "is None" instead of "== None", which is more reliable
      if True in [x is None for x in pbWeightsList]:
        covar = self.covariance(paramSamples)
      else:
        covar = self.covariance(paramSamples, weights = pbWeightsList)
      calculations[metric]['matrix'] = covar
      calculations[metric]['params'] = params

    def getCovarianceSubset(desired):
      """
        @ In, desired, list(str), list of parameters to extract from covariance matrix
        @ Out, reducedSecond, np.array, reduced covariance matrix
        @ Out, wantedParams, list(str), parameter labels for reduced covar matrix
      """
      wantedIndices = list(calculations['covariance']['params'].index(d) for d in desired)
      wantedParams = list(calculations['covariance']['params'][i] for i in wantedIndices)
      #retain rows, colums
      reducedFirst = calculations['covariance']['matrix'][wantedIndices]
      reducedSecond = reducedFirst[:,wantedIndices]
      return reducedSecond, wantedParams
    #
    # pearson matrix
    #
    # see comments in covariance for notes on C implementation
    metric = 'pearson'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar,reducedParams = getCovarianceSubset(params)
      calculations[metric]['matrix'] = self.corrCoeff(reducedCovar)
      calculations[metric]['params'] = reducedParams
    #
    # VarianceDependentSensitivity matrix
    # The formula for this calculation is coming from: http://www.math.uah.edu/stat/expect/Matrices.html
    # The best linear predictor: L(Y|X) = expectedValue(Y) + cov(Y,X) * [vc(X)]^(-1) * [X-expectedValue(X)]
    # where Y is a vector of outputs, and X is a vector of inputs, cov(Y,X) is the covariance matrix of Y and X,
    # vc(X) is the covariance matrix of X with itself.
    # The variance dependent sensitivity matrix is defined as: cov(Y,X) * [vc(X)]^(-1)
    #
    metric = 'VarianceDependentSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar,reducedParams = getCovarianceSubset(params)
      inputSamples = np.zeros((len(params),utils.first(input['targets'].values()).size))
      pbWeightsList = [None]*len(params)
      for p,param in enumerate(reducedParams):
        inputSamples[p,:] = input['targets'][param][:]
        pbWeightsList[p] = pbWeights['realization'] if param not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][param]
      pbWeightsList.append(pbWeights['realization'])
      for p,param in enumerate(reducedParams):
        calculations[metric][param] = {}
        targCoefs = list(r for r in reducedParams if r!=param)
        inpParams = np.delete(inputSamples,p,axis=0)
        inpCovMatrix = np.delete(reducedCovar,p,axis=0)
        inpCovMatrix = np.delete(inpCovMatrix,p,axis=1)
        outInpCov = np.delete(reducedCovar[p,:],p)
        sensCoefDict = dict(zip(targCoefs,np.dot(outInpCov,np.linalg.pinv(inpCovMatrix))))
        for f,feature in enumerate(reducedParams):
          if param == feature:
            calculations[metric][param][feature] = 1.0
          else:
            calculations[metric][param][feature] = sensCoefDict[feature]
    #
    # Normalized variance dependent sensitivity matrix
    # variance dependent sensitivity  normalized by the mean (% change of output)/(% change of input)
    #
    metric = 'NormalizedSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      reducedCovar,reducedParams = getCovarianceSubset(params)
      for p,param in enumerate(reducedParams):
        calculations[metric][param] = {}
        for f,feature in enumerate(reducedParams):
          expValueRatio = calculations['expectedValue'][feature]/calculations['expectedValue'][param]
          calculations[metric][param][feature] = calculations['VarianceDependentSensitivity'][param][feature]*expValueRatio

    #collect only the requested calculations
    outputDict = {}
    for metric,params in self.toDo.items():
      #TODO someday we might need to expand the "skipped" check to include scalars, but for now
      #   the only reason to skip is if an invalid matrix is requested
      #if percentile, special treatment
      if metric == 'percentile':
        for pct,targets in params.items():
          label = 'percentile_'+self.parameters['percentile_map'][pct]
          outputDict[label] = dict((target,calculations[label][target]) for target in targets)
      #if other scalar, just report the result
      elif metric in self.scalarVals:
        outputDict[metric] = dict((target,calculations[metric][target]) for target in params)
      #if a matrix block, extract desired values
      else:
        if metric in ['pearson','covariance']:
          outputDict[metric] = {}
          for entry in params:
            #check if it was skipped for some reason
            if entry == self.skipped.get(metric,None):
              self.raiseADebug('Metric',metric,'was skipped for parameters',entry,'!  See warnings for details.  Ignoring...')
              continue
            for target in entry['targets']:
              if target not in outputDict[metric].keys():
                outputDict[metric][target] = {}
              targetIndex = calculations[metric]['params'].index(target)
              for feature in entry['features']:
                featureIndex = calculations[metric]['params'].index(feature)
                outputDict[metric][target][feature] = calculations[metric]['matrix'][targetIndex,featureIndex]
        #if matrix but stored in dictionaries, just grab the values
        elif metric in ['sensitivity','NormalizedSensitivity','VarianceDependentSensitivity']:
          outputDict[metric] = {}
          for entry in params:
            #check if it was skipped for some reason
            if entry == self.skipped.get(metric,None):
              self.raiseADebug('Metric',metric,'was skipped for parameters',entry,'!  See warnings for details.  Ignoring...')
              continue
            for target in entry['targets']:
              outputDict[metric][target] = dict((feature,calculations[metric][target][feature]) for feature in entry['features'])

    # print on screen
    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam:
        methodToTest.append(key)
    self.printToScreen(outputDict)
    return outputDict

  def printToScreen(self,outputDict):
    """
      Prints all results of BasicStatistics to screen.
      @ In, outputDict, dict, dictionary of results
      @ Out, None
    """
    self.raiseADebug('BasicStatistics ' + str(self.name) + 'results:')
    for metric,valueDict in outputDict.items():
      self.raiseADebug('BasicStatistics Metric:',metric)
      if metric in self.scalarVals or metric.startswith('percentile'):
        for target,value in valueDict.items():
          self.raiseADebug('   ',target+':',value)
      elif metric in self.vectorVals:
        for target,wrt in valueDict.items():
          self.raiseADebug('   ',target,'with respect to:')
          for feature,value in wrt.items():
            self.raiseADebug('     ',feature+':',value)
      else:
        self.raiseADebug('   ',valueDict)

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputAdapted = self.inputToInternal(inputIn)
    if not self.dynamic:
      outputDict = self.__runLocal(inputAdapted)
    else:
      # time dependent (actually pivot-dependent)
      outputDict = OrderedDict()
      self.raiseADebug('BasicStatistics Pivot-Dependent output:')
      for pivotParamValue in inputAdapted['timeDepData'].keys():
        self.raiseADebug('Pivot Parameter Value: ' + str(pivotParamValue))
        outputDict[pivotParamValue] = self.__runLocal(inputAdapted['timeDepData'][pivotParamValue])


    return outputDict

  def covariance(self, feature, weights = None, rowVar = 1):
    """
      This method calculates the covariance Matrix for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calculated depending on the selection of the inputs.
      @ In,  feature, list/numpy.array, [#targets,#samples]  features' samples
      @ In,  weights, list of list/numpy.array, optional, [#targets,#samples,realizationWeights]  reliability weights, and the last one in the list is the realization weights. Default is None
      @ In,  rowVar, int, optional, If rowVar is non-zero, then each row represents a variable,
                                    with samples in the columns. Otherwise, the relationship is transposed. Default=1
      @ Out, covMatrix, list/numpy.array, [#targets,#targets] the covariance matrix
    """
    X = np.array(feature, ndmin = 2, dtype = np.result_type(feature, np.float64))
    w = np.zeros(feature.shape, dtype = np.result_type(feature, np.float64))
    if X.shape[0] == 1:
      rowVar = 1
    if rowVar:
      N = X.shape[1]
      featuresNumber = X.shape[0]
      axis = 0
      for myIndex in range(featuresNumber):
        if weights is None:
          w[myIndex,:] = np.ones(N)/float(N)
        else:
          w[myIndex,:] = np.array(weights[myIndex],dtype=np.result_type(feature, np.float64))[:] if weights is not None else np.ones(len(w[myIndex,:]),dtype =np.result_type(feature, np.float64))[:]
    else:
      N = X.shape[0]
      featuresNumber = X.shape[1]
      axis = 1
      for myIndex in range(featuresNumber):
        if weights is None:
          w[myIndex,:] = np.ones(N)/float(N)
        else:
          w[:,myIndex] = np.array(weights[myIndex], dtype=np.result_type(feature, np.float64))[:] if weights is not None else np.ones(len(w[:,myIndex]),dtype=np.result_type(feature, np.float64))[:]
    realizationWeights = weights[-1] if weights is not None else np.ones(N)/float(N)
    if N <= 1:
      self.raiseAWarning("Degrees of freedom <= 0")
      return np.zeros((featuresNumber,featuresNumber), dtype = np.result_type(feature, np.float64))
    diff = X - np.atleast_2d(np.average(X, axis = 1 - axis, weights = w)).T
    covMatrix = np.ones((featuresNumber,featuresNumber), dtype = np.result_type(feature, np.float64))
    for myIndex in range(featuresNumber):
      for myIndexTwo in range(featuresNumber):
        # The weights that are used here should represent the joint probability (P(x,y)).
        # Since I have no way yet to compute the joint probability with weights only (eventually I can think to use an estimation of the P(x,y) computed through a 2D histogram construction and weighted a posteriori with the 1-D weights),
        # I decided to construct a weighting function that is defined as Wi = (2.0*Wi,x*Wi,y)/(Wi,x+Wi,y) that respects the constrains of the
        # covariance (symmetric and that the diagonal is == variance) but that is completely arbitrary and for that not used. As already mentioned, I need the joint probability to compute the E[XY] = integral[xy*p(x,y)dxdy]. Andrea
        # for now I just use the realization weights
        #jointWeights = (2.0*weights[myIndex][:]*weights[myIndexTwo][:])/(weights[myIndex][:]+weights[myIndexTwo][:])
        #jointWeights = jointWeights[:]/np.sum(jointWeights)
        if myIndex == myIndexTwo:
          jointWeights = w[myIndex]/np.sum(w[myIndex])
        else:
          jointWeights = realizationWeights/np.sum(realizationWeights)
        fact = self.__computeUnbiasedCorrection(2,jointWeights) if not self.biased else 1.0/np.sum(jointWeights)
        covMatrix[myIndex,myIndexTwo] = np.sum(diff[:,myIndex]*diff[:,myIndexTwo]*jointWeights[:]*fact) if not rowVar else np.sum(diff[myIndex,:]*diff[myIndexTwo,:]*jointWeights[:]*fact)
    return covMatrix

  def corrCoeff(self, covM):
    """
      This method calculates the correlation coefficient Matrix (pearson) for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calcuated depending on the selection of the inputs.
      @ In,  covM, list/numpy.array, [#targets,#targets] covariance matrix
      @ Out, corrMatrix, list/numpy.array, [#targets,#targets] the correlation matrix
    """
    try:
      d = np.diag(covM)
      corrMatrix = covM / np.sqrt(np.multiply.outer(d, d))
    except ValueError:
      # scalar covariance
      # nan if incorrect value (nan, inf, 0), 1 otherwise
      corrMatrix = covM / covM
    # to prevent numerical instability
    return corrMatrix
