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
import copy
import time
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
import Files
import unSupervisedLearning
#Internal Modules End-----------------------------------------------------------

class DataMining(PostProcessor):
  """
    DataMiningPostProcessor class. It will apply the specified KDD algorithms in
    the models to a dataset, each specified algorithm's output can be loaded to
    dataObject.
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
    inputSpecification = super(RavenOutput, cls).getInputSpecification()

    ## TODO: Fill this in with the appropriate tags

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR DATAMINING'

    self.requiredAssObject = (True, (['PreProcessor','Metric'], ['-1','-1']))

    self.solutionExport = None                            ## A data object to
                                                          ## hold derived info
                                                          ## about the algorithm
                                                          ## being performed,
                                                          ## e.g., cluster
                                                          ## centers or a
                                                          ## projection matrix
                                                          ## for dimensionality
                                                          ## reduction methods

    self.labelFeature = None                              ## User customizable
                                                          ## column name for the
                                                          ## labels associated
                                                          ## to a clustering or
                                                          ## a DR algorithm

    self.PreProcessor = None
    self.metric = None

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In , None, None
      @ Out, dict, dictionary of objects needed
    """
    return {'internal':[(None,'jobHandler')]}

  def _localGenerateAssembler(self,initDict):
    """Generates the assembler.
      @ In, initDict, dict, init objects
      @ Out, None
    """
    self.jobHandler = initDict['internal']['jobHandler']

  def inputToInternal(self, currentInp):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, currentInp, list or DataObjects, Some form of data object or list of
        data objects handed to the post-processor
      @ Out, inputDict, dict, An input dictionary this object can process
    """

    if type(currentInp) == list:
      currentInput = currentInp[-1]
    else:
      currentInput = currentInp

    if currentInput.type == 'HistorySet' and self.PreProcessor is None and self.metric is None:
      # for testing time dependent dm - time dependent clustering
      inputDict = {'Features':{}, 'parameters':{}, 'Labels':{}, 'metadata':{}}

      # FIXME, this needs to be changed for asynchronous HistorySet
      if self.pivotParameter in currentInput.getParam('output',1).keys():
        self.pivotVariable = currentInput.getParam('output',1)[self.pivotParameter]
      else:
        self.raiseAnError(ValueError, 'Pivot variable not found in input historyset')
      # end of FIXME

      historyKey = currentInput.getOutParametersValues().keys()
      numberOfSample = len(historyKey)
      numberOfHistoryStep = len(self.pivotVariable)

      if self.initializationOptionDict['KDD']['Features'] == 'input':
        self.raiseAnError(ValueError, 'To perform data mining over input please use SciKitLearn library')
      elif self.initializationOptionDict['KDD']['Features'] in ['output', 'all']:
        features = currentInput.getParaKeys('output')
        features.remove(self.pivotParameter)
      else:
        features = self.initializationOptionDict['KDD']['Features'].split(',')

      for param in features:
        inputDict['Features'][param] = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
        for cnt, keyH in enumerate(historyKey):
          inputDict['Features'][param][cnt,:] = currentInput.getParam('output', keyH)[param]

      inputDict['metadata'] = currentInput.getAllMetadata()
      return inputDict

    if type(currentInp) == dict:
      if 'Features' in currentInput.keys():
        return
    if isinstance(currentInp, Files.File):
      if currentInput.subtype == 'csv':
        self.raiseAnError(IOError, 'CSV File received as an input!')
    if currentInput.type == 'HDF5':
      self.raiseAnError(IOError, 'HDF5 Object received as an input!')

    if self.PreProcessor != None:
      inputDict = {'Features':{}, 'parameters':{}, 'Labels':{}, 'metadata':{}}
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        features = currentInput.getParaKeys('input')
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        features = currentInput.getParaKeys('output')
      else:
        features = self.initializationOptionDict['KDD']['Features'].split(',')

      tempData = self.PreProcessor.interface.inputToInternal(currentInp)

      preProcessedData = self.PreProcessor.interface.run(tempData)
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        inputDict['Features'] = copy.deepcopy(preProcessedData['data']['input'])
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        inputDict['Features'] = copy.deepcopy(preProcessedData['data']['output'])
      else:
        features = self.initializationOptionDict['KDD']['Features'].split(',')
        for param in currentInput.getParaKeys('input'):
          if param in features:
            inputDict['Features'][param] = copy.deepcopy(preProcessedData['data']['input'][param])
        for param in currentInput.getParaKeys('output'):
          if param in features:
            inputDict['Features'][param] = copy.deepcopy(preProcessedData['data']['output'][param])

      inputDict['metadata'] = currentInput.getAllMetadata()

      return inputDict

    inputDict = {'Features':{}, 'parameters':{}, 'Labels':{}, 'metadata':{}}

    if currentInput.type in ['PointSet']:
      ## Get what is available in the data object being operated on
      ## This is potentially more information than we need at the moment, but
      ## it will make the code below easier to read and highlights where objects
      ## are reused more readily
      allInputFeatures = currentInput.getParaKeys('input')
      allOutputFeatures = currentInput.getParaKeys('output')
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        for param in allInputFeatures:
          inputDict['Features'][param] = currentInput.getParam('input', param)
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        for param in allOutputFeatures:
          inputDict['Features'][param] = currentInput.getParam('output', param)
      elif self.initializationOptionDict['KDD']['Features'] == 'all':
        for param in allInputFeatures:
          inputDict['Features'][param] = currentInput.getParam('input', param)
        for param in allOutputFeatures:
          inputDict['Features'][param] = currentInput.getParam('output', param)
      else:
        ## Get what the user asks requests
        features = set(self.initializationOptionDict['KDD']['Features'].split(','))

        ## Now intersect what the user wants and what is available.
        ## NB: this will not error, if the user asks for something that does not
        ##     exist in the data, it will silently ignore it.
        inParams = list(features.intersection(allInputFeatures))
        outParams = list(features.intersection(allOutputFeatures))

        for param in inParams:
          inputDict['Features'][param] = currentInput.getParam('input', param)
        for param in outParams:
          inputDict['Features'][param] = currentInput.getParam('output', param)

    elif currentInput.type in ['HistorySet']:
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        for param in currentInput.getParaKeys('input'):
          inputDict['Features'][param] = currentInput.getParam('input', param)
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        inputDict['Features'] = currentInput.getOutParametersValues()
      elif self.initializationOptionDict['KDD']['Features'] == 'all':
        for param in allInputFeatures:
          inputDict['Features'][param] = currentInput.getParam('input', param)
        for param in allOutputFeatures:
          inputDict['Features'][param] = currentInput.getParam('output', param)
      else:
        features = set(self.initializationOptionDict['KDD']['Features'].split(','))
        allInputFeatures = currentInput.getParaKeys('input')
        allOutputFeatures = currentInput.getParaKeys('output')
        inParams = list(features.intersection(allInputFeatures))
        outParams = list(features.intersection(allOutputFeatures))
        inputDict['Features'] = {}
        for hist in currentInput._dataContainer['outputs'].keys():
          inputDict['Features'][hist] = {}
          for param in inParams:
            inputDict['Features'][hist][param] = currentInput._dataContainer['inputs'][hist][param]
          for param in outParams:
            inputDict['Features'][hist][param] = currentInput._dataContainer['outputs'][hist][param]

      inputDict['metadata'] = currentInput.getAllMetadata()

    ## Redundant if-conditional preserved as a placeholder for potential future
    ## development working directly with files
    # elif isinstance(currentInp, Files.File):
    #   self.raiseAnError(IOError, 'Unsupported input type (' + currentInput.subtype + ') for PostProcessor ' + self.name + ' must be a PointSet.')
    else:
      self.raiseAnError(IOError, 'Unsupported input type (' + currentInput.type + ') for PostProcessor ' + self.name + ' must be a PointSet.')
    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """

    PostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']

    if "SolutionExport" in initDict:
      self.solutionExport = initDict["SolutionExport"]

    if "PreProcessor" in self.assemblerDict:
      self.PreProcessor = self.assemblerDict['PreProcessor'][0][3]
      if not '_inverse' in dir(self.PreProcessor.interface):
        self.raiseAnError(IOError, 'PostProcessor ' + self.name + ' is using a pre-processor where the method inverse has not implemented')


    if 'Metric' in self.assemblerDict:
      self.metric = self.assemblerDict['Metric'][0][3]

  def _localReadMoreXML(self, xmlNode):
    """
      Function that reads the portion of the xml input that belongs to this specialized class
      and initializes some elements based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """

    # paramInput = DataMining.getInputSpecification()()
    # paramInput.parseNode(xmlNode)

    ## By default, we want to name the 'labels' by the name of this
    ## postprocessor, but that name is not available before processing the XML
    ## At this point, we have that information
    self.initializationOptionDict = {}

    for child in xmlNode:
      if child.tag == 'KDD':
        if child.attrib:
          ## I'm not sure what this thing is used for, but it seems to make more
          ## sense to only put data that is not otherwise handled rather than
          ## put all of the information and then to remove the ones we process.
          ## - dpm 6/8/16
          self.initializationOptionDict[child.tag] = {}
          for key,value in child.attrib.iteritems():
            if key == 'lib':
              self.type = value
            elif key == 'labelFeature':
              self.labelFeature = value
            else:
              self.initializationOptionDict[child.tag][key] = value
        else:
          self.initializationOptionDict[child.tag] = utils.tryParse(child.text)

        for childChild in child:
          if childChild.attrib and not childChild.tag == 'PreProcessor':
            self.initializationOptionDict[child.tag][childChild.tag] = dict(childChild.attrib)
          else:
            self.initializationOptionDict[child.tag][childChild.tag] = utils.tryParse(childChild.text)
      elif child.tag == 'pivotParameter':
        self.pivotParameter = child.text

    if not hasattr(self, 'pivotParameter'):
      #TODO, if doing time dependent data mining that needs this, an error
      # should be thrown
      self.pivotParameter = None

    if self.type:
      #TODO unSurpervisedEngine needs to be able to handle both methods
      # without this if statement.
      if self.pivotParameter is not None:
        self.unSupervisedEngine = unSupervisedLearning.returnInstance("temporalSciKitLearn", self, **self.initializationOptionDict['KDD'])
      else:
        self.unSupervisedEngine = unSupervisedLearning.returnInstance(self.type, self, **self.initializationOptionDict['KDD'])
    else:
      self.raiseAnError(IOError, 'No Data Mining Algorithm is supplied!')

    ## If the user has not defined a label feature, then we will force it to be
    ## named by the PostProcessor name followed by:
    ## the word 'Labels' for clustering/GMM models;
    ## the word 'Dimension' + a numeric id for dimensionality reduction
    ## algorithms
    if self.labelFeature is None:
      if self.unSupervisedEngine.getDataMiningType() in ['cluster','mixture']:
        self.labelFeature = self.name+'Labels'
      elif self.unSupervisedEngine.getDataMiningType() in ['decomposition','manifold']:
        self.labelFeature = self.name+'Dimension'

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler
        object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed
        results
      @ Out, None
    """
    ## When does this actually happen?
    if finishedJob.getEvaluation() == -1:
      self.raiseAnError(RuntimeError, 'No available Output to collect (Run probably is not finished yet)')
    dataMineDict = finishedJob.getEvaluation()[1]
    for key in dataMineDict['outputs']:
      for param in output.getParaKeys('output'):
        if key == param:
          output.removeOutputValue(key)
      if output.type == 'PointSet':
        for value in dataMineDict['outputs'][key]:
          output.updateOutputValue(key, copy.copy(value))
      elif output.type == 'HistorySet':
        if self.PreProcessor is not None or self.metric is not None:
          for index,value in np.ndenumerate(dataMineDict['outputs'][key]):
            firstHist = output._dataContainer['outputs'].keys()[0]
            firstVar  = output._dataContainer['outputs'][index[0]+1].keys()[0]
            timeLength = output._dataContainer['outputs'][index[0]+1][firstVar].size
            arrayBase = value * np.ones(timeLength)
            output.updateOutputValue([index[0]+1,key], arrayBase)
        else:
          tlDict = finishedJob.getEvaluation()[1]
          historyKey = output.getOutParametersValues().keys()
          for index, keyH in enumerate(historyKey):
            for keyL in tlDict['outputs'].keys():
              output.updateOutputValue([keyH,keyL], tlDict['outputs'][keyL][index,:])

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    Input = self.inputToInternal(inputIn)
    if type(inputIn) == list:
      currentInput = inputIn[-1]
    else:
      currentInput = inputIn

    if currentInput.type == 'HistorySet' and self.PreProcessor is None and self.metric is None:
      return self.__runTemporalSciKitLearn(Input)
    else:
      return self.__runSciKitLearn(Input)

  def userInteraction(self):
    """
      A placeholder for allowing user's to interact and tweak the model in-situ
      before saving the analysis results
      @ In, None
      @ Out, None
    """
    pass

  def __runSciKitLearn(self, Input):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject.  This is for SciKitLearn
      @ In, Input, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    self.unSupervisedEngine.features = Input['Features']
    if not self.unSupervisedEngine.amITrained:
      self.unSupervisedEngine.train(Input['Features'], self.metric)
    self.unSupervisedEngine.confidence()

    self.userInteraction()

    outputDict = self.unSupervisedEngine.outputDict

    if 'bicluster' == self.unSupervisedEngine.getDataMiningType():
      self.raiseAnError(RuntimeError, 'Bicluster has not yet been implemented.')

    ## Rename the algorithm output to point to the user-defined label feature
    if 'labels' in outputDict['outputs']:
      outputDict['outputs'][self.labelFeature] = outputDict['outputs'].pop('labels')
    elif 'embeddingVectors' in outputDict['outputs']:
      transformedData = outputDict['outputs'].pop('embeddingVectors')
      reducedDimensionality = transformedData.shape[1]

      for i in range(reducedDimensionality):
        newColumnName = self.labelFeature + str(i + 1)
        outputDict['outputs'][newColumnName] =  transformedData[:, i]

    if self.solutionExport is not None:
      if 'cluster' == self.unSupervisedEngine.getDataMiningType():
        solutionExportDict = self.unSupervisedEngine.metaDict
        if 'clusterCenters' in solutionExportDict:
          centers = solutionExportDict['clusterCenters']

          ## Does skl not provide a correlation between label ids and cluster centers?
          if 'clusterCentersIndices' in solutionExportDict:
            indices = solutionExportDict['clusterCentersIndices']
          else:
            indices = list(range(len(centers)))

          if self.PreProcessor is None:
            for index,center in zip(indices,centers):
              self.solutionExport.updateInputValue(self.labelFeature,index)
              ## Can I be sure of the order of dimensions in the features dict, is
              ## the same order as the data held in the UnSupervisedLearning object?
              for key,value in zip(self.unSupervisedEngine.features.keys(),center):
                self.solutionExport.updateOutputValue(key,value)
          else:
            # if a pre-processor is used it is here assumed that the pre-processor has internally a
            # method (called "inverse") which converts the cluster centers back to their original format. If this method
            # does not exist a warning will be generated
            tempDict = {}
            for index,center in zip(indices,centers):
              tempDict[index] = center
            centers = self.PreProcessor.interface._inverse(tempDict)

            for index,center in zip(indices,centers):
              self.solutionExport.updateInputValue(self.labelFeature,index)

            if self.solutionExport.type == 'HistorySet':
              for hist in centers.keys():
                for key in centers[hist].keys():
                  self.solutionExport.updateOutputValue(key,centers[hist][key])
            else:
              for key in centers.keys():
                if key in self.solutionExport.getParaKeys('outputs'):
                  for value in centers[key]:
                    self.solutionExport.updateOutputValue(key,value)
      elif 'mixture' == self.unSupervisedEngine.getDataMiningType():
        solutionExportDict = self.unSupervisedEngine.metaDict
        mixtureMeans = solutionExportDict['means']
        mixtureCovars = solutionExportDict['covars']
        ## TODO: Export Gaussian centers to SolutionExport
        ## Get the centroids and push them to a SolutionExport data object, if
        ## we have both, also if we have the centers, assume we have the indices
        ## to match them.

        ## Does skl not provide a correlation between label ids and Gaussian
        ## centers?
        indices = list(range(len(mixtureMeans)))
        for index,center in zip(indices,mixtureMeans):
          self.solutionExport.updateInputValue(self.labelFeature,index)
          ## Can I be sure of the order of dimensions in the features dict, is
          ## the same order as the data held in the UnSupervisedLearning
          ## object?
          for key,value in zip(self.unSupervisedEngine.features.keys(),center):
            self.solutionExport.updateOutputValue(key,value)
          ## You may also want to output the covariances of each pair of
          ## dimensions as well
          for i,row in enumerate(self.unSupervisedEngine.features.keys()):
            for joffset,col in enumerate(self.unSupervisedEngine.features.keys()[i:]):
              j = i+joffset
              self.solutionExport.updateOutputValue('cov_'+str(row)+'_'+str(col),mixtureCovars[index][i,j])
      elif 'decomposition' == self.unSupervisedEngine.getDataMiningType():
        solutionExportDict = self.unSupervisedEngine.metaDict

        ## Get the transformation matrix and push it to a SolutionExport
        ## data object.
        ## Can I be sure of the order of dimensions in the features dict, is
        ## the same order as the data held in the UnSupervisedLearning object?
        if 'components' in solutionExportDict:
          components = solutionExportDict['components']
          for row,values in enumerate(components):
            self.solutionExport.updateInputValue(self.labelFeature, row+1)
            for col,value in zip(self.unSupervisedEngine.features.keys(),values):
              self.solutionExport.updateOutputValue(col,value)

            if 'explainedVarianceRatio' in solutionExportDict:
              self.solutionExport.updateOutputValue('ExplainedVarianceRatio',solutionExportDict['explainedVarianceRatio'][row])
    return outputDict

  def __runTemporalSciKitLearn(self, Input):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject.  This is for temporalSciKitLearn
      @ In, Input, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    self.unSupervisedEngine.features = Input['Features']
    self.unSupervisedEngine.pivotVariable = self.pivotVariable

    if not self.unSupervisedEngine.amITrained:
      self.unSupervisedEngine.train(Input['Features'])
    self.unSupervisedEngine.confidence()

    self.userInteraction()

    outputDict = self.unSupervisedEngine.outputDict

    numberOfHistoryStep = self.unSupervisedEngine.numberOfHistoryStep
    numberOfSample = self.unSupervisedEngine.numberOfSample

    if 'bicluster' == self.unSupervisedEngine.getDataMiningType():
      self.raiseAnError(RuntimeError, 'Bicluster has not yet been implemented.')

    ## Rename the algorithm output to point to the user-defined label feature
    # if 'labels' in outputDict:
    #   outputDict['outputs'][self.labelFeature] = outputDict['outputs'].pop('labels')
    # elif 'embeddingVectors' in outputDict['outputs']:
    #   transformedData = outputDict['outputs'].pop('embeddingVectors')
    #   reducedDimensionality = transformedData.shape[1]

    #   for i in range(reducedDimensionality):
    #     newColumnName = self.labelFeature + str(i + 1)
    #     outputDict['outputs'][newColumnName] =  transformedData[:, i]

    if 'labels' in self.unSupervisedEngine.outputDict['outputs'].keys():
      labels = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
      for t in range(numberOfHistoryStep):
        labels[:,t] = self.unSupervisedEngine.outputDict['outputs']['labels'][t]
      outputDict['outputs'][self.labelFeature] = labels
    elif 'embeddingVectors' in outputDict['outputs']:
      transformedData = outputDict['outputs'].pop('embeddingVectors')
      reducedDimensionality = transformedData.values()[0].shape[1]

      for i in range(reducedDimensionality):
        dimensionI = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
        newColumnName = self.labelFeature + str(i + 1)

        for t in range(numberOfHistoryStep):
          dimensionI[:, t] =  transformedData[t][:, i]
        outputDict['outputs'][newColumnName] = dimensionI

    if 'cluster' == self.unSupervisedEngine.getDataMiningType():
      ## SKL will always enumerate cluster centers starting from zero, if this
      ## is violated, then the indexing below will break.
      if 'clusterCentersIndices' in self.unSupervisedEngine.metaDict.keys():
        clusterCentersIndices = self.unSupervisedEngine.metaDict['clusterCentersIndices']

      if 'clusterCenters' in self.unSupervisedEngine.metaDict.keys():
        clusterCenters = self.unSupervisedEngine.metaDict['clusterCenters']
        # Output cluster centroid to solutionExport
        if self.solutionExport is not None:
          ## We will process each cluster in turn
          for clusterIdx in xrange(int(np.max(labels))+1):
            ## First store the label as the input for this cluster
            self.solutionExport.updateInputValue(self.labelFeature,clusterIdx)

            ## The time series will be the first output
            ## TODO: Ensure user requests this
            self.solutionExport.updateOutputValue(self.pivotParameter, self.pivotVariable)

            ## Now we will process each feature available
            ## TODO: Ensure user requests each of these
            for featureIdx, feat in enumerate(self.unSupervisedEngine.features):
              ## We will go through the time series and find every instance
              ## where this cluster exists, if it does not, then we put a NaN
              ## to signal that the information is missing for this timestep
              timeSeries = np.zeros(numberOfHistoryStep)

              for timeIdx in range(numberOfHistoryStep):
                ## Here we use the assumption that SKL provides clusters that
                ## are integer values beginning at zero, which make for nice
                ## indexes with no need to add another layer of obfuscation
                if clusterIdx in clusterCentersIndices[timeIdx]:
                  loc = clusterCentersIndices[timeIdx].index(clusterIdx)
                  timeSeries[timeIdx] = self.unSupervisedEngine.metaDict['clusterCenters'][timeIdx][loc,featureIdx]
                else:
                  timeSeries[timeIdx] = np.nan

              ## In summary, for each feature, we fill a temporary array and
              ## stuff it into the solutionExport, one question is how do we
              ## tell it which item we are exporting? I am assuming that if
              ## I add an input, then I need to do the corresponding
              ## updateOutputValue to associate everything with it? Once I
              ## call updateInputValue again, it will move the pointer? This
              ## needs verified
              self.solutionExport.updateOutputValue(feat, timeSeries)

      if 'inertia' in self.unSupervisedEngine.outputDict.keys():
        inertia = self.unSupervisedEngine.outputDict['inertia']

    elif 'mixture' == self.unSupervisedEngine.getDataMiningType():
      if 'covars' in self.unSupervisedEngine.metaDict.keys():
        mixtureCovars = self.unSupervisedEngine.metaDict['covars']
      else:
        mixtureCovars = None

      if 'precs' in self.unSupervisedEngine.metaDict.keys():
        mixturePrecs = self.unSupervisedEngine.metaDict['precs']
      else:
        mixturePrecs = None

      if 'componentMeanIndices' in self.unSupervisedEngine.metaDict.keys():
        componentMeanIndices = self.unSupervisedEngine.metaDict['componentMeanIndices']
      else:
        componentMeanIndices = None

      if 'means' in self.unSupervisedEngine.metaDict.keys():
        mixtureMeans = self.unSupervisedEngine.metaDict['means']
      else:
        mixtureMeans = None

      # Output cluster centroid to solutionExport
      if self.solutionExport is not None:
        ## We will process each cluster in turn
        for clusterIdx in xrange(int(np.max(componentMeanIndices.values()))+1):
          ## First store the label as the input for this cluster
          self.solutionExport.updateInputValue(self.labelFeature,clusterIdx)

          ## The time series will be the first output
          ## TODO: Ensure user requests this
          self.solutionExport.updateOutputValue(self.pivotParameter, self.pivotVariable)

          ## Now we will process each feature available
          ## TODO: Ensure user requests each of these
          if mixtureMeans is not None:
            for featureIdx, feat in enumerate(self.unSupervisedEngine.features):
              ## We will go through the time series and find every instance
              ## where this cluster exists, if it does not, then we put a NaN
              ## to signal that the information is missing for this timestep
              timeSeries = np.zeros(numberOfHistoryStep)

              for timeIdx in range(numberOfHistoryStep):
                loc = componentMeanIndices[timeIdx].index(clusterIdx)
                timeSeries[timeIdx] = mixtureMeans[timeIdx][loc,featureIdx]

              ## In summary, for each feature, we fill a temporary array and
              ## stuff it into the solutionExport, one question is how do we
              ## tell it which item we are exporting? I am assuming that if
              ## I add an input, then I need to do the corresponding
              ## updateOutputValue to associate everything with it? Once I
              ## call updateInputValue again, it will move the pointer? This
              ## needs verified
              self.solutionExport.updateOutputValue(feat, timeSeries)

          ## You may also want to output the covariances of each pair of
          ## dimensions as well
          if mixtureCovars is not None:
            for i,row in enumerate(self.unSupervisedEngine.features.keys()):
              for joffset,col in enumerate(self.unSupervisedEngine.features.keys()[i:]):
                j = i+joffset
                timeSeries = np.zeros(numberOfHistoryStep)
                for timeIdx in range(numberOfHistoryStep):
                  loc = componentMeanIndices[timeIdx].index(clusterIdx)
                  timeSeries[timeIdx] = mixtureCovars[timeIdx][loc][i,j]
                self.solutionExport.updateOutputValue('cov_'+str(row)+'_'+str(col),timeSeries)
    elif 'decomposition' == self.unSupervisedEngine.getDataMiningType():
      if self.solutionExport is not None:
        solutionExportDict = self.unSupervisedEngine.metaDict
        ## Get the transformation matrix and push it to a SolutionExport
        ## data object.
        ## Can I be sure of the order of dimensions in the features dict, is
        ## the same order as the data held in the UnSupervisedLearning object?
        if 'components' in solutionExportDict:
          components = solutionExportDict['components']

          ## Note, this implies some data exists (Really this information should
          ## be stored in a dictionary to begin with)
          numComponents,numDimensions = components[0].shape

          componentsArray = np.zeros((numberOfHistoryStep,numComponents, numDimensions))
          evrArray = np.zeros((numberOfHistoryStep,numComponents))

          for timeIdx in range(numberOfHistoryStep):
            for componentIdx,values in enumerate(components[timeIdx]):
              componentsArray[timeIdx,componentIdx,:] = values
              evrArray[timeIdx,componentIdx] = solutionExportDict['explainedVarianceRatio'][timeIdx][componentIdx]

          for componentIdx in range(numComponents):
            ## First store the dimension name as the input for this component
            self.solutionExport.updateInputValue(self.labelFeature, componentIdx+1)

            ## The time series will be the first output
            ## TODO: Ensure user requests this
            self.solutionExport.updateOutputValue(self.pivotParameter, self.pivotVariable)

            ## Now we will process each feature available
            ## TODO: Ensure user requests each of these
            for dimIdx,dimName in enumerate(self.unSupervisedEngine.features.keys()):
              values = componentsArray[:,componentIdx,dimIdx]
              self.solutionExport.updateOutputValue(dimName,values)

            if 'explainedVarianceRatio' in solutionExportDict:
              self.solutionExport.updateOutputValue('ExplainedVarianceRatio',evrArray[:,componentIdx])

    return outputDict

try:
  import qtpy.QtCore as qtc

  class QDataMining(DataMining,qtc.QObject):
    """
      DataMining class - Computes a hierarchical clustering from an input point
      cloud consisting of an arbitrary number of input parameters
    """
    requestUI = qtc.Signal(str,str,dict)
    def __init__(self, messageHandler):
      """
       Constructor
       @ In, messageHandler, message handler object
       @ Out, None
      """
      DataMining.__init__(self, messageHandler)
      qtc.QObject.__init__(self)
      self.interactive = False

    def _localReadMoreXML(self, xmlNode):
      """
        Function to grab the names of the methods this post-processor will be
        using
        @ In, xmlNode    : Xml element node
        @ Out, None
      """
      DataMining._localReadMoreXML(self, xmlNode)
      for child in xmlNode:
        for grandchild in child:
          if grandchild.tag == 'interactive':
            self.interactive = True

    def _localWhatDoINeed(self):
      """
        This method is a local mirror of the general whatDoINeed method.
        It is implemented by the samplers that need to request special objects
        @ In , None, None
        @ Out, needDict, list of objects needed
      """
      needDict = DataMining._localWhatDoINeed(self)
      needDict['internal'].append((None,'app'))
      return needDict

    def _localGenerateAssembler(self,initDict):
      """
        Generates the assembler.
        @ In, initDict, dict of init objects
        @ Out, None
      """
      DataMining._localGenerateAssembler(self, initDict)
      self.app = initDict['internal']['app']
      if self.app is None:
        self.interactive = False

    def userInteraction(self):
      """
        Launches an interface allowing the user to tweak specific model
        parameters before saving the results to the output object(s).
        @ In, None
        @ Out, None
      """

      ## If it has not been requested, then we are not waiting for a UI,
      ## otherwise the UI has been requested, and we are going to need to wait
      ## for it.
      self.uiDone = not self.interactive

      if self.interactive:

        ## Connect our own signal to the slot on the main thread
        self.requestUI.connect(self.app.createUI)

        ## Connect our own slot to listen for whenver the main thread signals a
        ## window has been closed
        self.app.windowClosed.connect(self.signalDone)

        ## Give this UI a unique id in case other threads are requesting UI
        ##  elements
        uiID = unicode(id(self))

        ## Send the request for a UI thread to the main application
        self.requestUI.emit('HierarchyWindow', uiID,
                            {'views': ['DendrogramView','ScatterView'],
                             'engine': self.unSupervisedEngine})

        ## Spinlock will wait until this instance's window has been closed
        while(not self.uiDone):
          time.sleep(1)

        ## First check that the requested UI exists, and then if that UI has the
        ## requested information, if not proceed as if it were not an
        ## interactive session.
        if uiID in self.app.UIs and hasattr(self.app.UIs[uiID],'level') and self.app.UIs[uiID].level is not None:
          self.initializationOptionDict['KDD']['level'] = self.app.UIs[uiID].level

    def signalDone(self,uiID):
      """
        In Qt language, this is a slot that will accept a signal from the UI
        saying that it has completed, thus allowing the computation to begin
        again with information updated by the user in the UI.
        @In, uiID, string, the ID of the user interface that signaled its
            completion. Thus, if several UI windows are open, we don't proceed,
            until the correct one has signaled it is done.
        @Out, None
      """
      if uiID == unicode(id(self)):
        self.uiDone = True
except ImportError as e:
  pass
