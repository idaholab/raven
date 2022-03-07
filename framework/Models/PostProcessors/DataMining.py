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
#External Modules---------------------------------------------------------------
import numpy as np
import copy
import time
import xarray as xr
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import utils, mathUtils
from ...utils import InputData, InputTypes
from ... import Files
from ... import unSupervisedLearning
from ... import MetricDistributor
#Internal Modules End-----------------------------------------------------------

class DataMining(PostProcessorInterface):
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
    inputSpecification = super(DataMining, cls).getInputSpecification()

    preProcessorInput = InputData.parameterInputFactory("PreProcessor", contentType=InputTypes.StringType)
    preProcessorInput.addParam("class", InputTypes.StringType)
    preProcessorInput.addParam("type", InputTypes.StringType)

    pivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)

    inputSpecification.addSub(pivotParameterInput)

    dataObjectInput = InputData.parameterInputFactory("DataObject", contentType=InputTypes.StringType)
    dataObjectInput.addParam("class", InputTypes.StringType)
    dataObjectInput.addParam("type", InputTypes.StringType)

    inputSpecification.addSub(dataObjectInput)

    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    metricInput.addParam("class", InputTypes.StringType)
    metricInput.addParam("type", InputTypes.StringType)

    inputSpecification.addSub(metricInput)

    kddInput = InputData.parameterInputFactory("KDD")
    kddInput.addParam("lib", InputTypes.StringType)
    kddInput.addParam("labelFeature", InputTypes.StringType)


    sklTypeInput = InputData.parameterInputFactory("SKLtype", contentType=InputTypes.StringType)
    kddInput.addSub(sklTypeInput)
    sciPyTypeInput = InputData.parameterInputFactory("SCIPYtype", contentType=InputTypes.StringType)
    kddInput.addSub(sciPyTypeInput)

    for name, inputType in [("interactive",InputTypes.StringType),
                            ("Features",InputTypes.StringType),
                            ("n_components",InputTypes.StringType),
                            ("covariance_type",InputTypes.StringType),
                            ("random_state",InputTypes.StringType),
                            ("min_covar",InputTypes.FloatType),
                            ("thresh",InputTypes.FloatType),
                            ("n_iter",InputTypes.IntegerType),
                            ("n_init",InputTypes.IntegerType),
                            ("params",InputTypes.StringType),
                            ("init_params",InputTypes.StringType),
                            ("alpha",InputTypes.FloatType),
                            ("n_clusters",InputTypes.IntegerType),
                            ("max_iter",InputTypes.IntegerType),
                            ("init",InputTypes.StringType),
                            ("precompute_distances",InputTypes.StringType),
                            ("tol",InputTypes.FloatType),
                            ("n_jobs",InputTypes.IntegerType),
                            ("max_no_improvement",InputTypes.IntegerType),
                            ("batch_size",InputTypes.IntegerType),
                            ("compute_labels",InputTypes.StringType),
                            ("reassignment_ratio",InputTypes.FloatType),
                            ("damping",InputTypes.StringType),
                            ("convergence_iter",InputTypes.IntegerType),
                            ("copy",InputTypes.StringType),
                            ("preference",InputTypes.StringType),
                            ("affinity",InputTypes.StringType),
                            ("verbose",InputTypes.StringType),
                            ("bandwidth",InputTypes.FloatType),
                            ("seeds",InputTypes.StringType),
                            ("bin_seeding",InputTypes.StringType),
                            ("min_bin_freq",InputTypes.IntegerType),
                            ("cluster_all",InputTypes.StringType),
                            ("gamma",InputTypes.FloatType),
                            ("degree",InputTypes.StringType),
                            ("coef0",InputTypes.FloatType),
                            ("n_neighbors",InputTypes.IntegerType),
                            ("eigen_solver",InputTypes.StringType),
                            ("eigen_tol",InputTypes.FloatType),
                            ("assign_labels",InputTypes.StringType),
                            ("kernel_params",InputTypes.StringType),
                            ("eps",InputTypes.StringType),
                            ("min_samples",InputTypes.IntegerType),
                            ("metric", InputTypes.StringType),
                            ("connectivity",InputTypes.StringType),
                            ("linkage",InputTypes.StringType),
                            ("whiten",InputTypes.StringType),
                            ("iterated_power",InputTypes.StringType),
                            ("kernel",InputTypes.StringType),
                            ("fit_inverse_transform",InputTypes.StringType),
                            ("remove_zero_eig",InputTypes.StringType),
                            ("ridge_alpha",InputTypes.FloatType),
                            ("method",InputTypes.StringType),
                            ("U_init",InputTypes.StringType),
                            ("V_init",InputTypes.StringType),
                            ("callback",InputTypes.StringType),
                            ("shuffle",InputTypes.StringType),
                            ("algorithm",InputTypes.StringType),
                            ("fun",InputTypes.StringType),
                            ("fun_args",InputTypes.StringType),
                            ("w_init",InputTypes.StringType),
                            ("path_method",InputTypes.StringType),
                            ("neighbors_algorithm",InputTypes.StringType),
                            ("reg",InputTypes.FloatType),
                            ("hessian_tol",InputTypes.FloatType),
                            ("modified_tol",InputTypes.FloatType),
                            ("dissimilarity",InputTypes.StringType),
                            ("level",InputTypes.StringType),
                            ("criterion",InputTypes.StringType),
                            ("dendrogram",InputTypes.StringType),
                            ("truncationMode",InputTypes.StringType),
                            ("p",InputTypes.IntegerType),
                            ("leafCounts",InputTypes.StringType),
                            ("showContracted",InputTypes.StringType),
                            ("annotatedAbove",InputTypes.FloatType),
                            ("dendFileID",InputTypes.StringType)]:
      dataType = InputData.parameterInputFactory(name, contentType=inputType)
      kddInput.addSub(dataType)

    inputSpecification.addSub(kddInput)

    inputSpecification.addSub(preProcessorInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR DATAMINING'

    self.addAssemblerObject('PreProcessor', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('Metric', InputData.Quantity.zero_to_one)

    self.solutionExport = None  ## A data object to hold derived info about the algorithm being performed,
                                ## e.g., cluster centers or a projection matrix for dimensionality reduction methods

    self.labelFeature = None    ## User customizable column name for the labels associated to a clustering or
                                ## a DR algorithm

    self.PreProcessor = None    ## Instance of PreProcessor, default is None
    self.metric = None          ## Instance of Metric, default is None
    self.pivotParameter = None  ## default pivotParameter for HistorySet
    self._type = None           ## the type of library that are used for data mining, i.e. SciKitLearn

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

  def inputToInternalForHistorySet(self,currentInput):
    """
      Function to convert the input history set into a format that this
      post-processor can understand
      @ In, currentInput, object, DataObject of currentInput
      @ Out, inputDict, dict, an input dictionary that this post-processor can process
    """
    dataSet = currentInput.asDataset()
    inputDict = {'Features': {}, 'parameters': {}, 'Labels': {}, 'metadata': {}}
    if self.pivotParameter is None:
      self.pivotParameter = currentInput.indexes[-1]
    if self.PreProcessor is None and self.metric is None:
      if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
        self.raiseAnError(IOError, "The data provided by the DataObject ", currentInput.name, " is not synchronized!")
      # for testing time dependent data mining - time dependent clustering
      self.pivotVariable = np.asarray([dataSet.isel(**{currentInput.sampleTag:i}).dropna(self.pivotParameter)[self.pivotParameter].values for i in range(len(currentInput))])
      historyKey          = dataSet[self.pivotParameter].values
      numberOfSample      = len(dataSet['RAVEN_sample_ID'].values)
      numberOfHistoryStep = len(dataSet[self.pivotParameter].values)

      if self.initializationOptionDict['KDD']['Features'] == 'input':
        self.raiseAnError(ValueError, 'To perform data mining over input please use SciKitLearn library')
      elif self.initializationOptionDict['KDD']['Features'] in ['output', 'all']:
        features = currentInput.getVars('output')
      else:
        features = [elem.strip() for elem in self.initializationOptionDict['KDD']['Features'].split(',')]

      for param in features:
        inputDict['Features'][param] = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
        # FIXME: Slow loop in case of many samples, improve performance
        for cnt in range(numberOfSample):
          inputDict['Features'][param][cnt,:] = currentInput.realization(index=cnt)[param]

    elif self.metric is not None:
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        self.raiseAnError(ValueError, 'KDD Post-processor for time dependent data with metric provided allows only output variables (time-dependent)')
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        numberOfSample = currentInput.size
        self.pivotVariable = np.asarray([dataSet.isel(**{currentInput.sampleTag:i}).dropna(self.pivotParameter)[self.pivotParameter].values for i in range(len(currentInput))])
        for i in range(numberOfSample):
          rlz = currentInput.realization(index=i)
          inputDict['Features'][i] = {}
          for var in currentInput.getVars('output'):
            inputDict['Features'][i][var] = rlz[var]
          inputDict['Features'][i][self.pivotParameter] = self.pivotVariable[i]

    elif self.PreProcessor is not None:
      self.pivotParameter = currentInput.indexes[-1]
      self.pivotVariable = np.asarray([dataSet.isel(RAVEN_sample_ID=i).dropna(self.pivotParameter)[self.pivotParameter].values for i in range(len(currentInput))])
      return self.inputToInternalForPreProcessor(currentInput)

    inputDict['metadata'] = currentInput.getMeta(pointwise=True,general=True)
    return inputDict

  def inputToInternalForPointSet(self,currentInput):
    """
      Function to convert the input point set into a format that this
      post-processor can understand
      @ In, currentInput, object, DataObject of currentInput
      @ Out, inputDict, dict, an input dictionary that this post-processor can process
    """
    ## Get what is available in the data object being operated on
    ## This is potentially more information than we need at the moment, but
    ## it will make the code below easier to read and highlights where objects
    ## are reused more readily

    data = currentInput.asDataset()
    allInputFeatures = currentInput.getVars('input')
    allOutputFeatures = currentInput.getVars('output')

    if self.PreProcessor is None:
      inputDict = {'Features': {}, 'parameters': {}, 'Labels': {}, 'metadata': {}}
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        for param in allInputFeatures:
          inputDict['Features'][param] = data[param].values
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        for param in allOutputFeatures:
          inputDict['Features'][param] = data[param].values
      elif self.initializationOptionDict['KDD']['Features'] == 'all':
        for param in allInputFeatures:
          inputDict['Features'][param] = data[param].values
        for param in allOutputFeatures:
          inputDict['Features'][param] = data[param].values
      else:
        ## Get what the user asks requests
        features = set(self.initializationOptionDict['KDD']['Features'].split(','))
        allFeatures = set(allInputFeatures + allOutputFeatures)
        if not features.issubset(allFeatures):
          self.raiseAnError(ValueError, 'Data Mining PP: features specified in the '
                                        'PP (' + str(features) + ') do not match the one available '
                                        'in the dataObject ('+ str(allInputFeatures+allOutputFeatures) +') ')
        ## Now intersect what the user wants and what is available.
        ## NB: this will not error, if the user asks for something that does not
        ##     exist in the data, it will silently ignore it.
        inParams  = list(features.intersection(allInputFeatures))
        outParams = list(features.intersection(allOutputFeatures))

        for param in inParams:
          inputDict['Features'][param] = data[param].values
        for param in outParams:
          inputDict['Features'][param] = data[param].values

      inputDict['metadata'] = currentInput.getMeta(pointwise=True,general=True)
      return inputDict

    elif self.PreProcessor is not None:
      return self.inputToInternalForPreProcessor(currentInput)

  def inputToInternalForPreProcessor(self,currentInput):
    """
      Function to convert the received input into a format that this
      post-processor can understand
      @ In, currentInput, object, DataObject of currentInput
      @ Out, inputDict, dict, an input dictionary that this post-processor can process
    """
    inputDict = {'Features': {}, 'parameters': {}, 'Labels': {}, 'metadata': {}}
    if not set(self.PreProcessor._pp.validDataType).issubset(set(['PointSet'])):
      self.raiseAnError(IOError, 'DataMining PP: this PP is employing a pre-processor PP which does not generates a PointSet.')

    tempData = self.PreProcessor._pp.createPostProcessorInput([currentInput])
    preProcessedData = self.PreProcessor._pp.run(tempData)

    if self.initializationOptionDict['KDD']['Features'] == 'input':
      featureList = currentInput.getVars('input')
    elif self.initializationOptionDict['KDD']['Features'] == 'output':
      dataList = preProcessedData['data'].keys()
      # FIXME: this fix is due to the changes in the data structure of © pp
      toRemove = currentInput.getVars('input') + currentInput.getVars('meta')
      featureList = [elem for elem in dataList if elem not in toRemove]
    else:
      featureList = [feature.strip() for feature in self.initializationOptionDict['KDD']['Features'].split(',')]
    for key in featureList:
      inputDict['Features'][key] = copy.deepcopy(preProcessedData['data'][key])

    inputDict['metadata'] = currentInput.getMeta(pointwise=True,general=True)
    return inputDict

  def inputToInternal(self, currentInp):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, currentInp, list or DataObjects, Some form of data object or list of
        data objects handed to the post-processor
      @ Out, inputDict, dict, An input dictionary this object can process
    """

    if type(currentInp) == list:
      if len(currentInp) > 1:
        self.raiseAnError(IOError, "Only one input is allowed for this post-processor: ", self.name)
      currentInput = currentInp[-1]
    else:
      currentInput = currentInp

    if hasattr(currentInput, 'type'):
      if currentInput.type == 'HistorySet':
        return self.inputToInternalForHistorySet(currentInput)

      elif currentInput.type == 'PointSet':
        return self.inputToInternalForPointSet(currentInput)

    elif type(currentInp) == dict:
      if 'Features' in currentInput.keys():
        return currentInput

    elif isinstance(currentInp, Files.File):
      self.raiseAnError(IOError, 'DataMining PP: this PP does not support files as input.')

    elif currentInput.type == 'HDF5':
      self.raiseAnError(IOError, 'DataMining PP: this PP does not support HDF5 Objects as input.')

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if "SolutionExport" in initDict:
      self.solutionExport = initDict["SolutionExport"]
    if "PreProcessor" in self.assemblerDict:
      self.PreProcessor = self.assemblerDict['PreProcessor'][0][3]
      if not '_inverse' in dir(self.PreProcessor._pp):
        self.raiseAnError(IOError, 'PostProcessor ' + self.name + ' is using a pre-processor where the method inverse has not implemented')
    if 'Metric' in self.assemblerDict:
      self.metric = self.assemblerDict['Metric'][0][3]

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    ## By default, we want to name the 'labels' by the name of this
    ## postprocessor, but that name is not available before processing the XML
    ## At this point, we have that information
    self.initializationOptionDict = {}
    for child in paramInput.subparts:
      if child.getName() == 'KDD':
        if len(child.parameterValues) > 0:
          ## I'm not sure what this thing is used for, but it seems to make more
          ## sense to only put data that is not otherwise handled rather than
          ## put all of the information and then to remove the ones we process.
          ## - dpm 6/8/16
          self.initializationOptionDict[child.getName()] = {}
          for key,value in child.parameterValues.items():
            if key == 'lib':
              self._type = value
            elif key == 'labelFeature':
              self.labelFeature = value
            else:
              self.initializationOptionDict[child.getName()][key] = value
        else:
          self.initializationOptionDict[child.getName()] = utils.tryParse(child.value)
        for childChild in child.subparts:
          if len(childChild.parameterValues) > 0 and not childChild.getName() == 'PreProcessor':
            self.initializationOptionDict[child.getName()][childChild.getName()] = dict(childChild.parameterValues)
          else:
            self.initializationOptionDict[child.getName()][childChild.getName()] = utils.tryParse(childChild.value)
      elif child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
    if not hasattr(self, 'pivotParameter'):
      #TODO, if doing time dependent data mining that needs this, an error
      # should be thrown
      self.pivotParameter = None
    if self._type:
      #TODO unSurpervisedEngine needs to be able to handle both methods
      # without this if statement.
      if self.pivotParameter is not None:
        self.unSupervisedEngine = unSupervisedLearning.factory.returnInstance("temporalSciKitLearn", **self.initializationOptionDict['KDD'])
      else:
        self.unSupervisedEngine = unSupervisedLearning.factory.returnInstance(self._type, **self.initializationOptionDict['KDD'])
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

  def collectOutput(self, finishedJob, outputObject):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler
        object that is in charge of running this post-processor
      @ InOut, outputObject, dataObjects, A reference to an object where we want
        to place our computed results
    """
    if len(outputObject) !=0:
      self.raiseAnError(IOError,"There is some information already stored in the DataObject",outputObject.name, \
              "the calculations from PostProcessor",self.name, " can not be stored in this DataObject!", \
              "Please provide a new empty DataObject for this PostProcessor!")
    ## When does this actually happen?
    evaluation = finishedJob.getEvaluation()
    inputObject, dataMineDict = evaluation
    ## This should not have to be a list
    ## TODO: figure out if there is a case where it can be in this processor
    inputObject = inputObject[0]
    ## Store everything we cannot wrangle from the input data object and the
    ## result of the dataMineDict
    ## We will explicitly copy everything from the input data object to the output data object.
    ############################################################################
    if inputObject.type != outputObject.type:
      self.raiseAnError(IOError,"The type of output DataObject",outputObject.name,"is not consistent with input",\
              "DataObject type, i.e. ",outputObject.type,"!=",inputObject.type)
    rlzs = {}
    # first create a new dataset from copying input data object
    dataset = inputObject.asDataset().copy(deep=True)
    sampleTag = inputObject.sampleTag
    sampleCoord = dataset[sampleTag].values
    availVars = dataset.data_vars.keys()
    # update variable values if the values in the dataset are different from the values in the dataMineDict
    # dataMineDict stores all the information generated by the datamining algorithm
    if outputObject.type == 'PointSet':
      for key,value in dataMineDict['outputs'].items():
        if key in availVars and not np.array_equal(value,dataset[key].values):
          newDA = xr.DataArray(value,dims=(sampleTag),coords={sampleTag:sampleCoord})
          dataset = dataset.drop(key)
          dataset[key] = newDA
        elif key not in availVars:
          newDA = xr.DataArray(value,dims=(sampleTag),coords={sampleTag:sampleCoord})
          dataset[key] = newDA
    elif outputObject.type == 'HistorySet':
      for key,values in dataMineDict['outputs'].items():
        if key not in availVars:
          expDict = {}
          for index, value in enumerate(values):
            timeLength = len(self.pivotVariable[index])
            arrayBase = value * np.ones(timeLength)
            xrArray = xr.DataArray(arrayBase,dims=(self.pivotParameter), coords=[self.pivotVariable[index]])
            expDict[sampleCoord[index]] = xrArray
          ds = xr.Dataset(data_vars=expDict)
          ds = ds.to_array().rename({'variable':sampleTag})
          dataset[key] = ds
    else:
      self.raiseAnError(IOError, 'Unrecognized type for output data object ', outputObject.name, \
              '! Available type are HistorySet or PointSet!')

    outputObject.load(dataset,style='dataset')

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
    if hasattr(currentInput, 'type') and currentInput.type == 'HistorySet' and self.PreProcessor is None and self.metric is None:
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

  def updateFeatures(self, features):
    """
      Change the Features that this classifier targets.
      @ In, features, list(str), list of new features
      @ Out, None
    """
    self.unSupervisedEngine.updateFeatures(features)

  def __adjustFeatures(self, features):
    """
      If the features are the output, then they need to be listed
      @ In, features, dict, dictionary of the features
      @ Out, None
    """
    if self.unSupervisedEngine.features == ['output']:
      self.unSupervisedEngine.features = sorted(features)
    assert set(self.unSupervisedEngine.features) == set(features)

  def __regulateLabels(self, originalLabels):
    """
      Regulates the labels such that the first one to appear is 0, second one is 1, and so on.
      @ In, originalLabels, list(int), the original labeling system
      @ Out, labels, list(int), fixed up labels
    """
    # this functionality relocated to serve more entities
    return mathUtils.orderClusterLabels(originalLabels)

  def __runSciKitLearn(self, Input):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject.  This is for SciKitLearn
      @ In, Input, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    self.__adjustFeatures(Input['Features'])
    if not self.unSupervisedEngine.amITrained:
      metric = None
      if self.metric is not None:
        metric = MetricDistributor.factory.returnInstance('MetricDistributor', self.metric)
      self.unSupervisedEngine.train(Input['Features'], metric)
    self.unSupervisedEngine.confidence()
    self.userInteraction()
    outputDict = self.unSupervisedEngine.outputDict
    if 'bicluster' == self.unSupervisedEngine.getDataMiningType():
      self.raiseAnError(RuntimeError, 'Bicluster has not yet been implemented.')
    ## Rename the algorithm output to point to the user-defined label feature
    if 'labels' in outputDict['outputs']:
      labels = self.__regulateLabels(outputDict['outputs'].pop('labels'))
      outputDict['outputs'][self.labelFeature] = labels #outputDict['outputs'].pop('labels')
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
          rlzs = {}
          if self.PreProcessor is None:
            rlzs[self.labelFeature] = np.atleast_1d(indices)
            for i, key in enumerate(self.unSupervisedEngine.features):
              ## FIXME: Can I be sure of the order of dimensions in the features dict, is
              ## the same order as the data held in the UnSupervisedLearning object?
              rlzs[key] = np.atleast_1d(centers[:,i])
            self.solutionExport.load(rlzs, style='dict')
          else:
            # if a pre-processor is used it is here assumed that the pre-processor has internally a
            # method (called "inverse") which converts the cluster centers back to their original format. If this method
            # does not exist a warning will be generated
            tempDict = {}
            rlzs = {}
            rlzDims = {}
            for index,center in zip(indices,centers):
              tempDict[index] = center
            centers = self.PreProcessor._pp._inverse(tempDict)
            rlzs[self.labelFeature] = np.atleast_1d(indices)
            rlzDims[self.labelFeature] = []
            if self.solutionExport.type == 'PointSet':
              for key in centers.keys():
                rlzs[key] = np.atleast_1d(centers[key])
              self.solutionExport.load(rlz, style='dict')
            else:
              for hist in centers.keys():
                for key in centers[hist].keys():
                  if key not in rlzs.keys():
                    rlzs[key] = copy.copy(centers[hist][key])
                    rlzDims[key] = [self.pivotParameter]
                  else:
                    rlzs[key] = np.vstack((rlzs[key], copy.copy(centers[hist][key])))
              self.solutionExport.load(rlzs, style='dict', dims=rlzDims)
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
        rlzs = {}
        additionalOutput = {}
        rlzs[self.labelFeature] = np.atleast_1d(indices)
        for i, key in enumerate(self.unSupervisedEngine.features):
          ## Can I be sure of the order of dimensions in the features dict, is
          ## the same order as the data held in the UnSupervisedLearning
          ## object?
          rlzs[key] = np.atleast_1d(mixtureMeans[:,i])
          ##FIXME: You may also want to output the covariances of each pair of
          ## dimensions as well, this is currently only accessible from the solution export metadata
          ## We should list the variables name the solution export in order to access this data
          for joffset,col in enumerate(list(self.unSupervisedEngine.features)[i:]):
            j = i+joffset
            covValues = mixtureCovars[:,i,j]
            covName = 'cov_'+str(key)+'_'+str(col)
            additionalOutput[covName] = np.atleast_1d(covValues)
        self.solutionExport.load(rlzs, style = 'dict')
        if additionalOutput:
          for key, value in additionalOutput.items():
            self.solutionExport.addVariable(key, value)
      elif 'decomposition' == self.unSupervisedEngine.getDataMiningType():
        solutionExportDict = self.unSupervisedEngine.metaDict
        ## Get the transformation matrix and push it to a SolutionExport
        ## data object.
        ## Can I be sure of the order of dimensions in the features dict, is
        ## the same order as the data held in the UnSupervisedLearning object?
        rlzs = {}
        if 'components' in solutionExportDict:
          components = solutionExportDict['components']
          indices = list(range(1, len(components)+1))
          rlzs[self.labelFeature] = np.atleast_1d(indices)
          for keyIndex, key in enumerate(self.unSupervisedEngine.features):
            rlzs[key] = np.atleast_1d(components[:,keyIndex])
        self.solutionExport.load(rlzs, style='dict')
        # FIXME: I think the user need to specify the following word in the solution export data object
        # in order to access this data, currently, we just added to the metadata of solution export
        if 'explainedVarianceRatio' in solutionExportDict:
          self.solutionExport.addVariable('explainedVarianceRatio', np.atleast_1d(solutionExportDict['explainedVarianceRatio']))

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

    if 'labels' in self.unSupervisedEngine.outputDict['outputs'].keys():
      labels = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
      for t in range(numberOfHistoryStep):
        labels[:,t] = self.unSupervisedEngine.outputDict['outputs']['labels'][t]
      outputDict['outputs'][self.labelFeature] = labels
    elif 'embeddingVectors' in outputDict['outputs']:
      transformedData = outputDict['outputs'].pop('embeddingVectors')
      reducedDimensionality = utils.first(transformedData.values()).shape[1]

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
          rlzDims = {}
          rlzs = {}
          clusterLabels = range(int(np.max(labels)) + 1)
          rlzs[self.labelFeature] = np.atleast_1d(clusterLabels)
          rlzs[self.pivotParameter] = self.pivotVariable
          ## We will process each cluster in turn
          for rlzIndex in clusterLabels:
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
                if rlzIndex in clusterCentersIndices[timeIdx]:
                  loc = clusterCentersIndices[timeIdx].index(rlzIndex)
                  timeSeries[timeIdx] = self.unSupervisedEngine.metaDict['clusterCenters'][timeIdx][loc,featureIdx]
                else:
                  timeSeries[timeIdx] = np.atleast_1d(np.nan)

              ## In summary, for each feature, we fill a temporary array and
              ## stuff it into the solutionExport, one question is how do we
              ## tell it which item we are exporting? I am assuming that if
              ## I add an input, then I need to do the corresponding
              ## updateOutputValue to associate everything with it? Once I
              ## call updateInputValue again, it will move the pointer? This
              ## needs verified
              if feat not in rlzs.keys():
                rlzs[feat] = np.zeros((len(clusterLabels), numberOfHistoryStep))
                rlzs[feat][rlzIndex] = copy.copy(timeSeries)
                rlzDims[feat] = [self.pivotParameter]
              else:
                rlzs[feat][rlzIndex] = copy.copy(timeSeries)
          self.solutionExport.load(rlzs, style='dict',dims=rlzDims)

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
        rlzDims = {}
        rlzs = {}
        ## First store the label as the input for this cluster
        mixLabels = range(int(np.max(list(componentMeanIndices.values())))+1)
        rlzs[self.labelFeature] = np.atleast_1d(mixLabels)
        rlzs[self.pivotParameter] = self.pivotVariable
        for rlzIndex in mixLabels:
          ## Now we will process each feature available
          ## TODO: Ensure user requests each of these
          if mixtureMeans is not None:
            for featureIdx, feat in enumerate(self.unSupervisedEngine.features):
              ## We will go through the time series and find every instance
              ## where this cluster exists, if it does not, then we put a NaN
              ## to signal that the information is missing for this timestep
              timeSeries = np.zeros(numberOfHistoryStep)
              for timeIdx in range(numberOfHistoryStep):
                loc = componentMeanIndices[timeIdx].index(rlzIndex)
                timeSeries[timeIdx] = mixtureMeans[timeIdx][loc,featureIdx]
              ## In summary, for each feature, we fill a temporary array and
              ## stuff it into the solutionExport, one question is how do we
              ## tell it which item we are exporting? I am assuming that if
              ## I add an input, then I need to do the corresponding
              ## updateOutputValue to associate everything with it? Once I
              ## call updateInputValue again, it will move the pointer? This
              ## needs verified
              if feat not in rlzs.keys():
                rlzs[feat] = copy.copy(timeSeries)
                rlzDims[feat] = [self.pivotParameter]
              else:
                rlzs[feat] = np.vstack((rlzs[feat], copy.copy(timeSeries)))
          ## You may also want to output the covariances of each pair of
          ## dimensions as well
          if mixtureCovars is not None:
            for i,row in enumerate(self.unSupervisedEngine.features.keys()):
              for joffset,col in enumerate(list(self.unSupervisedEngine.features.keys())[i:]):
                j = i+joffset
                timeSeries = np.zeros(numberOfHistoryStep)
                for timeIdx in range(numberOfHistoryStep):
                  loc = componentMeanIndices[timeIdx].index(rlzIndex)
                  timeSeries[timeIdx] = mixtureCovars[timeIdx][loc][i,j]
                covPairName = 'cov_' + str(row) + '_' + str(col)
                if covPairName not in rlzs.keys():
                  rlzs[covPairName] = timeSeries
                  rlzDims[covPairName] = [self.pivotParameter]
                else:
                  rlzs[covPairName] = np.vstack((rlzs[covPairName], timeSeries))
        self.solutionExport.load(rlzs, style='dict',dims=rlzDims)
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

          componentsArray = np.zeros((numComponents, numberOfHistoryStep, numDimensions))
          evrArray = np.zeros((numComponents, numberOfHistoryStep))

          for timeIdx in range(numberOfHistoryStep):
            for componentIdx,values in enumerate(components[timeIdx]):
              componentsArray[componentIdx,timeIdx,:] = values
              evrArray[componentIdx, timeIdx] = solutionExportDict['explainedVarianceRatio'][timeIdx][componentIdx]

          rlzs = {}
          rlzDims = {}
          ## First store the dimension name as the input for this component
          rlzs[self.labelFeature] =  np.atleast_1d(range(1,numComponents+1))
          rlzs[self.pivotParameter] = self.pivotVariable
          for dimIdx,dimName in enumerate(self.unSupervisedEngine.features.keys()):
            values = componentsArray[:,:,dimIdx]
            rlzs[dimName] = values
            rlzDims[dimName] = [self.pivotParameter]
            if 'explainedVarianceRatio' in solutionExportDict:
              rlzs['ExplainedVarianceRatio'] = evrArray
              rlzDims['ExplainedVarianceRatio'] = [self.pivotParameter]
        self.solutionExport.load(rlzs, style='dict',dims=rlzDims)

    return outputDict

try:
  import PySide.QtCore as qtc
  __QtAvailable = True
except ImportError as e:
  try:
    import PySide2.QtCore as qtc
    __QtAvailable = True
  except ImportError as e:
    __QtAvailable = False

if __QtAvailable:
  class mQDataMining(type(DataMining), type(qtc.QObject)):
    """
      Class used to solve the metaclass conflict
    """
    pass

  class QDataMining(DataMining, qtc.QObject, metaclass=mQDataMining):
    """
      DataMining class - Computes a hierarchical clustering from an input point
      cloud consisting of an arbitrary number of input parameters
    """
    requestUI = qtc.Signal(str,str,dict)
    @classmethod
    def getInputSpecification(cls):
      """
        Method to get a reference to a class that specifies the input data for
        class cls.
        @ In, cls, the class for which we are retrieving the specification
        @ Out, inputSpecification, InputData.ParameterInput, class to use for
          specifying input of cls.
      """
      inputSpecification = super(QDataMining, cls).getInputSpecification()
      return inputSpecification

    def __init__(self):
      """
       Constructor
       @ In, None
       @ Out, None
      """
      super().__init__()
      # DataMining.__init__(self, runInfoDict)
      # qtc.QObject.__init__(self)
      self.interactive = False

    def _localReadMoreXML(self, xmlNode):
      """
        Function to grab the names of the methods this post-processor will be
        using
        @ In, xmlNode    : Xml element node
        @ Out, None
      """
      paramInput = QDataMining.getInputSpecification()()
      paramInput.parseNode(xmlNode)
      self._handleInput(paramInput)

    def _handleInput(self, paramInput):
      """
        Function to handle the parsed paramInput for this class.
        @ In, paramInput, ParameterInput, the already parsed input.
        @ Out, None
      """
      DataMining._handleInput(self, paramInput)
      for child in paramInput.subparts:
        for grandchild in child.subparts:
          if grandchild.getName() == 'interactive':
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
        uiID = str(id(self))

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
      if uiID == str(id(self)):
        self.uiDone = True
