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
  Created on March 20, 2021
  @author: alfoa
  description: Postprocessor named Validation. This postprocessor is aimed to
               to represent a gate for any validation tecniques and processes
"""

#External Modules---------------------------------------------------------------
import numpy as np
import copy
import time
import xarray as xr
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from . import validationAlgorithms
from utils import utils, mathUtils
from utils import InputData, InputTypes
import DataObjects
import MetricDistributor
#Internal Modules End-----------------------------------------------------------


class Validation(PostProcessorInterface):
  """
    Validation class. It will apply the specified validation algorithms in
    the models to a dataset, each specified algorithm's output can be loaded to
    dataObject.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    specs = super(Validation, cls).getInputSpecification()
    preProcessorInput = InputData.parameterInputFactory("PreProcessor", contentType=InputTypes.StringType)
    preProcessorInput.addParam("class", InputTypes.StringType)
    preProcessorInput.addParam("type", InputTypes.StringType)
    specs.addSub(preProcessorInput)
    pivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    specs.addSub(pivotParameterInput)
    featuresInput = InputData.parameterInputFactory("Features", contentType=InputTypes.StringListType)
    featuresInput.addParam("type", InputTypes.StringType)
    specs.addSub(featuresInput)
    targetsInput = InputData.parameterInputFactory("Targets", contentType=InputTypes.StringListType)
    targetsInput.addParam("type", InputTypes.StringType)
    specs.addSub(targetsInput)
    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    metricInput.addParam("class", InputTypes.StringType)
    metricInput.addParam("type", InputTypes.StringType)
    specs.addSub(metricInput)
    # registration of validation algorithm
    for typ in validationAlgorithms.factory.knownTypes():
      algoInput = validationAlgorithms.factory.returnClass(typ)
      specs.addSub(algoInput.getInputSpecification())
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR VALIDATION'
    self.pivotParameter = None  ## default pivotParameter for HistorySet
    self._type = None           ## the type of library that are used for validation, i.e. DSS
    # add assembly objects (and set up pointers)
    self.PreProcessor = None    ## Instance of PreProcessor, default is None
    self.metrics = None          ## Instance of Metric, default is None
    self.addAssemblerObject('Metric', InputData.Quantity.one_to_infinity)
    self.addAssemblerObject('PreProcessor', InputData.Quantity.zero_to_infinity)

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

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if 'PreProcessor' in self.assemblerDict:
      self.PreProcessor = self.assemblerDict['PreProcessor'][0][3]
    if 'Metric' in self.assemblerDict:
      metrics = [metric[3] for metric in self.assemblerDict['Metric']]
      self.metrics = [MetricDistributor.factory.returnInstance('MetricDistributor', metric) for metric in metrics]

    if len(inputs) > 1:
      # if inputs > 1, check if the | is present to understand where to get the features and target
      notStandard = [k for k in self.features + self.targets if "|" not in k]
      if notStandard:
        self.raiseAnError(IOError, "# Input Datasets/DataObjects > 1! features and targets must use the syntax DataObjectName|feature to be usable! Not standard features are: {}!".format(",".join(notStandard)))
    # now lets check that the variables are in the dataobjects
    if isinstance(inputs[0], DataObjects.DataSet):
      do = [inp.name for inp in inputs]
      if len(inputs) > 1:
        allFound = [feat.split("|")[0].strip() in do for feat in self.features]
        allFound += [targ.split("|")[0].strip() in do for targ in self.targets]
        if not all(allFound):
          self.raiseAnError(IOError, "Targets and Features are linked to DataObjects that have not been listed as inputs in the Step. Please check input!")
      # check variables
      for indx, dobj in enumerate(do):
        variables = [var.split("|")[-1].strip() for var in (self.features + self.targets) if dobj in var]
        if not utils.isASubset(variables,inputs[indx].getVars()):
          self.raiseAnError(IOError, "The variables '{}' not found in input DataObjet '{}'!".format(",".join(list(set(list(inputs[indx].getVars())) - set(variables))), dobj))
    self.model.initialize(self.features, self.targets, **{'metrics': self.metrics, 'pivotParameter': self.pivotParameter})

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

    ## FIXME: this should be a type of the node <Algorithm> once we can handel "conditional choice" in InputData:
    ## ******* Replace:
    ##<PostProcessor name="blabla">
    ##  <Validation name="2bla2bla">
    ##    ...
    ##    <DSS>
    ##
    ##    </DSS>
    ##  </Validation>
    ##</PostProcessor>
    ## ******* with:
    ##<PostProcessor name="blabla">
    ##  <Validation name="2bla2bla">
    ##    ...
    ##    <Algorithm type="DSS">
    ##
    ##    </Agorithm>
    ##  </Validation>
    ##</PostProcessor>
    # check algorithms
    valAlgo = validationAlgorithms.factory.knownTypes()
    foundAll = [paramInput.findFirst(algo) for algo in valAlgo]
    nNone =  foundAll.count(None)
    if nNone != len(valAlgo) - 1:
      msg =  "Only one validation algorithm at the time can be inputted in PostProcessor {}. Got >= 1. Check your input!".format(self.name)  \
        if nNone != len(valAlgo) else "No validation algorithm has been specified in PostProcessor {}".format(self.name)
      self.raiseAnError(IOError, msg)
    # get validation algorithm to apply
    modelInputPart = utils.first([x for x in foundAll if x is not None])
    self._type =  modelInputPart.name
    # return algo instance
    self.model = validationAlgorithms.factory.returnInstance(self._type)
    # handle input in the interface instance
    self.model._handleInput(modelInputPart)
    # this loop set the pivot parameter (it could use paramInput.findFirst but we want to show how to add more paramters)
    for child in paramInput.subparts:
      if child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      elif child.getName() == 'Features':
        self.features = child.value
      elif child.getName() == 'Targets':
        self.targets = child.value
    if 'static' not in self.model.dataType and self.pivotParameter is None:
      self.raiseAnError(IOError, "The validation algorithm '{}' is a dynamic model ONLY but no <pivotParameter> node has been inputted".format(self._type))
    if not self.features:
      self.raiseAnError(IOError, "XML node 'Features' is required but not provided")
    elif len(self.features) != len(self.targets):
      self.raiseAnError(IOError, 'The number of variables found in XML node "Features" is not equal the number of variables found in XML node "Targets"')

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler
        object that is in charge of running this post-processor
      @ In, output, dataObjects, A reference to an object where we want
        to place our computed results
      @ Out, None
    """
    if len(output) !=0:
      self.raiseAnError(IOError,"There is some information already stored in the DataObject",output.name, \
              "the calculations from PostProcessor",self.name, " can not be stored in this DataObject!", \
              "Please provide a new empty DataObject for this PostProcessor!")
    ## When does this actually happen?
    evaluation = finishedJob.getEvaluation()
    _, validationDict = evaluation

    self.raiseADebug('Adding output in data object named', output.name)
    rlz = {}
    for key, val in validationDict.items():
      rlz[key] = val
    output.addRealization(rlz)
    # add metadata
    #  in case we want to add specific metdata, we can add the functionality in the evalidation algo base class

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    # assert
    assert(isinstance(inputIn, list))
    assert(isinstance(inputIn[0], xr.Dataset) or isinstance(inputIn[0], DataObjects.DataSet))
    # the input can be either be a list of dataobjects or a list of datasets (xarray)
    datasets = [inp if isinstance(inp, xr.Dataset) else inp.asDataset() for inp in inputIn]
    names = []
    pivotParameter = self.pivotParameter
    if isinstance(inputIn[0], DataObjects.DataSet):
      names =  [inp.name for inp in inputIn]
      if len(inputIn[0].indexes) and self.pivotParameter is None:
        if 'dynamic' not in self.model.dataType:
          self.raiseAnError(IOError, "The validation algorithm '{}' is not a dynamic model but time-dependent data has been inputted in object {}".format(self._type, inputIn[0].name))
        else:
          pivotParameter = inputIn[0].indexes[0]
    #  check if pivotParameter
    if pivotParameter:
      #  in case of dataobjects we check that the dataobject is either an HistorySet or a DataSet
      if isinstance(inputIn[0], DataObjects.DataSet) and not all([True if inp.type in ['HistorySet', 'DataSet']  else False for inp in inputIn]):
        self.raiseAnError(RuntimeError, "The pivotParameter '{}' has been inputted but PointSets have been used as input of PostProcessor '{}'".format(pivotParameter, self.name))
      if not all([True if pivotParameter in inp else False for inp in datasets]):
        self.raiseAnError(RuntimeError, "The pivotParameter '{}' not found in datasets used as input of PostProcessor '{}'".format(pivotParameter, self.name))
    evaluation ={k: np.atleast_1d(val) for k, val in  self.model.run(datasets, **{'dataobjectNames': names}).items()}

    if pivotParameter:
      if len(datasets[0][pivotParameter]) != len(list(evaluation.values())[0]):
        self.raiseAnError(RuntimeError, "The pivotParameter value '{}' has size '{}' and validation output has size '{}'".format( len(datasets[0][self.pivotParameter]), len(evaluation.values()[0])))
      if pivotParameter not in evaluation:
        evaluation[pivotParameter] = datasets[0][pivotParameter]
    return evaluation
