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
Created on May 10, 2023

@author: aalfonsi
"""
#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from .BasicStatistics import BasicStatistics
from ...utils import utils
from ...utils import InputData, InputTypes
#Internal Modules End-----------------------------------------------------------

class SubdomainBasicStatistics(PostProcessorInterface):
  """
    Subdomain basic statitistics class. It computes all statistics on subdomains
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
    # We get the input specs from the basic statistics and we just add
    # the subdomain info
    inputSpecification = BasicStatistics.getInputSpecification()

    subdomainInputs = InputData.parameterInputFactory("subdomain", printPriority=100,
              descr='defines the subdomain specs to be used for the subdomain statistics')
    variableInput = InputData.parameterInputFactory("variable", printPriority=80,
              descr="defines the variables to be used for the subdomain statistics.")
    variableInput.addParam("name", InputTypes.StringNoLeadingSpacesType,
        descr=r"""Name of the variable for this grid/subdomain. \nb As for the other objects,
              this is the name that can be used to refer to this specific entity from other input blocks""")
    gridInput = InputData.parameterInputFactory("grid", contentType=InputTypes.StringType)
    gridInput.addParam("type", InputTypes.makeEnumType("type", "selectionType",['value']))
    gridInput.addParam("construction", InputTypes.StringType)
    gridInput.addParam("steps", InputTypes.IntegerType)
    variableInput.addSub(gridInput)
    subdomainInputs.addSub(variableInput)
    inputSpecification.addSub(subdomainInputs)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # delay import to allow definition
    from ...Models.PostProcessors import factory as ppFactory
    from ... import GridEntities
    self.stat = ppFactory.returnInstance('BasicStatistics')
    self.gridEntity = GridEntities.factory.returnInstance("GridEntity")
    self.validDataType  = ['PointSet', 'HistorySet', 'DataSet']
    self.outputMultipleRealizations = True
    self.printTag = 'PostProcessor SUBDOMAIN STATISTICS'

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, cellBasedData, dict, the dict of datasets of inputs and the corresponding variable probability weight (cellId:(processedDataSet, pbWeights))
    """
    cellBasedData = {}
    cellIDs = self.gridEntity.returnCellIdsWithCoordinates()
    dimensionNames =  self.gridEntity.returnParameter('dimensionNames')
    self.dynamic = False
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0:
      self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")
    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'This Postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)

    # extract all required data from input DataObjects, an input dataset is constructed
    dataSet = currentInput.asDataset()
    processedDataSet, pbWeights = self.stat.inputToInternal(currentInput)
    for cellId, verteces in cellIDs.items():
      # create masks
      maskDataset = None
      # hyperBox shape(number dimensions, number of verteces)
      hyperBox = np.atleast_2d(verteces).T
      for dim, dimName in enumerate(dimensionNames):
        if  maskDataset is None:
          maskDataset = (dataSet[dimName] >= min(hyperBox[dim])) & (dataSet[dimName] <= max(hyperBox[dim]))
        else:
          maskDataset = maskDataset & (dataSet[dimName] >= min(hyperBox[dim])) & (dataSet[dimName] <= max(hyperBox[dim]))

      # the following is the cropped dataset that we need to use for the subdomain
      #cellDataset = dataSet.where(maskDataset, drop=True)
      cellDataset = processedDataSet.where(maskDataset, drop=True)
      cellPbWeights =  pbWeights.where(maskDataset, drop=True)
      # check if at least sample is available (for scalar quantities) and at least 2 samples for derivative quantities
      setWhat = set(self.stat.what)
      minimumNumberOfSamples = 2 if len(setWhat.intersection(set(self.stat.vectorVals))) > 0 else 1
      if len(cellDataset[currentInput.sampleTag]) < minimumNumberOfSamples:
        self.raiseAnError(RuntimeError,"Number of samples in cell "
                          f"{cellId}  < {minimumNumberOfSamples}. Found {len(cellDataset[currentInput.sampleTag])}"
                          " samples within the cell. Please make the evaluation grid coarser or increase number of samples!")

      # store datasets
      cellBasedData[cellId] = cellDataset, cellPbWeights
    return cellBasedData

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the BasicStatistic pp. In here the working dir is
      grepped.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    self.stat.initialize(runInfo, inputs, initDict)
    self.gridEntity.initialize({'computeCells': True,'constructTensor': True,})
    # check that the grid is defined only in the input space.
    dims = self.gridEntity.returnParameter("dimensionNames")
    inputs = inputs[-1].getVars("input")
    if not all(item in inputs for item in dims):
      unset = ', '.join(list(set(dims) - set(inputs)))
      self.raiseAnError(RuntimeError, "Subdomain grid must be defined on the input space only (inputs)."
                        f"The following variables '{unset}' are not part of the input space of DataObject {inputs[-1].name}!")


  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    # initialize basic stats
    subdomain = paramInput.findFirst('subdomain')
    if subdomain is None:
      self.raiseAnError(IOError,'<subdomain> tag not found!')
    self.stat._handleInput(paramInput, ['subdomain'])
    self.gridEntity._handleInput(subdomain, dimensionTags=["variable"])

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, results, xarray.Dataset or dictionary, dataset or dictionary containing the results
    """
    inputData = self.inputToInternal(inputIn)
    results = {}
    outputRealization = {}
    midPoint = self.gridEntity.returnCellsMidPoints(returnDict=True)
    firstPass = True
    for i, (cellId, data) in enumerate(inputData.items()):
      cellData = self.stat.inputToInternal(data)
      res = self.stat._runLocal(cellData)
      for k in res:
        if firstPass:
          results[k] = np.zeros(len(inputData), dtype=object if self.stat.dynamic else None)
        results[k][i] = res[k][0] if not self.stat.dynamic else res[k]
        for k in midPoint[cellId]:
          #res[k] =  np.atleast_1d(midPoint[cellId][k])
          if firstPass:
            results[k] = np.zeros(len(inputData))
          results[k][i] =  np.atleast_1d(midPoint[cellId][k])
      firstPass = False
    outputRealization['data'] =  results
    if self.stat.dynamic:
      dims = dict.fromkeys(results.keys(), inputIn[-1].indexes if type(inputIn) == list else inputIn.indexes)
      for k in list(midPoint.values())[0]:
        dims[k] = []
      outputRealization['dims'] = dims

    return outputRealization

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler.Runner instance, the instance containing the completed job
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    super().collectOutput(finishedJob, output)
