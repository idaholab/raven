
"""
This is to implement a delay or lagged parameters in a HistorySet
"""

import copy
import numpy as np
import xarray as xr

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase, CheckInterfacePP
from utils import InputData, InputTypes
from .PostProcessorInterface import PostProcessorInterface

class HistorySetDelay(PostProcessorInterface):
  """
  Class to get lagged or delayed data out of a history set.
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
    #inputSpecification.setCheckClass(CheckInterfacePP("HistorySetDelay"))
    delayClass = InputData.parameterInputFactory("delay", InputTypes.StringType,
                                                 descr="Adds a delay variable")
    delayClass.addParam("original", InputTypes.StringType, True,
                        descr="Original variable name")
    delayClass.addParam("new", InputTypes.StringType, True,
                        descr="New (delayed) variable name")
    delayClass.addParam("steps", InputTypes.IntegerType, True,
                        descr="Steps to offset (-1 is previous step)")
    delayClass.addParam("default", InputTypes.FloatType, True,
                        descr="Default value to use for unavailable steps")
    inputSpecification.addSub(delayClass, InputData.Quantity.one_to_infinity)
    inputSpecification.addSub(InputData.parameterInputFactory("method", contentType=InputTypes.StringType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.validDataType = ['HistorySet']       #only available output is HistorySet
    self.outputMultipleRealizations = True    #this PP will return a full set of realization
    self.printTag = 'PostProcessor HistorySetDelay'

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the HistorySetDelay
    """
    super().initialize(runInfo, inputs, initDict)

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    self.delays = []
    for child in paramInput.subparts:
      if child.getName() == 'delay':
        self.delays.append((child.parameterValues['original'],
                            child.parameterValues['new'],
                            child.parameterValues['steps'],
                            child.parameterValues['default']))

  def run(self,inputDic):
    """
      Method to post-process the dataObjects
      @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
      @ Out, data, xarray.DataSet, output dataset
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'HistorySetDelay Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')

    data = inputDic[0].asDataset()
    for delay in self.delays:
      original, new, steps, default = delay
      coords = {key: data[original][key] for key in data[original].dims}
      orig_data = data[original].values
      new_data = copy.copy(orig_data)
      if steps < 0:
        new_data[:, :-steps] = default
        new_data[:, -steps:] = orig_data[:,:steps]
      elif steps > 0:
        new_data[:, -steps:] = default
        new_data[:, :-steps] = orig_data[:,steps:]
      # else:
      # steps is 0, so just keep the copy
      data[new] = xr.DataArray(data=new_data, coords=coords, dims=coords.keys())
    return data
