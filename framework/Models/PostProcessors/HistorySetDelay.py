
"""
This is to implement a delay or lagged parameters in a HistorySet
"""

import copy
import numpy as np
import xarray as xr

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase, CheckInterfacePP
from utils import InputData, InputTypes
from .PostProcessor import PostProcessor

class HistorySetDelay(PostProcessor):
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

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the HistorySetDelay
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)
    self.validDataType = ['HistorySet']
    #self.inputFormat = 'HistorySet'
    #self.outputFormat = 'HistorySet'
    #self.outputMultipleRealizations = True

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
      @ Out, outputDic, dict, output dictionary
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'HistorySetDelay Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')

    inputDic = inputDic[0]
    newHistorySet = copy.deepcopy(inputDic)
    for delay in self.delays:
      original, new, steps, default = delay
      orig_data = inputDic._data[original]
      new_data = np.empty(orig_data.shape, orig_data.dtype)
      new_data.fill(default)
      if steps < 0:
          new_data[:,-steps:] = orig_data[:,:steps]
      elif steps > 0:
        new_data[:,:-steps] = orig_data[:,steps:]
      else:
        # steps is 0, so just copy array
        new_data[:,:] = orig_data[:,:]
      #XXX How do we add this to the history set?
      #newHistorySet.addVariable(new, xr.DataArray(new_data, dims=orig_data.dims), 'output')
      #newHistorySet.addRealization({new:new_data})
      #import pdb; pdb.set_trace()
      #newHistorySet._data[new] = (orig_data.dims, new_data)
    #XXX How do we return this history set?
    #outDict = newHistorySet._convertToDict()
    #import pdb; pdb.set_trace()
    #return {"output":newHistorySet}
    #return outDict
    return
