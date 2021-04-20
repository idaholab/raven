
"""
This is to implement a delay or lagged parameters in a HistorySet
"""

import copy
import numpy as np

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase, CheckInterfacePP
from utils import InputData, InputTypes

class HistorySetDelay(PostProcessorInterfaceBase):
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
    inputSpecification.setCheckClass(CheckInterfacePP("HistorySetDelay"))
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

  def initialize(self):
    """
      Method to initialize the HistorySetDelay
    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat = 'HistorySet'
    self.outputFormat = 'HistorySet'

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
    outputDic = copy.deepcopy(inputDic)
    for delay in self.delays:
      original, new, steps, default = delay
      orig_data = inputDic['data'][original]
      new_data = np.empty(orig_data.shape, orig_data.dtype)
      for i, array in enumerate(orig_data):
        new_data[i] = np.empty(array.shape)
        new_data[i].fill(default)
        if steps < 0:
          new_data[i][-steps:] = array[:steps]
        elif steps > 0:
          new_data[i][:-steps] = array[steps:]
        else:
          # steps is 0, so just copy array
          new_data[i][:] = array[:]
        outputDic['data'][new] = new_data
      outputDic['outVars'].append(new)
    return outputDic
