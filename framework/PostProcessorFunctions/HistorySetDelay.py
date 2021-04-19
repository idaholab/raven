
"""
This is to implement a delay or lagged parameters in a HistorySet
"""

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
    inputSpecification.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType))
    delayClass = InputData.parameterInputFactory("delay", InputTypes.StringType)
    delayClass.addParam("original", InputTypes.StringType, True)
    delayClass.addParam("new", InputTypes.StringType, True)
    delayClass.addParam("steps", InputTypes.IntegerType, True)
    delayClass.addParam("default", InputTypes.FloatType, True)
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
    pass
