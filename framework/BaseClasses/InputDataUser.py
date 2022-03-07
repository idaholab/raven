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
Created on March 22, 2021

@author: talbpaul
"""
from ..utils import InputData, InputTypes

class InputDataUser(object):
  """
    Inheriting from this class grants access to methods used by the InputData.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, None
      @ Out, spec, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = InputData.parameterInputFactory(cls.__name__, ordered=False, baseNode=InputData.RavenBase)
    return spec

  def parseXML(self, xml):
    """
      Parse XML into input parameters
      @ In, xml, xml.etree.ElementTree.Element, XML element node
      @ Out, InputData.ParameterInput, the parsed input
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xml)
    return paramInput

  def handleInput(self, paramInput, **kwargs):
    """
      Handles the input from the user.
      @ In, InputData.ParameterInput, the parsed input
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    pass # extend this method to parse input
