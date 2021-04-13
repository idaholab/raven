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
  Created on April 04, 2021

  @author: alfoa
  
  This class represents a base class for the validation algorithms
  It inherits from the PostProcessor directly
  ##TODO: Recast it once the new PostProcesso API gets in place
"""

#External Modules------------------------------------------------------------------------------------

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#from utils import utils, InputData, InputTypes
import BaseInterface
#Internal Modules End--------------------------------------------------------------------------------

class ValidationBase(BaseInterface):
  """
    ValidationBase is a base class for validation problems
    It represents the base class for most validation problems
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    return spec

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'ValidationBase'
    self.dynamic        = False # is it time-dependent?
    self.features       = None  # list of feature variables
    self.targets        = None  # list of target variables
    self.pivotParameter = None  # pivot parameter (present if dynamic == True)

  def initialize(self, features, targets, **kwargs):
    """
      Set up this interface for a particular activity
      @ In, features, list, list of features
      @ In, targets, list, list of targets
      @ In, kwargs, dict, keyword arguments
    """
    super().initialize()
    self.features, self.targets = features, targets
    self.pivotParameter = kwargs.get('pivotParameter')
    self.metrics = kwargs.get('Metric')

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    pass
    
  def run(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, tuple, tuple of datasets (data1,data2,etc.) to "validate"
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    
    
    return None
    
    
    
    
