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
from BaseClasses import BaseInterface
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
    self.dataType = ['static', 'dynamic'] # the type of data can be passed in (static aka PointSet, dynamic aka HistorySet) (if both are present the validation algorithm can work for both data types)
    self.acceptableMetrics = [] # if not populated all types of metrics are accepted, otherwise list the metrics (see Probablistic.py for an example)
    self.features = None        # list of feature variables
    self.targets = None         # list of target variables
    self.pivotParameter = None  # pivot parameter (present if dynamic == True)
    self.pivotValues = None     # pivot values (present if dynamic == True)

  def initialize(self, features, targets, **kwargs):
    """
      Set up this interface for a particular activity
      @ In, features, list, list of features
      @ In, targets, list, list of targets
      @ In, kwargs, dict, keyword arguments
    """
    super().initialize(features, targets, **kwargs)
    self.features, self.targets = features, targets
    self.pivotParameter = kwargs.get('pivotParameter')
    self.metrics = kwargs.get('metrics')
    if self.acceptableMetrics:
      accetable = [True if metric.type in self.acceptableMetrics else False for metric in self.metrics]
      if not all(accetable):
        notAcceptable = [self.metrics[i].type for i, x in enumerate(accetable) if not x]
        self.raiseAnError(IOError, "The metrics '{}' are not acceptable for validation algorithm {}".format(','.join(notAcceptable), self.name))

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

  ### utility functions

  def _getDataFromDatasets(self, datasets, var, names=None):
    """
      Utility function to retrieve the data from datasets
      @ In, datasets, list, list of datasets (data1,data2,etc.) to search from.
      @ In, names, list, optional, list of datasets names (data1,data2,etc.). If not present, the search will be done on the full list.
      @ In, var, str, the variable to find (either in fromat dataobject|var or simply var)
      @ Out, data, xarray.DataArray, the retrived data
    """
    data = None
    if "|" in var and names is not None:
      do, feat =  var.split("|")
      data = datasets[names.index(do)][feat]
    else:
      for ds in datasets:
        if var in ds:
          data = ds[var]
          break
    return data

