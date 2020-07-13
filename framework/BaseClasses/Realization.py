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
Created on Jan 20, 2015

@author: senrs
based on alfoa design

"""
from __future__ import division, print_function, unicode_literals, absolute_import
#External Modules------------------------------------------------------------------------------------
import abc
import xarray as xa
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
from utils import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class Realization(MessageHandler.MessageUser):
  """
    The Realization class is used as container for the realization that are
    generated via RAVEN optimizers and samplers.
    The realization concept is abstracted here since Populations can be
    considered realizations too
  """
  def __init__(self, rlz = None):
    """
      Constructor
      @ In, rlz, dict or xarray.DataArray, optional, the realization to encapsulate
      @ In, rlz, None
      @ Out, None
    """
    # xa.DataArray.from_dict(d)
    self.name = self.__class__.__name__ # name
    self.type = type(rlz).__name__ # type
    self._rlz = None # the rlz container   
    self._shape = None # shape of the realization (e.g. single realization _shape = 1, popuplation _shape = n)
    self._converted = False # convert to xarray DataSet?   
    self._id =  None #  realization id (e.g. job id or prefix or such)
    self._nVars =  0 #  number of variables currently stored
  
  ### EXTERNAL API ###
  def addRealization(self, rlz, **kwargs):
    """
      Method to add a complete realization.
      @ In, rlz, dict or list(dict), the dictionary containing the realization {'var1':ndarray,..,'varN':ndarray} or a list of such dictionaries
      @ In, **kwargs, kwarded dict, options to link to this realization. The following options are available:
                                                                         - id, string, the realization id (this can be either be the job id or the population id). Default "None"
                                                                         - toXArray, bool, convert to xarray DataSet?, Default False
      the dictionary containing the realization {'var1':ndarray,..,'varN':ndarray} or a list of such dictionaries
    """
    assert ()
    self._id = kwargs.get("id", "None")
    self._converted = kwargs.get("toXArray", False)
    if isinstance(rlz, list):
      self._shape = len(rlz)
      
    else:
      self._shape = 1
      self._nVars = len(rlz)
      if self._converted:
        
      else:
        self.rlz = rlz
      
    
    
    
    
    
      
    
    
    
    
    
    
   
  def realization(self, ):
    """
     
    
    """
    pass 

  def getVariable(self, ):
    """
     
    
    """
    pass
  
  def getVariable(self, ):
    """
     
    
    """
    pass   
   
 
  ### BUIlTINS AND PROPERTIES ###
  # These are special commands that RAVEN entities can use to interact with the data object
  def __len__(self):
    """
      Overloads the len() operator.
      @ In, None
      @ Out, int, number of samples in this dataset
    """
    return self.size

  @property
  def isEmpty(self):
    """
      @ In, None
      @ Out, boolean, True if the dataset is empty otherwise False
    """
    empty = True if self.size == 0 else False
    return empty

  @property
  def vars(self):
    """
      Property to access all the pointwise variables being controlled by this data object.
      As opposed to "self._orderedVars", returns the variables clustered by subset (inp, out, meta) instead of order added
      @ In, None
      @ Out, vars, list(str), variable names list
    """
    return self._inputs + self._outputs + self._metavars

  @property
  def size(self):
    """
      Property to access the amount of data in this data object.
      @ In, None
      @ Out, size, int, number of samples
    """
    s = 0 # counter for size
    # from collector
    s += self._collector.size if self._collector is not None else 0
    # from data
    try:
      s += len(self._data[self.sampleTag]) if self._data is not None else 0
    except KeyError: #sampleTag not found, so it _should_ be empty ...
      s += 0
    return s

  @property
  def indexes(self):
    """
      Property to access the independent axes in this problem
      @ In, None
      @ Out, indexes, list(str), independent index names (e.g. ['time'])
    """
    return list(self._pivotParams.keys())
