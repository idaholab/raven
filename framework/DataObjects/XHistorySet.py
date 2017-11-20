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

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys,os
import __builtin__
import functools
import copy
import cPickle as pk
import xml.etree.ElementTree as ET

import abc
import numpy as np
import pandas as pd
import xarray as xr

from BaseClasses import BaseType
from Files import StaticXMLOutput
from utils import utils, cached_ndarray, InputData, xmlUtils, mathUtils
try:
  from .XDataSet import DataSet
except ValueError: #attempted relative import in non-package
  from XDataSet import DataSet

# for profiling with kernprof
try:
  __builtin__.profile
except AttributeError:
  # profiler not preset, so pass through
  def profile(func): return func

#
#
#
#
class HistorySet(DataSet):
  """
    DataObject developed Oct 2017 to obtain linear performance from data objects when appending, over
    thousands of variables and millions of samples.  Wraps np.ndarray for collecting and uses xarray.Dataset
    for final form.  This form is a shortcut for ASSUMED only-float inputs and shared-single-pivot outputs
  """
  # only a few changes from the base class; the external API is identical.

  ### INITIALIZATION ###
  # These are the necessary functions to construct and initialize this data object
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    DataSet.__init__(self)
    self.name      = 'HistorySet'
    self.type      = 'HistorySet'
    self.printTag  = self.name
    self._tempPivotParam = None

  def _readMoreXML(self,xmlNode):
    """
      Initializes data object based on XML input.
      @ In, xmlNode, xml.etree.ElementTree.Element or InputData.ParameterInput, input specification
      @ Out, None
    """
    DataSet._readMoreXML(self,xmlNode)
    # propagate provided pivot parameter to all variables.
    # If not set, use "time" as default.
    if self._tempPivotParam is None:
      self.raiseAWarning('No pivot parameter provided; defaulting to \"time\".')
      self._tempPivotParam = 'time'
    # don't use setter, set directly, since there's only one var
    self._pivotParams = {self._tempPivotParam:self._outputs[:]}

  ### INTERNAL USE FUNCTIONS ###
  def _collapseNDtoDataArray(self,data,var,labels=None):
    """
      Converts a row of numpy samples into a single DataArray suitable for a xr.Dataset.
      @ In, data, np.ndarray, array of either float or xr.DataArray; array must be single-dimension
      @ In, var, str, name of the variable being acted on
      @ In, labels, list, list of labels to use for collapsed array under self.sampleTag title
      @ Out, DataArray, xr.DataArray, single dataarray object
    """
    # TODO this is only type-checking before using the base class implementation.
    ## TODO these assertions are identical to the base class right now; should abstract
    assert(isinstance(data,np.ndarray))
    assert(len(data.shape) == 1)
    if labels is None:
      labels = range(len(data))
    else:
      assert(len(labels) == len(data))
    ## these assertions are specific to history sets -> should they be in addRealization instead?
    # Inputs and meta should all be single entries, outputs should all be xr.DataArray that depend only on pivotParam
    if var in self._inputs:
      assert(isinstance(data[0],(float,str,unicode,int)))
    elif var in self._outputs:
      # all outputs are xr.DataArrays
      assert(isinstance(data[0],xr.DataArray))
      # all outputs have a single independent coordinate
      assert(len(data[0].dims) == 1)
      # all outputs depend only on the pivot parameter
      assert(data[0].dims[0] == self._pivotParams.keys()[0])
    return DataSet._collapseNDtoDataArray(self,data,var,labels)

  def _selectiveRealization(self,rlz):
    """
      Uses "options" parameters from input to select part of the collected data
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    # TODO this could be much more efficient on the parallel (finalizeCodeOutput) than on serial
    # TODO someday needs to be implemented for when ND data is collected!  For now, use base class.
    return DataSet._selectiveRealization(self,rlz)
