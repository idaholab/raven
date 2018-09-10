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
"""
  Contains the general class for DataObjects who may have mixed scalars, vectors, and higher dimensional
  needs, depending on any combination of "index" dimensions (time, space, etc).
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import sys
import copy
import itertools
import __builtin__
import cPickle as pk
import xml.etree.ElementTree as ET

import abc
import numpy as np
import pandas as pd
import xarray as xr

# relative import for RAVEN, local import for unit tests
try:
  from .DataObject import DataObject
except ValueError:
  from DataObject import DataObject
from Files import StaticXMLOutput
from utils import utils, cached_ndarray, InputData, xmlUtils, mathUtils

# for profiling with kernprof
try:
  __builtin__.profile
except AttributeError:
  # profiler not preset, so pass through
  def profile(func):
    """
      Dummy for when profiler is missing.
      @ In, func, method, method to run
      @ Out, func, method, method to run
    """
    return func

#
#
#
#
class DataSet(DataObject):
  """
    This class outlines the behavior for the basic in-memory DataObject, including support
    for ND and ragged input/output variable data shapes.  Other in-memory DataObjects are
    specialized implementations of this class.

    DataObject developed Oct 2017 with the intent to obtain linear performance from data objects when appending, over
    thousands of variables and millions of samples.  Wraps np.ndarray for collecting and uses xarray.Dataset
    for final form.  Subclasses are shortcuts (recipes) for this most general case.

    The interface for these data objects is specific.  The methods under "EXTERNAL API", "INITIALIZATION",
    and "BUILTINS" are the only methods that should be called to interact with the object.
  """
  ### INITIALIZATION ###
  # These are the necessary functions to construct and initialize this data object
  def __init__(self):#, in_vars, out_vars, meta_vars=None, dynamic=False, var_dims=None,cacheSize=100,prealloc=False):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    DataObject.__init__(self)
    self.name             = 'DataSet'
    self.type             = 'DataSet'
    self.types            = None             # list of type objects, for each realization entry
    self.printTag         = self.name
    self.defaultDtype     = object
    self._scaleFactors    = {}               # mean, sigma for data for matching purposes
    self._alignedIndexes  = {}               # dict {index:values} of indexes with aligned coordinates (so they are not in the collector, but here instead)
    self._neededForReload = [self.sampleTag] # metavariables required to reload this data object.

  def _readMoreXML(self,xmlNode):
    """
      Initializes data object based on XML input
      @ In, xmlNode, xml.etree.ElementTree.Element, input information
      @ Out, None
    """
    inp = DataSet.getInputSpecification()()
    inp.parseNode(xmlNode)
    # let parent read first
    DataObject._readMoreXML(self,inp)
    # any additional custom reading below

  ### EXTERNAL API ###
  # These are the methods that RAVEN entities should call to interact with the data object
  def addExpectedMeta(self,keys):
    """
      Registers meta to look for in realizations.
      @ In, keys, set(str), keys to register
      @ Out, None
    """
    # TODO add option to skip parts of meta if user wants to
    # remove already existing keys
    keys = list(key for key in keys if key not in self.getVars()+self.indexes)
    # if no new meta, move along
    if len(keys) == 0:
      return
    # CANNOT add expected meta after samples are started
    assert(self._data is None)
    assert(self._collector is None or len(self._collector) == 0)
    self._metavars.extend(keys)
    self._orderedVars.extend(keys)

  def addMeta(self,tag,xmlDict):
    """
      Adds general (not pointwise) metadata to this data object.  Can add several values at once, collected
      as a dict keyed by target variables.
      Data ends up being written as follows (see docstrings above for dict structure)
       - A good default for 'target' is 'general' if there's not a specific target
      <tag>
        <target>
          <scalarMetric>value</scalarMetric>
          <scalarMetric>value</scalarMetric>
          <vectorMetric>
            <wrt>value</wrt>
            <wrt>value</wrt>
          </vectorMetric>
        </target>
        <target>
          <scalarMetric>value</scalarMetric>
          <vectorMetric>
            <wrt>value</wrt>
          </vectorMetric>
        </target>
      </tag>
      @ In, tag, str, section to add metadata to, usually the data submitter (BasicStatistics, DataObject, etc)
      @ In, xmlDict, dict, data to change, of the form {target:{scalarMetric:value,scalarMetric:value,vectorMetric:{wrt:value,wrt:value}}}
      @ Out, None
    """
    # TODO potentially slow if MANY top level tags
    if tag not in self._meta.keys():
      # TODO store elements as Files object XML, for now
      new = StaticXMLOutput()
      new.initialize(self.name,self.messageHandler) # TODO replace name when writing later
      new.newTree(tag)
      self._meta[tag] = new
    destination = self._meta[tag]
    for target in xmlDict.keys():
      for metric,value in xmlDict[target].items():
        # Two options: if a dict is given, means vectorMetric case
        if isinstance(value,dict):
          destination.addVector(target,metric,value)
        # Otherwise, scalarMetric
        else:
          # sanity check to make sure suitable values are passed in
          assert(mathUtils.isSingleValued(value))
          destination.addScalar(target,metric,value)

  def addRealization(self,rlz):
    """
      Adds a "row" (or "sample") to this data object.
      This is the method to add data to this data object.
      Note that rlz can include many more variables than this data object actually wants.
      Before actually adding the realization, data is formatted for this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" is a np.ndarray of values.
      @ Out, None
    """
    # protect against back-changing realization
    rlz = copy.deepcopy(rlz)
    # clean out entries that aren't desired
    try:
      rlz = dict((var,rlz[var]) for var in self.getVars()+self.indexes)
    except KeyError as e:
      self.raiseADebug('Variables provided:',rlz.keys())
      self.raiseAnError(KeyError,'Provided realization does not have all requisite values for object "{}": "{}"'.format(self.name,e.args[0]))
    # check consistency, but make it an assertion so it can be passed over
    if not self._checkRealizationFormat(rlz):
      self.raiseAnError(SyntaxError,'Realization was not formatted correctly for "{}"! See warnings above.'.format(self.name))
    # format the data
    rlz = self._formatRealization(rlz)
    ## establish types if not done yet
    self._setDataTypes(rlz)
    # perform selective collapsing/picking of data
    rlz = self._selectiveRealization(rlz)

    ## check alignment of indexes
    self._checkAlignedIndexes(rlz)
    #  NB If no scalar entry is made, this construction fails.  In that case,
    #  instead of treating each dataarrray as an object, numpy.asarray calls their asarray methods,
    #  unfolding them and making a full numpy array with more dimensions, instead of effectively
    #  a list of realizations, where each realization is effectively a list of xr.DataArray objects.
    #
    #  To mitigate this behavior, we forcibly add a [0.0] entry to each realization, then exclude
    #  it once the realizations are constructed.  This seems like an innefficient option; others
    #  should be explored.  - talbpaul, 12/2017
    # newData is a numpy array of realizations,
    #   each of which is a numpy array of some combination of scalar values and/or xr.DataArrays.
    #   This is because the cNDarray collector expects a LIST of realization, not a single realization.
    #   Maybe the "append" method should be renamed to "extend" or changed to append one at a time.
    ## set realizations as a list of realizations (which are ordered lists)
    newData = np.array(list(rlz[var] for var in self._orderedVars)+[0.0],dtype=object)
    newData = newData[:-1]
    # if data storage isn't set up, set it up
    if self._collector is None:
      self._collector = self._newCollector(width=len(rlz))
    # append
    self._collector.append(newData)

    # if hierarchical, clear the parent as an ending
    self._clearParentEndingStatus(rlz)
    # reset scaling factors, kd tree
    self._resetScaling()

  def addVariable(self,varName,values,classify='meta'):
    """
      Adds a variable/column to the data.  "values" needs to be as long as self.size.
      @ In, varName, str, name of new variable
      @ In, values, np.array, new values (floats/str for scalars, xr.DataArray for hists)
      @ In, classify, str, optional, either 'input', 'output', or 'meta'
      @ Out, None
    """
    # TODO might be removable
    assert(isinstance(values,np.ndarray))
    assert(len(values) == self.size)
    assert(classify in ['input','output','meta'])
    # first, collapse existing entries
    self.asDataset()
    # format as single data array
    # TODO worry about sampleTag values?
    column = self._collapseNDtoDataArray(values,varName,labels=self._data[self.sampleTag])
    # add to the dataset
    self._data = self._data.assign(**{varName:column})
    if classify == 'input':
      self._inputs.append(varName)
    elif classify == 'output':
      self._outputs.append(varName)
    else:
      self._metavars.append(varName)
    self._orderedVars.append(varName)

  def asDataset(self, outType='xrDataset'):
    """
      Casts this dataObject as dictionary or an xr.Dataset depending on outType.
      @ In, outType, str, optional, type of output object (xr.Dataset or dictionary).
      @ Out, xr.Dataset or dictionary.  If dictionary, a copy is returned; if dataset, then a reference is returned.
    """
    if outType == 'xrDataset':
      # return reference to the xArray
      return self._convertToXrDataset()
    elif outType=='dict':
      # return a dict (copy of data, no link to original)
      return self._convertToDict()
    else:
      # raise an error
      self.raiseAnError(ValueError, 'DataObject method "asDataset" has been called with wrong '
                                    'type: ' +str(outType) + '. Allowed values are: xrDataset, dict.')

  def checkIndexAlignment(self,indexesToCheck=None):
    """
      Checks that all realizations share common coordinates along given indexes.
      That is, assures data is not sparse, but full (no NaN entries).
      @ In, indexesToCheck, list(str) or str or None, optional, indexes to check (or single index if string, or if None will check ALL indexes)
      @ Out, same, bool, if True then alignment is good
    """
    # format request so that indexesToCheck is always a list
    if mathUtils.isAString(indexesToCheck):
      indexesToCheck = [indexesToCheck]
    elif indexesToCheck is None:
      indexesToCheck = self.indexes[:]
    else:
      try:
        indexesToCheck = list(indexesToCheck) # TODO what if this errs?
      except TypeError:
        self.raiseAnError('Unrecognized input to checkIndexAlignment!  Expected list, string, or None, but got "{}"'.format(type(indexesToCheck)))
    # check the alignment of each index by checking for NaN values in each slice
    data = self.asDataset()
    for index in indexesToCheck:
      # check that index is indeed an index
      assert(index in self.indexes)
      # get a typical variable from set to look at
      ## NB we can do this because each variable within one realization must be aligned with the rest
      ##    of the variables in that same realization, so checking one variable that depends on "index"
      ##    is as good as checking all of them.
      ##TODO: This approach is only working for our current data struture, for ND case, this should be
      ## improved.
      data = data[self._pivotParams[index][-1]]
      # if any nulls exist in this data, this suggests missing data, therefore misalignment.
      if data.isnull().sum() > 0:
        self.raiseADebug('Found misalignment index variable "{}".'.format(index))
        return False
    # if you haven't returned False by now, you must be aligned
    return True

  def constructNDSample(self,vals,dims,coords,name=None):
    """
      Constructs a single realization instance (for one variable) from a realization entry.
      @ In, vals, np.ndarray, should have shape of (len(coords[d]) for d in dims)
      @ In, dims, list(str), names of dependent dimensions IN ORDER of appearance in vals, e.g. ['time','x','y']
      @ In, coords, dict, {dimension:list(float)}, values for each dimension at which 'val' was obtained, e.g. {'time':
      @ Out, obj, xr.DataArray, completed realization instance suitable for sending to "addRealization"
    """
    # while simple, this API will allow easier extensibility in the future.
    obj = xr.DataArray(vals,dims=dims,coords=coords)
    obj.rename(name)
    return obj

  def getDimensions(self,var=None):
    """
      Provides the independent dimensions that this variable depends on.
      To get all dimensions at once, use self.indexes property.
      @ In, var, str, optional, name of variable (or None, or 'input', or 'output')
      @ Out, dims, dict, {name:values} of independent dimensions
    """
    # TODO add unit tests
    # TODO allow several variables requested at once?
    if var is None:
      var = self.getVars()
    elif var in ['input','output']:
      var = self.getVars(var)
    else:
      var = [var]
    dims = dict((v,list(key for key in self._pivotParams.keys() if v in self._pivotParams[key])) for v in var)
    return dims

  def getMeta(self,keys=None,pointwise=False,general=False):
    """
      Method to obtain entries in the metadata.  If neither pointwise nor general, then returns an empty dict.
       @ In, keys, list(str), optional, the keys (or main tag) to search for.  If None, return all.
       @ In, pointwise, bool, optional, if True then matches will be searched in the pointwise metadata
       @ In, general, bool, optional, if True then matches will be searched in the general metadata
       @ Out, meta, dict, key variables/xpaths to data object entries (column if pointwise, XML if general)
    """
    # if keys is None, keys is all of them
    if keys is None:
      keys = []
      if pointwise:
        keys += self._metavars
      if general:
        keys += self._meta.keys()
    gKeys = set([]) if not general else set(self._meta.keys()).intersection(set(keys))
    pKeys = set([]) if not pointwise else set(self._metavars).intersection(set(keys))
    # get any left overs
    missing = set(keys).difference(gKeys.union(pKeys))
    if len(missing)>0:
      self.raiseAnError(KeyError,'Some requested keys could not be found in the requested metadata:',missing)
    meta = {}
    if pointwise:
      # TODO slow key crawl
      for var in self._metavars:
        if var in pKeys:
          # TODO if still collecting, an option to NOT call asDataset
          meta[var] = self.asDataset()[var]
    if general:
      meta.update(dict((key,self._meta[key]) for key in gKeys))
    return meta

  def getVars(self,subset=None):
    """
      Gives list of variables that are part of this dataset.
      @ In, subset, str, optional, if given can return 'input','output','meta' subset types
      @ Out, getVars, list(str), list of variable names requested
    """
    if subset is None:
      return self.vars
    subset = subset.strip().lower()
    if subset == 'input':
      return self._inputs[:]
    elif subset == 'output':
      return self._outputs[:]
    elif subset == 'meta':
      return self._metavars[:]
    elif subset == 'indexes':
      return self.indexes[:]
    else:
      self.raiseAnError(KeyError,'Unrecognized subset choice: "{}"'.format(subset))

  def getVarValues(self,var):
    """
      Returns the sampled values of "var"
      @ In, var, str or list(str), name(s) of variable(s)
      @ Out, res, xr.DataArray, samples (or dict of {var:xr.DataArray} if multiple variables requested)
    """
    ## NOTE TO DEVELOPER:
    # This method will make a COPY of all the data into dictionaries.
    # This is necessarily fairly cumbersome and slow.
    # For faster access, consider using data.asDataset()['varName'] for one variable, or
    #                                   data.asDataset()[ ('var1','var2','var3') ] for multiple.
    self.asDataset()
    if mathUtils.isAString(var):
      val = self._data[var]
      #format as scalar
      if len(val.dims) == 0:
        res = self._data[var].item(0)
      #format as dataarray
      else:
        res = self._data[var]
    elif isinstance(var,list):
      res = dict((v,self.getVarValues(v)) for v in var)
    else:
      self.raiseAnError(RuntimeError,'Unrecognized request type:',type(var))
    return res

  def load(self,dataIn,style='netCDF',**kwargs):
    """
      Reads this dataset from disk based on the format.
      @ In, dataIn, str, path and name of file to read
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to reading function
      @ Out, None
    """
    style = style.lower()
    # if fileToLoad in kwargs, then filename is actualle fileName/fileToLoad
    if 'fileToLoad' in kwargs.keys():
      dataIn = kwargs['fileToLoad'].getAbsFile()
    # load based on style for loading
    if style == 'netcdf':
      self._fromNetCDF(dataIn,**kwargs)
    elif style == 'csv':
      # make sure we don't include the "csv"
      if dataIn.endswith('.csv'):
        dataIn = dataIn[:-4]
      self._fromCSV(dataIn,**kwargs)
    elif style == 'dict':
      self._fromDict(dataIn,**kwargs)
    elif style == 'dataset':
      self._fromXarrayDataset(dataIn)
    # TODO dask
    else:
      self.raiseAnError(NotImplementedError,'Unrecognized read style: "{}"'.format(style))
    # after loading, set or reset scaling factors
    self._setScalingFactors()

  def realization(self,index=None,matchDict=None,tol=1e-15, unpackXArray=False):
    """
      Method to obtain a realization from the data, either by index or matching value.
      Either "index" or "matchDict" must be supplied.
      If matchDict and no match is found, will return (len(self),None) after the pattern of numpy, scipy
      @ In, index, int, optional, number of row to retrieve (by index, not be "sample")
      @ In, matchDict, dict, optional, {key:val} to search for matches
      @ In, tol, float, optional, tolerance to which match should be made
      @ In, unpackXArray, bool, optional, True if the coordinates of the xarray variables must be exposed in the dict (e.g. if P(t) => {P:ndarray, t:ndarray}) (valid only for dataset)
      @ Out, index, int, optional, index where found (or len(self) if not found), only returned if matchDict
      @ Out, rlz, dict, realization requested (None if not found)
    """
    # TODO convert input space to KD tree for faster searching -> XArray.DataArray has this built in?
    ## first, check that some direction was given, either an index or a match to find
    if (index is None and matchDict is None) or (index is not None and matchDict is not None):
      self.raiseAnError(TypeError,'Either "index" OR "matchDict" (not both) must be specified to use "realization!"')
    numInData = len(self._data[self.sampleTag]) if self._data is not None else 0
    numInCollector = len(self._collector) if self._collector is not None else 0
    ## next, depends on if we're doing an index lookup or a realization match
    if index is not None:
      # traditional request: nonnegative integers
      if index >= 0:
        ## if index past the data, try the collector
        if index > numInData-1:
          ## if past the data AND the collector, we don't have that entry
          if index > numInData + numInCollector - 1:
            self.raiseAnError(IndexError,'{}: Requested index "{}" but only have {} entries (zero-indexed)!'.format(self.name,index,numInData+numInCollector))
          ## otherwise, take from the collector
          else:
            rlz = self._getRealizationFromCollectorByIndex(index - numInData)
        ## otherwise, take from the data
        else:
          rlz = self._getRealizationFromDataByIndex(index, unpackXArray)
      # handle "-" requests (counting from the end): first end of collector, or if not then end of data
      else:
        # caller is requesting so many "from the end", so work backwards
        ## if going back further than what's in the collector ...
        if abs(index) > numInCollector:
          ## if further going back further than what's in the data, then we don't have that entry
          if abs(index) > numInData + numInCollector:
            self.raiseAnError(IndexError,'Requested index "{}" but only have {} entries!'.format(index,numInData+numInCollector))
          ## otherwise, grab the requested index from the data
          else:
            rlz = self._getRealizationFromDataByIndex(index + numInCollector, unpackXArray)
        ## otherwise, grab the entry from the collector
        else:
          rlz = self._getRealizationFromCollectorByIndex(index)
      return rlz
    ## END select by index
    ## START collect by matching realization
    else: # matchDict must not be None
      # if nothing in data, try collector
      if numInData == 0:
        # if nothing in data OR collector, we can't have a match
        if numInCollector == 0:
          return 0,None
        # otherwise, get it from the collector
        else:
          index,rlz = self._getRealizationFromCollectorByValue(matchDict,tol=tol)
      # otherwise, first try to find it in the data
      else:
        index,rlz = self._getRealizationFromDataByValue(matchDict,tol=tol, unpackXArray=unpackXArray)
        # if no match found in data, try in the collector (if there's anything in it)
        if rlz is None:
          if numInCollector > 0:
            index,rlz = self._getRealizationFromCollectorByValue(matchDict,tol=tol)
      return index,rlz

  def remove(self,variable):
    """
      Used to remove either a realization or a variable from this data object.
      @ In, variable, str, name of "column" to remove
      @ Out, None
    """
    if self.size == 0:
      self.raiseAWarning('Called "remove" on DataObject, but it is empty!')
      return
    noData = self._data is None or len(self._data) == 0
    noColl = self._collector is None or len(self._collector) == 0
    # remove from self._data
    if not noData:
      self._data = self._data.drop(variable)
    # remove from self._collector
    if not noColl:
      varIndex = self._orderedVars.index(variable)
      self._collector.removeEntity(varIndex)
    # remove references to variable in lists
    self._orderedVars.remove(variable)
    # TODO potentially slow lookups
    for varlist in [self._inputs,self._outputs,self._metavars]:
      if variable in varlist:
        varlist.remove(variable)
    # remove from pivotParams, and remove any indexes without keys
    for pivot in self.indexes:
      if variable in self._pivotParams[pivot]:
        self._pivotParams[pivot].remove(variable)
      if len(self._pivotParams[pivot]) == 0:
        del self._pivotParams[pivot]
        # if in self._data, clear the index
        if not noData and pivot in self._data.dims:
          del self._data[pivot]
        # if in aligned indexes, remove it there as well
        if pivot in self._alignedIndexes:
          del self._alignedIndexes[pivot]
    # TODO remove references from general metadata?

    if self._scaleFactors is not None:
      self._scaleFactors.pop(variable,None)
    #either way reset kdtree
    self.inputKDTree = None

  def renameVariable(self,old,new):
    """
      Changes the name of a variable from "old" to "new".
      @ In, old, str, old name
      @ In, new, str, new name
      @ Out, None
    """
    # determine where the old variable was
    isInput = old in self._inputs
    isOutput = old in self._outputs
    isMeta = old in self._metavars
    isIndex = old in self.indexes
    # make the changes to the variable listings
    if isInput:
      self._inputs = list(a if (a != old) else new for a in self._inputs)
    if isOutput:
      self._outputs = list(a if (a != old) else new for a in self._outputs)
    if isMeta:
      self._metavars = list(a if (a != old) else new for a in self._metavars)
    if isIndex:
      # change the pivotParameters listing, as well as the sync/unsynced listings
      self._pivotParams[new] = self._pivotParams.pop(old)
      if old in self._alignedIndexes.keys():
        self._alignedIndexes[new] = self._alignedIndexes.pop(old)
      else:
        self._orderedVars = list(a if a != old else new for a in self._orderedVars)
    # if in/out/meta, change allvars (TODO wastefully already done if an unaligned index)
    if isInput or isOutput or isMeta:
      self._orderedVars = list(a if a != old else new for a in self._orderedVars)
    # change scaling factor entry
    if old in self._scaleFactors:
      self._scaleFactors[new] = self._scaleFactors.pop(old)
    if self._data is not None:
      self._data.rename({old:new},inplace=True)

  def reset(self):
    """
      Sets this object back to its initial state, keeping only the lists of the variables but removing
      all of the variable values.
      @ In, None
      @ Out, None
    """
    self._data = None
    self._collector = None
    self._meta = {}
    self._alignedIndexes = {}
    self._scaleFactors = {}

  def sliceByIndex(self,index):
    """
      Returns list of realizations at "snapshots" along dimension "index".
      For example, if 'index' is 'time', then returns cross-sectional slices of the dataobject at each recorded 'time' index value.
      @ In, index, str, name of index along which to obtain slices
      @ Out, slices, list, list of xr.Dataset slices.
    """
    data = self.asDataset()
    # if empty, nothing to do
    if self._data is None or len(self._data) == 0:
      self.raiseAWarning('Tried to return sliced data, but DataObject is empty!')
      return []
    # assert that index is indeed an index
    if index not in self.indexes + [self.sampleTag]:
      self.raiseAnError(IOError,'Requested slices along "{}" but that variable is not an index!  Options are: {}'.format(index,self.indexes))
    numIndexCoords = len(data[index])
    slices = list(data.isel(**{index:i}) for i in range(numIndexCoords))
    # NOTE: The slice may include NaN if a variable does not have a value along a different index for this snapshot along "index"
    return slices

  def write(self,fileName,style='netCDF',**kwargs):
    """
      Writes this dataset to disk based on the format.
      @ In, fileName, str, path and name of file to write
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to writing function
          Includes:  firstIndex, int, optional, if included then is the realization index that writing should start from (implies appending instead of rewriting)
      @ Out, index, int, index of latest rlz to be written, for tracking purposes
    """
    self.asDataset() #just in case there is stuff left in the collector
    if style.lower() == 'netcdf':
      self._toNetCDF(fileName,**kwargs)
    elif style.lower() == 'csv':
      if len(self.asDataset().variables)==0: #TODO what if it's just metadata?
        self.raiseAWarning('Nothing to write!')
        return
      #first write the CSV
      firstIndex = kwargs.get('firstIndex',0)
      self._toCSV(fileName,start=firstIndex,**kwargs)
      # then the metaxml
      if 'DataSet' in self._meta.keys():
        self._toCSVXML(fileName,**kwargs)
    # TODO dask?
    else:
      self.raiseAnError(NotImplementedError,'Unrecognized write style: "{}"'.format(style))
    if not self.hierarchical and 'RAVEN_isEnding' in self.getVars():
      return len(self._data.where(self._data['RAVEN_isEnding']==True,drop=True)['RAVEN_isEnding'])
    else:
      return len(self) # so that other entities can track which realization we've written

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
    s += len(self._data[self.sampleTag]) if self._data is not None else 0
    return s

  @property
  def indexes(self):
    """
      Property to access the independent axes in this problem
      @ In, None
      @ Out, indexes, list(str), independent index names (e.g. ['time'])
    """
    return list(self._pivotParams.keys())

  ### INTERNAL USE FUNCTIONS ###
  def _changeVariableValue(self,index,var,value):
    """
      Changes the value of a variable for a particular realization in the data object, in collector or data.
      Should only rarely be called!  Adding or removing data is recommended.
      For now, only works for scalar variables.
      @ In, index, int, index of realization to be modified
      @ In, var, str, name of variable to change
      @ In, value, float or int or str, new value for entry
      @ Out, None
    """
    assert(var in self._orderedVars)
    assert(mathUtils.isSingleValued(value)) #['float','str','int','unicode','bool'])
    lenColl = len(self._collector) if self._collector is not None else 0
    lenData = len(self._data[self.sampleTag]) if self._data      is not None else 0
    # if it's in the data ...
    if index < lenData:
      self._data[var].values[index] = value
    # if it's in the collector ...
    elif index < lenColl + lenData:
      self._collector[index][self._orderedVars.index(var)] = value
    else:
      self.raiseAnError(IndexError,'Requested value change for realization "{}", which is past the end of the data object!'.format(index))

  def _checkAlignedIndexes(self,rlz,tol=1e-15):
    """
      Checks to see if indexes should be stored as "aligned" or if they need to be stored distinctly.
      If store distinctly for the first time, adds a variable to the collector columns instead of storing it as
      an aligned index.
      @ In, rlz, dict, formatted realization with either singular or np arrays as values
      @ In, tol, float, optional, matching tolerance
      @ Out, None
    """
    for index in self.indexes:
      # if it's aligned so far, check if it still is
      if index in self._alignedIndexes.keys():
        # first, if lengths don't match, they're not aligned.
        # TODO there are concerns this check may slow down runs; it should be profiled along with other bottlenecks to optimize our efforts.
        if len(rlz[index]) != len(self._alignedIndexes[index]):
          closeEnough = False
        else:
          # "close enough" if float/int, otherwise require exactness
          if mathUtils.isAFloatOrInt(rlz[index][0]):
            closeEnough = all(np.isclose(rlz[index],self._alignedIndexes[index],rtol=tol))
          else:
            closeEnough = all(rlz[index] == self._alignedIndexes[index])
        # if close enough, then keep the aligned values; otherwise, take action
        if not closeEnough:
          dtype = rlz[index].dtype
          # TODO add new column to collector, propagate values up to (not incl) current rlz
          self.raiseAWarning('A mismatch in time scales has been found between consecutive realizations.  Consider synchronizing before doing any postprocessing!')
          # TODO if self._data is not none!
          if self._collector is not None:
            aligned = self._alignedIndexes.pop(index)
            values = [aligned] * len(self._collector)
            self._collector.addEntity(values)
            self._orderedVars.append(index)
        # otherwise, they are close enough, so no action needs to be taken
      # if index is not among the aligned, check if it is already in the collector/data
      else:
        # if we don't have any samples in the collector, congratulations, you're aligned with yourself
        if self._collector is None or len(self._collector) == 0:
          try:
            self._alignedIndexes[index] = rlz.pop(index)
          except KeyError:
            # it's already gone; this can happen if this pivot parameter is only being used to collapse data (like in PointSet case)
            pass
        # otherwise, you're misaligned, and have been since before this realization, no action.
    return

  def _checkRealizationFormat(self,rlz):
    """
      Checks that a passed-in realization has a format acceptable to data objects.
      Data objects require a CSV-like result with either float or np.ndarray instances.
      @ In, rlz, dict, realization with {key:value} pairs.
      @ Out, okay, bool, True if acceptable or False if not
    """
    if not isinstance(rlz,dict):
      self.raiseAWarning('Realization is not a "dict" instance!')
      return False
    for key,value in rlz.items():
      #if not isinstance(value,(float,int,unicode,str,np.ndarray)): TODO someday be more flexible with entries?
      if not isinstance(value,np.ndarray):
        self.raiseAWarning('Variable "{}" is not an acceptable type: "{}"'.format(key,type(value)))
        return False
      # check if index-dependent variables have matching shapes
      # FIXME: this check will not work in case of variables depending on multiple indexes.
      #    When this need comes, we will change this check(alfoa)
      if self.indexes:
        if key in self._fromVarToIndex and rlz[self._fromVarToIndex[key]].shape != rlz[key].shape:
          self.raiseAWarning(('Variable "{}" with shape {} '+
                              'is not consistent with respect its index "{}" with shape {}!')
                              .format(key,
                                      rlz[key].shape,
                                      self._fromVarToIndex[key],
                                      rlz[self._fromVarToIndex[key]].shape)
                              )
          return False
    # all conditions for failing formatting were not met, so formatting is fine
    return True

  def _clearAlignment(self):
    """
      Clears the alignment tracking for the collector, and removes columns from it if necessary
      @ In, None
      @ Out, None
    """
    # get list of indexes that need to be removed since we're starting over with alignment
    toRemove = list(self._orderedVars.index(var) for var in self.indexes if (var not in self._alignedIndexes.keys()
                                                                             and var in self._orderedVars))
    # sort them in reverse order so we don't screw up indexing while removing
    toRemove.sort(reverse=True)
    for index in toRemove:
      self._orderedVars.pop(index)
      self._collector.removeEntity(index)
    self._alignedIndexes = {}

  def _clearParentEndingStatus(self,rlz):
    """
      If self is hierarchical, then set the parent of the given realization "rlz" to False.
      @ In, rlz, dict, realization (from addRealization, already formatted)
      @ Out, None
    """
    # TODO set global status of 'parentID' instead of check every time
    idVar = 'RAVEN_parentID'
    endVar = 'RAVEN_isEnding'
    if idVar in self.getVars():
      # get the parent ID
      parentID = rlz[idVar]
      # if root or parentless, nothing to do
      if parentID == "None":
        return
      # otherwise, find the index of the match
      idx,match = self.realization(matchDict={'prefix':parentID})
      self._changeVariableValue(idx,endVar,False)

  def _collapseNDtoDataArray(self,data,var,labels=None,dtype=None):
    """
      Converts a row of numpy samples (float or xr.DataArray) into a single DataArray suitable for a xr.Dataset.
      @ In, data, np.ndarray, array of either float or xr.DataArray; array must be single-dimension
      @ In, var, str, name of the variable being acted on
      @ In, labels, list, optional, list of labels to use for collapsed array under self.sampleTag title
      @ In, dtype, type, optional, type from _getCompatibleType to cast data as
      @ Out, DataArray, xr.DataArray, single dataarray object
    """
    assert(isinstance(data,np.ndarray))
    assert(len(data.shape) == 1)
    # set up sampleTag values
    if labels is None:
      labels = range(len(data))
    else:
      assert(len(labels) == len(data))
    # find first non-None entry, and get its type if needed
    dataType = type(None)
    i = -1
    while dataType is type(None):
      i += 1
      dataType = type(data[i])
    # if "type" predetermined, override it (but we still needed "i" so always do the loop above)
    # TODO this can be sped up probably, by checking the "type" directly with dtype; but we ALSO need to know if
    #   it's a history or not, so we need to check the first non-NaN entry....
    if dtype is not None:
      dataType = dtype
    # method = 'once' # see below, parallelization is possible but not implemented
    # first case: single entry per node: floats, strings, ints, etc
    if mathUtils.isSingleValued(data[i]):
      data = np.array(data,dtype=dataType)
      array = xr.DataArray(data,
                           dims=[self.sampleTag],
                           coords={self.sampleTag:labels},
                           name=var) # THIS is very fast
    # second case: ND set (history set or higher dimension) --> CURRENTLY should be unused
    elif type(data[i]) == xr.DataArray:
      # two methods: all at "once" or "split" into multiple parts.  "once" is faster, but not parallelizable.
      # ONCE #
      #if method == 'once':
      val = dict((i,data[i]) for i in range(len(data)))
      val = xr.Dataset(data_vars=val)
      val = val.to_array(dim=self.sampleTag) # TODO labels preserved?
      val.coords[self.sampleTag] = labels
      # SPLIT # currently unused, but could be for parallel performance
      #elif method == 'split':
      #  chunk = 150
      #  start = 0
      #  N = len(data)
      #  vals = []
      #  # TODO can be parallelized
      #  while start < N-1:
      #    stop = min(start+chunk+1,N)
      #    ival = dict((i,data[i,v]) for i in range(start,stop))
      #    ival = xr.Dataset(data_vars=ival)
      #    ival = ival.to_array(dim=self.sampleTag) # TODO does this end up indexed correctly?
      #    vals.append(ival)
      #    start = stop
      #  val = xr.concat(vals,dim=self.sampleTag)
      # END #
      array = val
    else:
      self.raiseAnError(TypeError,'Unrecognized data type for var "{}": "{}"'.format(var,type(data[0])))
    array.rename(var)
    return array

  def _convertArrayListToDataset(self,array,action='return'):
    """
      Converts a 1-D array of xr.DataArrays into a xr.Dataset, then takes action on self._data:
      action=='replace': replace self._data with the new dataset
      action=='extend' : add new dataset to self._data using merge
      action=='return' : (default) return new dataset
      @ In, array, list(xr.DataArray), list of variables as samples to turn into dataset
      @ In, action, str, optional, can be used to specify the action to take with the new dataset
      @ Out, new, xr.Dataset, single data entity
    """
    try:
      new = xr.Dataset(array)
    except ValueError as e:
      self.raiseAnError(RuntimeError,'While trying to create a new Dataset, a variable has itself as an index!'+\
                        '  Error: ' +str(e))
    # if "action" is "extend" but self._data is None, then we really want to "replace".
    if action == 'extend' and self._data is None:
      action = 'replace'
    if action == 'return':
      return new
    elif action == 'replace':
      self._data = new
      # general metadata included if first time
      self._data.attrs = self._meta # appears to NOT be a reference
      # determine dimensions for each variable
      dimsMeta = {}
      # TODO potentially slow loop
      for var in self._inputs + self._outputs:
        dims = list(new[var].dims)
        # don't list if only entry is sampleTag
        if dims == [self.sampleTag]:
          continue
        # even then, don't list sampleTag
        try:
          dims.remove(self.sampleTag)
        except ValueError:
          pass #not there, so didn't need to remove
        dimsMeta[var] = ','.join(dims)
      # store sample tag, IO information, coordinates
      self.addMeta('DataSet',{'general':{'sampleTag':self.sampleTag,
                                         'inputs':','.join(self._inputs),
                                         'outputs':','.join(self._outputs),
                                         'pointwise_meta':','.join(self._metavars),
                                         },
                              'dims':dimsMeta,
                             })
    elif action == 'extend':
      # TODO compatability check!
      # TODO Metadata update?
      # merge can change dtypes b/c no NaN int type: self._data.merge(new,inplace=True)
      self._data = xr.concat([self._data,new],dim=self.sampleTag)
    else:
      self.raiseAnError(RuntimeError,'action "{}" was not an expected value for converting array list to dataset!'
                                      .format(action))
    # regardless if "replace" or "return", set up scaling factors
    self._setScalingFactors()
    return new

  def _convertFinalizedDataRealizationToDict(self,rlz, unpackXarray=False):
    """
      After collapsing into xr.Dataset, all entries are stored as xr.DataArrays.
      This converts them into a dictionary like the realization sent in.
      @ In, rlz, dict(varname:xr.DataArray), "row" from self._data
      @ In, unpackXarray, bool, unpack XArray coordinates in numpy arrays (it assumes that the coordinates are consistent among the data)
      @ Out, new, dict(varname:value), where "value" could be singular (float,str) or xr.DataArray
    """
    # TODO this has a lot of looping and might be slow for many variables.  Bypass or rewrite where possible.
    new = {}
    for k,v in rlz.items():
      # if singular, eliminate dataarray container
      if len(v.dims)==0:
        new[k] = v.item(0)
      # otherwise, trim NaN entries before returning
      else:
        for dim in v.dims:
          v = v.dropna(dim)
          if unpackXarray:
            new[dim] = v.coords[dim].values
        new[k] = v if not unpackXarray else v.values
    return new

  def _convertToDict(self):
    """
      Casts this dataObject as dictionary.
      @ In, None
      @ Out, asDataset, xr.Dataset or dict, data in requested format
    """
    self.raiseAWarning('DataObject._convertToDict can be a slow operation and should be avoided where possible!')
    # container for all necessary information
    dataDict = {}
    # supporting data
    dataDict['dims']     = self.getDimensions()
    dataDict['metadata'] = self.getMeta(general=True)
    # main data
    if self.type == "PointSet":
      ## initialize with np arrays of objects
      dataDict['data'] = dict((var,np.zeros(self.size,dtype=object)) for var in self.vars)
      for var in self.vars:
        dataDict['data'][var] = self.asDataset()[var].values
    else:
      dataDict['data'] = dict((var,np.zeros(self.size,dtype=object)) for var in self.vars+self.indexes)
      # need to remove NaNs, so loop over slices
      for s,rlz in enumerate(self.sliceByIndex(self.sampleTag)):
        ## loop over realizations to get distinct values without NaNs
        for var in self.vars:
          # how we get and store variables depends on the dimensionality of the variable
          dims=self.getDimensions(var)[var]
          # if scalar (no dims and not an index), just grab the values
          if len(dims)==0 and var not in self.indexes:
            dataDict['data'][var] = self.asDataset()[var].values
            continue
          # get data specific to this var for this realization (slice)
          data = rlz[var]
          # need to drop indexes for which no values are present
          for index in dims:
            data = data.dropna(index)
            dataDict['data'][index][s] = data[index].values
          dataDict['data'][var][s] = data.values
    return dataDict

  def _convertToXrDataset(self):
    """
      Casts this dataobject as an xr.Dataset and returns a REFERENCE to the underlying data structure.
      Functionally, typically collects the data from self._collector and places it in self._data.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, xarray.Dataset, all the data from this data object.
    """
    # TODO make into a protected method? Should it be called from outside?
    # if we have collected data, collapse it
    if self._collector is not None and len(self._collector) > 0:
      # keep track of the first sampling index, if we already have some samples (otherwise 0)
      firstSample = int(self._data[self.sampleTag][-1])+1 if self._data is not None else 0
      # storage array for each variable's xr.DataArray with all rlz data from every rlz
      arrays = {}
      # loop over variables IN ORDER of collector storage to collapse data into nice xr.DataArray of realization data
      for v,var in enumerate(self._orderedVars):
        # only converting variables, so ignore indexes (they'll be used by the variables)
        if var in self.indexes:
          continue
        # gather the data type from first realization: if np.array, it's ND; otherwise singular
        dtype = self.types[v]
        if isinstance(self._collector[0,v],np.ndarray):
          # for each index, determine if all aligned; make data arrays as required
          dims = self.getDimensions(var)[var]
          # make sure "dims" isn't polluted
          assert(self.sampleTag not in dims)
          # TODO not ready for ND; this only uses single-dependency cases, but should be easily extensible
          if len(dims) > 1:
            self.raiseAnError(NotImplementedError,'Currently cannot handle more than 1 pivot per variable')
          # loop over indexes (just one for now) and create data
          for index in dims:
            # if aligned, grab the data into one large chunk and make a datarray with all rlzs
            if index in self._alignedIndexes.keys():
              data = np.vstack(self._collector[:,v]).astype(dtype)
              coords = dict((idx,self._alignedIndexes[idx]) for idx in dims)
              arrays[var] = self.constructNDSample(data,dims=[self.sampleTag]+dims,coords=coords)
            # otherwise, we're better off making one dataarray for each rlz, then collapsing
            else:
              # first make a datarray out of each realization value
              for r in range(len(self._collector)):
                values = self._collector[r,v]
                dtype = self._getCompatibleType(values[0])
                values = np.array(values,dtype=dtype)
                coords = dict((idx,self._collector[r,self._orderedVars.index(idx)]) for idx in dims)
                self._collector[r][v] = self.constructNDSample(values,dims,coords,name=str(r))
              # then collapse these entries into a single datarray
              arrays[var] = self._collapseNDtoDataArray(self._collector[:,v],var,dtype=dtype)
        # if it's a dataarray, then that's old-style histories, no-can do right now
        elif isinstance(self._collector[0,v],xr.DataArray):
          self.raiseAnError(NotImplementedError,'History entries should be numpy arrays, not data arrays!')
        # if not ND, then it's a simple data array construction
        else:
          try:
            varData = np.array(self._collector[:,v],dtype=dtype)
          except ValueError as e:
            # infinte/missing data can't be cast to anything but floats or objects, as far as I can tell
            if dtype != float and pd.isnull(self._collector[:,v]).sum() != 0:
              self.raiseAWarning('NaN detected, but no safe casting NaN to "{}" so switching to "object" type. '.format(dtype) \
                  + ' This may cause problems with other entities in RAVEN.')
              varData = self._collector[:,v][:]
              dtype=object
            # otherwise, let error be raised.
            else:
              raise e
          # create single dataarrays
          arrays[var] = self._collapseNDtoDataArray(varData,var,dtype=dtype)
        # END if for variable data type (ndarray, xarray, or scalar)
        # re-index samples
        arrays[var][self.sampleTag] += firstSample
      # collect all data into dataset, and update self._data
      self._convertArrayListToDataset(arrays,action='extend')
      # reset collector
      self._collector = self._newCollector(width=self._collector.width)
      # write hierarchal data to general meta, if any
      paths = self._generateHierPaths()
      for p in paths:
        self.addMeta('DataSet',{'Hierarchical':{'path':','.join(p)}})
      # clear alignment tracking for indexes
      self._clearAlignment()
    return self._data

  def _formatRealization(self,rlz):
    """
      Formats realization without truncating data
      Namely, assures indexes are correctly typed and length-1 variable arrays become floats
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    # TODO this could be much more efficient on the parallel (finalizeCodeOutput) than on serial
    # TODO costly for loop
    # do indexes first to assure correct typing on first realization collection
    # - note, the other variables are set in _setDataTypes which is called after _formatRealization in addRealization
    if self._collector is None or len(self._collector) == 0:
      for var in self._pivotParams.keys():
        dtype = self._getCompatibleType(rlz[var][0])
        current = rlz[var].dtype
        # Note, I don't like this action happening here, but I don't have an alternative way to assure
        # indexes have the correct dtype.  In the first pass, they aren't going into the collector, but into alignedIndexes.
        rlz[var] = np.array(rlz[var],dtype=dtype)
    # for now, leave them as the arrays they are, except single entries need converting
    for var,val in rlz.items():
      # if an index variable, skip it
      if var in self._pivotParams:
        continue
      dims = self.getDimensions(var)[var]
      ## change dimensionless to floats -> TODO use operator to collapse!
      if dims in [[self.sampleTag], []]:
        if len(val) == 1:
          rlz[var] = val[0]
    return rlz

  def _fromCSV(self,fileName,**kwargs):
    """
      Loads a dataset from CSV (preferably one it wrote itself, but maybe not necessarily?
      @ In, fileName, str, filename to load from (not including .csv or .xml)
      @ In, kwargs, dict, optional arguments
      @ Out, None
    """
    # first, try to read from csv
    df = self._readPandasCSV(fileName+'.csv')
    # load in metadata
    dims = self._loadCsvMeta(fileName)
    # find distinct number of samples
    try:
      samples = list(set(df[self.sampleTag]))
    except KeyError:
      # sample ID wasn't given, so assume each row is sample
      samples = range(len(df.index))
      df[self.sampleTag] = samples
    # create arrays from which to create the data set
    arrays = {}
    for var in self.getVars():
      if var in dims.keys():
        data = df[[var,self.sampleTag]+dims[var]]
        data.set_index(self.sampleTag,inplace=True)
        ndat = np.zeros(len(samples),dtype=object)
        for s,sample in enumerate(samples):
          # set dtype on first pass
          if s == 0:
            dtype = self._getCompatibleType(sample)
          places = data.index.get_loc(sample)
          vals = data[places].dropna().set_index(dims[var])
          #vals.drop('dim_1')
          # TODO this needs to be improved before ND will work; we need the individual sub-indices (time, space, etc)
          ndat[s] = xr.DataArray(vals.values[:,0],
                                 dims=dims[var],
                                 coords=dict((var,vals.index.values) for var in dims[var]))
        # END for sample in samples
        arrays[var] = self._collapseNDtoDataArray(ndat,var,labels=samples,dtype=dtype)
      else:
        # scalar example
        data = df[[var,self.sampleTag]].groupby(self.sampleTag).first().values[:,0]
        dtype = self._getCompatibleType(data.item(0))
        arrays[var] = self._collapseNDtoDataArray(data,var,labels=samples,dtype=dtype)
    self._convertArrayListToDataset(arrays,action='extend')

  def _fromCSVXML(self,fileName):
    """
      Loads in the XML portion of a CSV if it exists.  Returns information found.
      @ In, fileName, str, filename to read as filename.xml
      @ Out, metadata, dict, metadata discovered
    """
    metadata = {}
    # check if we have anything from which to read
    try:
      meta,_ = xmlUtils.loadToTree(fileName+'.xml')
      self.raiseADebug('Reading metadata from "{}.xml"'.format(fileName))
      haveMeta = True
    except IOError:
      haveMeta = False
    # if nothing to load, return nothing
    if not haveMeta:
      return metadata
    tagNode = xmlUtils.findPath(meta,'DataSet/general/sampleTag')
    # read samplerTag
    if tagNode is not None:
      metadata['sampleTag'] = tagNode.text
    # read dimensional data
    dimsNode = xmlUtils.findPath(meta,'DataSet/dims')
    if dimsNode is not None:
      metadata['pivotParams'] = dict((child.tag,child.text.split(',')) for child in dimsNode)
    inputsNode = xmlUtils.findPath(meta,'DataSet/general/inputs')
    if inputsNode is not None:
      metadata['inputs'] = inputsNode.text.split(',')
    outputsNode = xmlUtils.findPath(meta,'DataSet/general/outputs')
    if outputsNode is not None:
      metadata['outputs'] = outputsNode.text.split(',')
    # these DO have to be read from meta if present
    metavarsNode = xmlUtils.findPath(meta,'DataSet/general/pointwise_meta')
    if metavarsNode is not None:
      metadata['metavars'] = metavarsNode.text.split(',')
    # return
    return metadata

  def _fromDict(self,source,dims=None,**kwargs):
    """
      Loads data from a dictionary with variables as keys and values as np.arrays of realization values
      Format for entries in "source":
        - scalars: source['a'] = np.array([1, 2, 3, 4])  -> each entry is a realization
        - vectors: source['b'] = np.array([ np.array([1, 2]), np.array([3,4,5]) ])  -> each entry is a realization
        - indexes: same as "vectors"
      @ In, source, dict, as {var:values} with types {str:np.array}
      @ In, dims, dict, optional, ordered list of dimensions that each var depends on as {var:[list]}
      @ In, kwargs, dict, optional, additional arguments
      @ Out, None
    """
    # if anything is in the collector, collapse it first
    if self._collector is not None:
      self.asDataset()
    # not safe to default to dict, so if "dims" not specified set it here
    if dims is None:
      dims = {}
    # data sent in is as follows:
    #   single-entry (scalars) - np.array([val, val, val])
    #   histories              - np.array([np.array(vals), np.array(vals), np.array(vals)])
    #   etc
    ## check that all inputs, outputs required are provided
    providedVars = set(source.keys())
    requiredVars = set(self.getVars())
    ## figure out who's missing from the IO space
    missing = requiredVars - providedVars
    if len(missing) > 0:
      self.raiseAnError(KeyError,'Variables are missing from "source" that are required for data object "',
                                  self.name.strip(),'":',",".join(missing))
    # set orderedVars to all vars, for now don't be fancy with alignedIndexes
    self._orderedVars = self.vars + self.indexes
    # make a collector from scratch
    rows = len(source.values()[0])
    cols = len(self._orderedVars)
    # can this for-loop be done in a comprehension?  The dtype seems to be a bit of an issue.
    data = np.zeros([rows,cols],dtype=object)
    for v,var in enumerate(self._orderedVars):
      if len(source[var].shape) > 1:
        # we can't set all at once, because the user gave us an ND array instead of a np.array(dtype=object) of np.array.
        #    if we try -> ValueError: "could not broadcast input array from shape (#rlz,#time) into shape (#rlz)
        for i in range(len(data)):
          data[i,v] = source[var][i]
      else:
        # we can set it at once, the fast way.
        data[:,v] = source[var]
    # set up collector as cached nd array of values -> TODO might be some wasteful copying here
    self._collector = cached_ndarray.cNDarray(values=data,dtype=object)
    # set datatypes for each variable
    rlz = self.realization(index=0)
    self._setDataTypes(rlz)
    # collapse into xr.Dataset
    self.asDataset()

  def _fromNetCDF(self,fileName, **kwargs):
    """
      Reads this data object from file that is netCDF.  If not netCDF4, this could be slow.
      Loads data lazily; it won't be pulled into memory until operations are attempted on the specific data
      @ In, fileName, str, path/name to read file
      @ In, kwargs, dict, optional, keywords to pass to netCDF4 reading
                                    See http://xarray.pydata.org/en/stable/io.html#netcdf for options
      @ Out, None
    """
    # TODO set up to use dask for on-disk operations -> or is that a different data object?
    self._data = xr.open_dataset(fileName)
    # NOTE: open_dataset does NOT close the file object after loading (lazy loading)
    ## -> if you try to rm the file in Windows before closing, it will fail with WindowsError 32: file in use!
    # convert metadata back to XML files
    for key,val in self._data.attrs.items():
      self._meta[key] = pk.loads(val.encode('utf-8'))

  def _fromXarrayDataset(self,dataset):
    """
    """
    if not self.isEmpty:
      self.raiseAnError(IOError, 'DataObject', self.name.strip(),'is not empty!')
    #select data from dataset
    providedVars  = set(dataset.data_vars.keys())
    requiredVars  = set(self.getVars())
    ## figure out who's missing from the IO space
    missing = requiredVars - providedVars
    if len(missing) > 0:
      self.raiseAnError(KeyError,'Variables are missing from "source" that are required for data object "',
                                  self.name.strip(),'":',",".join(missing))
    # remove self.sampleTag since it is an internal used dimension
    providedDims = set(dataset.sizes.keys()) - set([self.sampleTag])
    requiredDims = set(self.indexes)
    missing = requiredDims - providedDims
    if len(missing) > 0:
      self.raiseAnError(KeyError,'Dimensions are missing from "source" that are required for data object "',
                                  self.name.strip(),'":',",".join(missing))
    # select the required data from given dataset
    datasetSub = dataset[list(requiredVars)]
    # check the dimensions
    for var in self.vars:
      requiredDims = set(self.getDimensions(var)[var])
      # make sure "dims" isn't polluted
      assert(self.sampleTag not in requiredDims)
      providedDims = set(datasetSub[var].sizes.keys()) - set([self.sampleTag])
      if requiredDims != providedDims:
        self.raiseAnError(KeyError,'Dimensions of variable',var,'from "source"', ",".join(providedDims),
                'is not consistent with the required dimensions for data object "',
                self.name.strip(),'":',",".join(requiredDims))
    self._orderedVars = self.vars + self.indexes
    self._data = datasetSub
    for key, val in self._data.attrs.items():
      self._meta[key] = val

  def _getCompatibleType(self,val):
    """
      Determines the data type for "val" that is compatible with the rest of the data object.
      @ In, val, object, item whose type should be determined.
      @ Out, _type, type instance, type to use
    """
    # ND uses first entry as example type
    if isinstance(val,(xr.DataArray,np.ndarray)):
      val = val.item(0)
    # identify other scalars by instance
    if mathUtils.isAFloat(val):
      _type = float
    elif mathUtils.isABoolean(val):
      _type = bool
    elif mathUtils.isAnInteger(val):
      _type = int
    # strings and unicode have to be stored as objects to prevent string sizing in numpy
    elif mathUtils.isAString(val):
      _type = object
    # catchall
    else:
      _type = object
    return _type

  def _getRealizationFromCollectorByIndex(self,index):
    """
      Obtains a realization from the collector storage using the provided index.
      @ In, index, int, index to return
      @ Out, rlz, dict, realization as {var:value}
    """
    assert(self._collector is not None)
    assert(index < len(self._collector))
    rlz = dict(zip(self._orderedVars,self._collector[index]))
    # don't forget the aligned indices! If indexes stored there instead of in collector, retrieve them
    for var,vals in self._alignedIndexes.items():
      rlz[var] = vals
    return rlz

  def _getRealizationFromCollectorByValue(self,toMatch,tol=1e-15):
    """
      Obtains a realization from the collector storage matching the provided index
      @ In, toMatch, dict, elements to match
      @ In, tol, float, optional, tolerance to which match should be made
      @ Out, r, int, index where match was found OR size of data if not found
      @ Out, rlz, dict, realization as {var:value} OR None if not found
    """
    assert(self._collector is not None)
    # TODO KD Tree for faster values -> still want in collector?
    # TODO slow double loop
    lookingFor = toMatch.values()
    for r,row in enumerate(self._collector[:,tuple(self._orderedVars.index(var) for var in toMatch.keys())]):
      match = True
      for e,element in enumerate(row):
        # check for matching based on if a number or not
        if mathUtils.isAFloatOrInt(element):
          match &= mathUtils.compareFloats(lookingFor[e],element,tol=tol)
        else:
          match &= lookingFor[e] == element
        # if this element doesn't match, the row doesn't match
        if not match:
          break
      # if each element matched, we found a match, so stop looking
      if match:
        break
    # did we find a match?
    if match:
      return r,self._getRealizationFromCollectorByIndex(r)
    else:
      return len(self),None

  def _getRealizationFromDataByIndex(self,index, unpackXArray=False):
    """
      Obtains a realization from the data storage using the provided index.
      @ In, index, int, index to return
      @ In, unpackXArray, bool, optional, True if the coordinates of the xarray variables must be exposed in the dict (e.g. if P(t) => {P:ndarray, t:ndarray})
      @ Out, rlz, dict, realization as {var:value} where value is a DataArray with only coordinate dimensions
    """
    assert(self._data is not None)
    rlz = self._data[{self.sampleTag:index}].drop(self.sampleTag).data_vars
    rlz = self._convertFinalizedDataRealizationToDict(rlz, unpackXArray)
    return rlz

  def _getRealizationFromDataByValue(self,match, tol=1e-15, unpackXArray=False):
    """
      Obtains a realization from the data storage using the provided index.
      @ In, match, dict, elements to match
      @ In, tol, float, optional, tolerance to which match should be made
      @ In, unpackXArray, bool, optional, True if the coordinates of the xarray variables must be exposed in the dict (e.g. if P(t) => {P:ndarray, t:ndarray})
      @ Out, r, int, index where match was found OR size of data if not found
      @ Out, rlz, dict, realization as {var:value} OR None if not found
    """
    assert(self._data is not None)
    # TODO this could be slow, should do KD tree instead
    mask = 1.0
    for var,val in match.items():
      # float instances are relative, others are absolute
      if mathUtils.isAFloatOrInt(val):
        # scale if we know how
        try:
          loc,scale = self._scaleFactors[var]
        #except TypeError:
        #  # self._scaleFactors is None, so set them
        #  self._setScalingFactors(var)
        except KeyError: # IndexError?
        # variable doesn't have a scale factor (yet? Why not?)
          loc = 0.0
          scale = 1.0
        if scale == 0:
          # TODO: Seem to me, we need to find a better way to compare data
          # The scale will be zero if Grid Sampler is used, reset to 1.0
          scale = 1.0
        scaleVal = (val-loc)/scale
        # create mask of where the dataarray matches the desired value
        mask *= abs((self._data[var]-loc)/scale - scaleVal) < tol
      else:
        mask *= self._data[var] == val
    rlz = self._data.where(mask,drop=True)
    try:
      idx = rlz[self.sampleTag].item(0)
    except IndexError:
      return len(self),None
    return idx,self._getRealizationFromDataByIndex(idx,unpackXArray)

  def _getRequestedElements(self,options):
    """
      Obtains a list of the elements to be written, based on defaults and options[what]
      @ In, options, dict, general list of options for writing output files
      @ Out, keep, list(str), list of variables that will be written to file
    """
    if 'what' in options.keys():
      elements = options['what'].split(',')
      keep = []
      for entry in elements:
        small = entry.strip().lower()
        if small == 'input':
          keep += self._inputs
          continue
        elif small == 'output':
          keep += self._outputs
          continue
        elif small == 'metadata':
          keep += self._metavars
          continue
        else:
          keep.append(entry.split('|')[-1].strip())
    else:
      # need the sampleTag meta to load histories
      # BY DEFAULT keep everything needed to reload this entity.  Inheritors can define _neededForReload to specify what that is.
      keep = set(self._inputs + self._outputs + self._metavars + self._neededForReload)
    return keep

  def _getVariableIndex(self,var):
    """
      Obtains the index in the list of variables for the requested var.
      @ In, var, str, variable name (input, output, or pointwise medatada)
      @ Out, index, int, column corresponding to the variable
    """
    return self._orderedVars.index(var)

  def _identifyVariablesInCSV(self,fileName):
    """
      Gets the list of available variables from the file "fileName.csv".  A method is necessary because HistorySets
      don't store all the data in one csv.
      @ In, fileName, str, name of file without extension
      @ Out, varList, list(str), list of variables
    """
    with open(fileName+'.csv','rU') as f:
      provided = list(s.strip() for s in f.readline().split(','))
    return provided

  def _loadCsvMeta(self,fileName):
    """
      Attempts to load metadata from an associated XML file.
      If found, update stateful parameters.
      If not available, check the CSV itself for the available variables.
      @ In, fileName, str, filename (without extension) of the CSV/XML combination
      @ Out, dims, dict, dimensionality dictionary with {index:[vars]} structure
    """
    meta = self._fromCSVXML(fileName)
    # if we have meta, use it to load data, as it will be efficient to read from
    if len(meta) > 0:
      # TODO shouldn't we be respecting user wishes more carefully? TODO
      self._samplerTag = meta.get('sampleTag',self.sampleTag)
      dims = meta.get('pivotParams',{})
      if len(dims)>0:
        self.setPivotParams(dims)
      # add metadata, so we get probability weights and etc
      self.addExpectedMeta(meta.get('metavars',[]))
      # check all variables desired are available
      provided = set(meta.get('inputs',[])+meta.get('outputs',[])+meta.get('metavars',[]))
    # otherwise, if we have no meta XML to load from, infer what we can from the CSV, which is only the available variables.
    else:
      provided = set(self._identifyVariablesInCSV(fileName))
      dims = {}
    # check provided match needed
    needed = set(self._orderedVars)
    missing = needed - provided
    if len(missing) > 0:
      self.raiseAnError(IOError,'Not all variables requested for data object "{}" were found in csv "{}.csv"! Missing: {}'.format(self.name,fileName,missing))
    # otherwise, return happily and continue loading the CSV
    return dims

  def _newCollector(self,width=1,length=100,dtype=None):
    """
      Creates a new collector object and returns it.
      @ In, width, int, optional, width of collector
      @ In, length, int, optional, initial length of (allocated) collector
      @ In, dtype, type, optional, type of entires (float if all float, usually should be object)
    """
    if dtype is None:
      dtype = self.defaultDtype # set in subclasses if different
    return cached_ndarray.cNDarray(width=width,length=length,dtype=dtype)

  def _readPandasCSV(self,fname,nullOK=None):
    """
      Reads in a CSV and does some simple checking.
      @ In, fname, str, name of file to read in (WITH the .csv extension)
      @ In, nullOK, bool, optional, if provided then determines whether to error on nulls or not
      @ Out, df, pd.DataFrame, contents of file
    """
    # if nullOK not provided, infer from type: points and histories can't have them
    if self.type in ['PointSet','HistorySet']:
      nullOK = False
    # datasets can have them because we don't have a 2d+ CSV storage strategy yet
    else:
      nullOK = True
    # first try reading the file
    try:
      df = pd.read_csv(fname)
    except pd.errors.EmptyDataError:
      # no data in file
      self.raiseAWarning('Tried to read data from "{}", but the file is empty!'.format(fname+'.csv'))
      return
    else:
      self.raiseADebug('Reading data from "{}.csv"'.format(fname))
    # check for NaN contents -> this isn't allowed in RAVEN currently, although we might need to change this for ND
    if (not nullOK) and (pd.isnull(df).values.sum() != 0):
      bad = pd.isnull(df).any(1).nonzero()[0][0]
      self.raiseAnError(IOError,'Invalid data in input file: row "{}" in "{}"'.format(bad+1,fname))
    return df

  def _resetScaling(self):
    """
      Removes the KDTree and scaling factors, usually because the data changed in some way
      @ In, None
      @ Out, None
    """
    self._scaleFactors = {}
    self._inputKDTree = None

  def _selectiveRealization(self,rlz):
    """
      Used for selecting a subset of the given data.  Not implemented for ND.
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    return rlz

  def _setDataTypes(self,rlz):
    """
      Set the data types according to the given realization.
      @ In, rlz, dict, standardized and formatted realization
      @ Out, None
    """
    if self.types is None:
      self.types = [None]*len(self.getVars())
      for v,name in enumerate(self.getVars()):
        val = rlz[name]
        self.types[v] = self._getCompatibleType(val)

  def _setScalingFactors(self,var=None):
    """
      Sets the scaling factors for the data (mean, scale).
      @ In, var, str, optional, if given then will only set factors for "var"
      @ Out, None
    """
    if var is None:
      # clear existing factors and set list to "all"
      self._scaleFactors = {}
      varList = self.getVars()
    else:
      # clear existing factor and reset variable scale, if existing
      varList = [var]
      try:
        del self._scaleFactors[var]
      except KeyError:
        pass
    # TODO someday make KDTree too!
    assert(self._data is not None) # TODO check against collector entries?
    for var in varList:
      ## commented code. We use a try now for speed. It probably needs to be modified for ND arrays
      # if not a float or int, don't scale it
      # TODO this check is pretty convoluted; there's probably a better way to figure out the type of the variable
      #first = self._data.groupby(var).first()[var].item(0)
      #if (not mathUtils.isAFloatOrInt(first)) or np.isnan(first):# or self._data[var].isnull().all():
      #  continue
      try:
        mean = float(self._data[var].mean())
        scale = float(self._data[var].std())
        self._scaleFactors[var] = (mean,scale)
      except Exception:
        self.raiseADebug('Had an issue with setting scaling factors for variable "{}". No big deal.'.format(var))
        pass

  def _toCSV(self,fileName,start=0,**kwargs):
    """
      Writes this data object to CSV file (except the general metadata, see _toCSVXML)
      @ In, fileName, str, path/name to write file
      @ In, start, int, optional, first realization to start printing from (if > 0, implies append mode)
      @ In, kwargs, dict, optional, keywords for options
            Possibly includes:
                'clusterLabel': name of variable to cluster printing by.  If included then triggers history-like printing.
      @ Out, None
    """
    filenameLocal = fileName # TODO path?
    keep = self._getRequestedElements(kwargs)
    toDrop = list(var for var in self.getVars() if var not in keep)
    # if printing by cluster, divert now
    if 'clusterLabel' in kwargs:
      clusterLabel = kwargs.pop('clusterLabel')
      self._toCSVCluster(fileName,start,clusterLabel,**kwargs)
      return
    # set up data to write
    if start > 0:
      # slice data starting from "start"
      sl = slice(start,None,None)
      data = self._data.isel(**{self.sampleTag:sl})
      mode = 'a'
    else:
      data = self._data
      mode = 'w'

    data = data.drop(toDrop)
    self.raiseADebug('Printing data to CSV: "{}"'.format(filenameLocal+'.csv'))
    # get the list of elements the user requested to write
    # order data according to user specs # TODO might be time-inefficient, allow user to skip?
    ordered = list(i for i in self._inputs if i in keep)
    ordered += list(o for o in self._outputs if o in keep)
    ordered += list(m for m in self._metavars if m in keep)
    self._usePandasWriteCSV(filenameLocal,data,ordered,keepSampleTag = self.sampleTag in keep,mode=mode)

  def _toCSVCluster(self,fileName,start,clusterLabel,**kwargs):
    """
      Writes this data object as a chain of CSVs, grouped by the cluster
      @ In, fileName, str, path/name to write file
      @ In, start, int, optional, TODO UNUSED first realization to start printing from (if > 0, implies append mode)
      @ In, clusterLable, str, variable by which to cluster printing
      @ In, kwargs, dict, optional, keywords for options
      @ Out, None
    """
    # get list of variables to print
    keep = self._getRequestedElements(kwargs)
    # get unique cluster labels
    clusterIDs = set(self._data[clusterLabel].values)
    # write main CSV pointing to other files
    with open(fileName+'.csv','w') as writeFile: # TODO append mode if printing each step
      writeFile.writelines('{},filename\n'.format(clusterLabel))
      for ID in clusterIDs:
        writeFile.writelines('{},{}_{}.csv\n'.format(ID,fileName,ID))
      self.raiseADebug('Wrote master cluster file to "{}.csv"'.format(fileName))
    # write sub files as point sets
    ordered = list(var for var in itertools.chain(self._inputs,self._outputs,self._metavars) if (var != clusterLabel and var in keep))
    for ID in clusterIDs:
      data = self._data.where(self._data[clusterLabel] == ID, drop = True).drop(clusterLabel)
      subName = '{}_{}'.format(fileName,ID)
      self._usePandasWriteCSV(subName, data, ordered, keepSampleTag=self.sampleTag in keep, mode='w') # TODO append mode
      self.raiseADebug('Wrote sub-cluster file to "{}.csv"'.format(subName))

  def _toCSVXML(self,fileName,**kwargs):
    """
      Writes the general metadata of this data object to XML file
      @ In, fileName, str, path/name to write file
      @ In, kwargs, dict, additional options
      @ Out, None
    """
    # make copy of XML and modify it
    meta = copy.deepcopy(self._meta)
    # remove variables that aren't being "kept" from the meta record
    keep = self._getRequestedElements(kwargs)
    ## remove from "dims"
    dimsNode = xmlUtils.findPath(meta['DataSet'].tree.getroot(),'dims')
    if dimsNode is not None:
      toRemove = []
      for child in dimsNode:
        if child.tag not in keep:
          toRemove.append(child)
      for r in toRemove:
        dimsNode.remove(r)
    ## remove from "inputs, outputs, pointwise"
    genNode =  xmlUtils.findPath(meta['DataSet'].tree.getroot(),'general')
    toRemove = []
    for child in genNode:
      if child.tag in ['inputs','outputs','pointwise_meta']:
        vs = []
        for var in child.text.split(','):
          if var.strip() in keep:
            vs.append(var)
        if len(vs) == 0:
          toRemove.append(child)
        else:
          child.text = ','.join(vs)
    for r in toRemove:
      genNode.remove(r)

    self.raiseADebug('Printing metadata XML: "{}"'.format(fileName+'.xml'))
    with open(fileName+'.xml','w') as ofile:
      #header
      ofile.writelines('<DataObjectMetadata name="{}">\n'.format(self.name))
      for name,target in meta.items():
        xml = target.writeFile(asString=True,startingTabs=1,addRavenNewlines=False)
        ofile.writelines('  '+xml+'\n')
      ofile.writelines('</DataObjectMetadata>\n')

  def _toNetCDF(self,fileName,**kwargs):
    """
      Writes this data object to file in netCDF4.
      @ In, fileName, str, path/name to write file
      @ In, kwargs, dict, optional, keywords to pass to netCDF4 writing
                                    One good option is format='NETCDF4' to assure netCDF4 is used
                                    See http://xarray.pydata.org/en/stable/io.html#netcdf for options
      @ Out, None
    """
    # TODO set up to use dask for on-disk operations -> or is that a different data object?
    # convert metadata into writeable
    self._data.attrs = dict((key,pk.dumps(val)) for key,val in self._meta.items())
    self._data.to_netcdf(fileName,**kwargs)

  def _usePandasWriteCSV(self,fileName,data,ordered,keepSampleTag=False,keepIndex=False,mode='w'):
    """
      Uses Pandas to write a CSV.
      @ In, fileName, str, path/name to write file
      @ In, data, xr.Dataset, data to write (with only "keep" vars included, plus self.sampleTag)
      @ In, ordered, list(str), ordered list of headers
      @ In, keepSampleTag, bool, optional, if True then keep the sampleTag in the CSV
      @ In, keepIndex, bool, optional, if True then keep indices in the CSV even if not multiindex
      @ In, mode, str, optional, mode to write CSV in (write, append as 'w','a')
      @ Out, None
    """
    # TODO asserts
    # make a pandas dataframe, they write to CSV very well
    data = data.to_dataframe()
    # order entries
    data = data[ordered]
    # set up writing mode; if append, don't write headers
    if mode == 'a':
      header = False
    else:
      header = True
    # write, depending on whether to keep sampleTag in index or not
    if keepSampleTag:
      data.to_csv(fileName+'.csv',mode=mode,header=header)
    else:
      # if other multiindices included, don't omit them #for ND DataSets only
      if isinstance(data.index,pd.MultiIndex):
        # if we have just the self.sampleTag index (we can not drop it otherwise pandas fail). We use index=False (a.a.)
        localIndex = True
        if len(data.index.names) == 1:
          localIndex = self.sampleTag not in data.index.names
        else:
          if self.sampleTag in data.index.names:
            data.index = data.index.droplevel(self.sampleTag)
        if not localIndex:
          data.to_csv(fileName+'.csv',mode=mode,header=header, index=localIndex)
        else:
          data.to_csv(fileName+'.csv',mode=mode,header=header)
          ## START garbled index fix ##
          ## At one point we were seeing "garbled" indexes printed from Pandas: a,b,(RAVEN_sample_ID,),c
          ## Here, commented is a workaround that @alfoa set up to prevent that problem.
          ## However, it is painfully slow, so if garbled data shows up again, we can
          ##   revisit this fix.
          ## When using this fix, comment out the data.to_csv line above.
          #dataString = data.to_string()
          # find headers
          #splitted = [",".join(elm.split())+"\n" for elm in data.to_string().split("\n")]
          #header, stringData = splitted[0:2], splitted[2:]
          #header.reverse()
          #toPrint = [",".join(header).replace("\n","")+"\n"]+stringData
          #with open(fileName+'.csv', mode='w+') as fileObject:
          #  fileObject.writelines(toPrint)
          ## END garbled index fix ##
      # if keepIndex, then print as is
      elif keepIndex:
        data.to_csv(fileName+'.csv',mode=mode,header=header)
      # if only index was sampleTag and we don't want it, index = False takes care of that
      else:
        data.to_csv(fileName+'.csv',index=False,mode=mode,header=header)
    #raw_input('Just wrote to CSV "{}.csv", press enter to continue ...'.format(fileName))

  # _useNumpyWriteCSV (below) is a secondary method to write out POINT SET CSVs.  When benchmarked with Pandas, I tested using
  # different numbers of variables (M=5,25,100) and different numbers of realizations (R=4,100,1000).
  # For each test, I did a unit check just on _usePandasWriteCSV versus _useNumpyWriteCSV, and took the average time
  # to run a trial over 1000 trials (in seconds).  The results are as follows:
  #    R     M   pandas     numpy       ratio    per float p  per float n  per float ratio
  #    4     5  0.001748  0.001004  1.741035857   0.00008740   0.00005020  1.741035857
  #    4    25  0.002855  0.001378  2.071843251   0.00002855   0.00001378  2.071843251
  #    4   100  0.007006  0.002633  2.660843145   0.00001752   6.5825E-06  2.660843145
  #  100    5   0.001982  0.001819  1.089609676   0.00000396   0.00000364  1.089609676
  #  100   25   0.003922  0.003898  1.006182658   1.5688E-06   1.5592E-06  1.006182658
  #  100  100   0.011124  0.011386  0.976989285   1.1124E-06   1.1386E-06  0.976989285
  # 1000    5   0.004108  0.008688  0.472859116   8.2164E-07   1.7376E-06  0.472859116
  # 1000   25   0.013367  0.027660  0.483261027   5.3468E-07   1.1064E-06  0.483261027
  # 1000  100   0.048791  0.095213  0.512442602   4.8791E-07   9.5213E-07  0.512442602
  # The per-float columns divide the time taken by (R*M) to give a fair comparison.  The summary of the # var versus # realizations per float is:
  #          ---------- R ----------------
  #   M      4             100        1000
  #   5  1.741035857  1.089609676  0.472859116
  #  25  2.071843251  1.006182658  0.483261027
  # 100  2.660843145  0.976989285  0.512442602
  # When the value is > 1, numpy is better (so when < 1, pandas is better).  It seems that "R" is a better
  # indicator of which method is better, and R < 100 is a fairly simple case that is pretty fast anyway,
  # so for now we just keep everything using Pandas. - talbpaul and alfoa, January 2018
  #
  #def _useNumpyWriteCSV(self,fileName,data,ordered,keepSampleTag=False,keepIndex=False,mode='w'):
  #  # TODO docstrings
  #  # TODO assert point set -> does not work right for ND (use Pandas)
  #  # TODO the "mode" should be changed for python 3: mode has to be 'ba' if appending, not 'a' when using numpy.savetxt
  #  with open(fileName+'.csv',mode) as outFile:
  #    if mode == 'w':
  #      #write header
  #      header = ','.join(ordered)
  #    else:
  #      header = ''
  #    data = data[ordered].to_array()
  #    if not keepSampleTag:
  #      data = data.drop(self.sampleTag)
  #    data = data.values.transpose()
  #    # set up formatting for types
  #    # TODO potentially slow loop
  #    types = list('%.18e' if self._getCompatibleType(data[0][i]) == float else '%s' for i in range(len(ordered)))
  #    np.savetxt(outFile,data,header=header,fmt=types)
  #  # format data?


  ### HIERARCHICAL STUFF ###
  def _constructHierPaths(self):
    """
      Construct a list of xr.Datasets, each of which is the samples taken along one hierarchical path
      @ In, None
      @ Out, results, list(xr.Dataset), dataset containing only the path information
    """
    # TODO can we do this without collapsing? Should we?
    data = self.asDataset()
    paths = self._generateHierPaths()
    results = [None] * len(paths)
    for p,path in enumerate(paths):
      rlzs = list(self._data.where(data['prefix']==ID,drop=True) for ID in path)
      results[p] = xr.concat(rlzs,dim=self.sampleTag)
    return results

  def _generateHierPaths(self):
    """
      Returns paths followed to obtain endings
      @ In, None
      @ Out, paths, list(list(str)), list of paths (which are lists of prefixes)
    """
    # get the ending realizations
    endings = self._getPathEndings()
    paths = [None]*len(endings)
    for e,ending in enumerate(endings):
      # reconstruct path that leads to this ending
      path = [ending['prefix']]
      while ending['RAVEN_parentID'] != "None" and not pd.isnull(ending['RAVEN_parentID']):
        _,ending = self.realization(matchDict={'prefix':ending['RAVEN_parentID']})
        path.append(ending['prefix'])
      # sort it in order by progression
      path.reverse()
      # add it to the path list
      paths[e] = path
    return paths

  def _getPathEndings(self):
    """
      Finds all those nodes who are the end of the line.
      @ In, None
      @ Out, endings, list({var:float/str or xr.DataArray}, ...), realizations
    """
    # TODO returning dicts means copying the data!  Do more efficiently by masking and creating xr.Dataset instances!
    # check if hierarchal data exists, by checking for the isEnding tag
    if not 'RAVEN_isEnding' in self.getVars():
      return []
    # get realization slices for each realization that is an ending
    # get from the collector first
    if self._collector is not None and len(self._collector) > 0:
      # first get rows from collector
      fromColl = self._collector[np.where(self._collector[:,self._orderedVars.index('RAVEN_isEnding')])]
      # then turn them into realization-like
      fromColl = list( dict(zip(self._orderedVars,c)) for c in fromColl )
    else:
      fromColl = []
    # then get from data
    if self._data is not None and len(self._data[self.sampleTag].values) > 0:
      # first get indexes of realizations
      indexes = self._data.where(self._data['RAVEN_isEnding'],drop=True)[self.sampleTag].values
      # then collect them into a list
      fromData = list(self._getRealizationFromDataByIndex(i) for i in indexes)
    else:
      fromData = []
    endings = fromColl + fromData
    return endings

