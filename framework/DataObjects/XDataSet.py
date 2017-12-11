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
  This class outlines the behavior for the basic in-memory DataObject, including support
  for ND and ragged input/output variable data shapes.  Other in-memory DataObjects are
  specialized implementations of this class.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys,os
import __builtin__
import itertools
import copy
import cPickle as pk
import xml.etree.ElementTree as ET

import abc
import numpy as np
import pandas as pd
import xarray as xr

# relative import for RAVEN, local import for unit tests
try:
  from .XDataObject import DataObject
except ValueError:
  from XDataObject import DataObject
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
    DataObject developed Oct 2017 to obtain linear performance from data objects when appending, over
    thousands of variables and millions of samples.  Wraps np.ndarray for collecting and uses xarray.Dataset
    for final form.  Subclasses are shortcuts (recipes) for this most general case.

    The interface for these data objects is specific.  The methods under "EXTERNAL API", "INITIALIZATION",
    and "BUILTINS" are the only methods that should be called to interact with the object.
  """
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
    keys = list(key for key in keys if key not in self._allvars)
    # if no new meta, move along
    if len(keys) == 0:
      return
    # CANNOT add expected meta after samples are started
    assert(self._data is None)
    assert(self._collector is None or len(self._collector) == 0)
    self._metavars.extend(keys)
    self._allvars.extend(keys)

  def addMeta(self,tag,xmlDict):
    """
      Adds general (not pointwise) metadata to this data object.  Can add several values at once, collected
      as a dict keyed by target variables.
      @ In, tag, str, section to add metadata to, usually the data submitter (BasicStatistics, DataObject, etc)
      @ In, xmlDict, dict, data to change, of the form {target:{scalarMetric:value,scalarMetric:value,vectorMetric:{wrt:value,wrt:value}}}
      @ Out, None
    """
    # Data ends up being written as follows (see docstrings above for dict structure)
    #  - A good default for 'target' is 'general' if there's not a specific target
    # <tag>
    #   <target>
    #     <scalarMetric>value</scalarMetric>
    #     <scalarMetric>value</scalarMetric>
    #     <vectorMetric>
    #       <wrt>value</wrt>
    #       <wrt>value</wrt>
    #     </vectorMetric>
    #   </target>
    #   <target>
    #     <scalarMetric>value</scalarMetric>
    #     <vectorMetric>
    #       <wrt>value</wrt>
    #     </vectorMetric>
    #   </target>
    # </tag>
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
          assert(isinstance(value,(str,unicode,float,int)))
          destination.addScalar(target,metric,value)

  def addRealization(self,rlz):
    """
      Adds a "row" (or "sample") to this data object.
      This is the method to add data to this data object.
      Note that rlz can include many more variables than this data object actually wants.
      Before actually adding the realization, data is formatted for this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" is either a float or a np.ndarray of values.
      @ Out, None
    """
    # protect against back-changing realization
    rlz = copy.deepcopy(rlz)
    # clean out entries that aren't desired
    try:
      rlz = dict((var,rlz[var]) for var in self._allvars+self.indexes)
    except KeyError as e:
      self.raiseAnError(KeyError,'Provided realization does not have all requisite values: "{}"'.format(e.args[0]))
    # check consistency, but make it an assertion so it can be passed over
    if not self._checkRealizationFormat(rlz):
      self.raiseAnError(SyntaxError,'Realization was not formatted correctly! See warnings above.')
    # format the data
    rlz = self._formatRealization(rlz)
    # perform selective collapsing/picking of data
    rlz = self._selectiveRealization(rlz)
    # FIXME if no scalar entry is made, this construction fails.
    #  Instead of treating each dataarrray as an object, numpy.asarray calls their asarray methods,
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
    newData = np.asarray([list(rlz[var] for var in self._allvars)+[0.0]],dtype=object)
    newData = newData[:,:-1]
    # if data storage isn't set up, set it up
    if self._collector is None:
      self._collector = self._newCollector(width=len(rlz))
    # append
    self._collector.append(newData)
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
    self._allvars.append(varName)

  def asDataset(self, outType=None):
    """
      Casts this dataObject as dictionary or an xr.Dataset depending on outType.
      @ In, outType, str, optional, type of output object (xr.Dataset or dictionary).
      @ Out, xr.Dataset or dictionary.
    """
    if outType is None or outType=='xrDataset':
      # return the xArray, i.e., the old asDataset()
      return self._convertToXrDataset()
    elif outType=='dict':
      # return a dict
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
    if isinstance(indexesToCheck,(str,unicode)):
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
    # TODO while simple, this API will allow easier extensibility in the future.
    obj = xr.DataArray(vals,dims=dims,coords=coords)
    obj.rename(name)
    return obj

  def extendExistingEntry(self,rlz):
    """
      Extends an ND sample to include more data.
      Probably only useful for the hacky way the Optimizer stores Trajectories.
      @ In, rlz, dict, {name:value} as {str:float} of the variables to extend
      @ Out, None
    """
    assert(type(rlz) == dict)
    # modify outputs to be pivot-dependent
    # set up index vars for removal
    toRemove = (var for var in self.indexes if var in allDims)
    # dimensionalize outputs
    for var in self._outputs:
      # TODO are they always all there?
      # TODO check the right dimensional shape and order
      vals = rlz[var]
      dims = self.getDimensions(var)[var]
      coords = dict((dim,rlz[dim]) for dim in dims)
      rlz[var] = self.constructNDSample(vals,dims,coords)
    # remove indexes
    for var in toRemove:
      del rlz[var]
    # first time? Initialize everything
    if (self._collector is None or len(self._collector)==0) and self._data is None:
      if self._collector is None:
        self._collector = self._newCollector(width=len(rlz),dtype=object)
      self.addRealization(rlz)
    # collector is present # TODO match search should happen both in collector and in data!
    elif len(self._collector) > 0:
      # find the entry to modify
      matchIdx,matchRlz = self.realization(matchDict=dict((var,rlz[var]) for var in self._inputs))
      if matchRlz is None:
        #we're adding the entry as if new # TODO duplicated code
        self.addRealization(rlz)
      else:
        # extend each existing dataarray with the new value
        extend
    else:
      indata

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
      var = self._allvars
    elif var in ['input','output']:
      var = self.getVars(var)
    else:
      var = [var]
    dims = dict((v,list(key for key in self._pivotParams.keys() if v in self._pivotParams[key])) for v in var)
    return dims

  def getMeta(self,keys=None,pointwise=False,general=False):
    """
      Method to obtain entries in the metadata.  If niether pointwise nor general, then returns an empty dict.
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
    if isinstance(var,(str,unicode)):
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

  def load(self,fName,style='netCDF',**kwargs):
    """
      Reads this dataset from disk based on the format.
      @ In, fName, str, path and name of file to read
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to reading function
      @ Out, None
    """
    style = style.lower()
    # if fileToLoad in kwargs, then filename is actualle fName/fileToLoad
    if 'fileToLoad' in kwargs.keys():
      fName = kwargs['fileToLoad'].getAbsFile()
    # load based on style for loading
    if style == 'netcdf':
      self._fromNetCDF(fName,**kwargs)
    elif style == 'csv':
      # make sure we don't include the "csv"
      if fName.endswith('.csv'):
        fName = fName[:-4]
      self._fromCSV(fName,**kwargs)
    elif style == 'dict':
      self._fromDict(fName,**kwargs)
    # TODO dask
    else:
      self.raiseAnError(NotImplementedError,'Unrecognized read style: "{}"'.format(style))
    # after loading, set or reset scaling factors
    self._setScalingFactors()

  def realization(self,index=None,matchDict=None,tol=1e-15):
    """
      Method to obtain a realization from the data, either by index or matching value.
      Either "index" or "matchDict" must be supplied.
      If matchDict and no match is found, will return (len(self),None) after the pattern of numpy, scipy
      @ In, index, int, optional, number of row to retrieve (by index, not be "sample")
      @ In, matchDict, dict, optional, {key:val} to search for matches
      @ In, tol, float, optional, tolerance to which match should be made
      @ Out, index, int, optional, index where found (or len(self) if not found), only returned if matchDict
      @ Out, rlz, dict, realization requested (None if not found)
    """
    # FIXME the caller should have no idea whether to read the collector or not.
    # TODO convert input space to KD tree for faster searching -> XArray.DataArray has this built in
    # TODO option to read both collector and data for matches/indices
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
            self.raiseAnError(IndexError,'Requested index "{}" but only have {} entries (zero-indexed)!'.format(index,numInData+numInCollector))
          ## otherwise, take from the collector
          else:
            rlz = self._getRealizationFromCollectorByIndex(index - numInData)
        ## otherwise, take from the data
        else:
          rlz = self._getRealizationFromDataByIndex(index)
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
            rlz = self._getRealizationFromDataByIndex(index + numInCollector)
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
        index,rlz = self._getRealizationFromDataByValue(matchDict,tol=tol)
        # if no match found in data, try in the collector (if there's anything in it)
        if rlz is None:
          if numInCollector > 0:
            index,rlz = self._getRealizationFromCollectorByValue(matchDict,tol=tol)
      return index,rlz

  def remove(self,realization=None,variable=None):
    """
      Used to remove either a realization or a variable from this data object.
      @ In, realization, dict or int, optional, (matching or index of) realization to remove
      @ In, variable, str, optional, name of "column" to remove
      @ Out, None
    """
    if self.size == 0:
      self.raiseAWarning('Called "remove" on DataObject, but it is empty!')
      return
    noData = self._data is None or len(self._data) == 0
    noColl = self._collector is None or len(self._collector) == 0
    assert(not (realization is None and variable is None))
    assert(not (realization is not None and variable is not None))
    assert(self._data is not None)
    # TODO what about removing from collector?
    if realization is not None:
      # TODO reset scaling factors
      self.raiseAnError(NotImplementedError,'TODO implementation for removing realizations is not currently implemented!')
    elif variable is not None:
      # remove from self._data
      if not noData:
        self._data = self._data.drop(variable)
      # remove from self._collector
      if not noColl:
        varIndex = self._allvars.index(variable)
        self._collector.removeEntity(varIndex)
      # remove references to variable in lists
      self._allvars.remove(variable)
      # TODO potentially slow lookups
      for varlist in [self._inputs,self._outputs,self._metavars]:
        if variable in varlist:
          varlist.remove(variable)
      # TODO remove references from general metadata?
    if self._scaleFactors is not None:
      self._scaleFactors.pop(variable,None)
    #either way reset kdtree
    self.inputKDTree = None

  def reset(self):
    """
      Sets this object back to its initial state.
      @ In, None
      @ Out, None
    """
    self._data = None
    self._collector = None
    self._meta = {}
    # TODO others?

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

  def write(self,fName,style='netCDF',**kwargs):
    """
      Writes this dataset to disk based on the format.
      @ In, fName, str, path and name of file to write
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to writing function
          Includes:  firstIndex, int, optional, if included then is the realization index that writing should start from (implies appending instead of rewriting)
      @ Out, index, int, index of latest rlz to be written, for tracking purposes
    """
    self.asDataset() #just in case there is stuff left in the collector
    if style.lower() == 'netcdf':
      self._toNetCDF(fName,**kwargs)
    elif style.lower() == 'csv':
      if len(self._data[self.sampleTag])==0: #TODO what if it's just metadata?
        self.raiseAWarning('Nothing to write!')
        return
      #first write the CSV
      firstIndex = kwargs.get('firstIndex',0)
      self._toCSV(fName,start=firstIndex,**kwargs)
      # then the metaxml
      self._toCSVXML(fName,**kwargs)
    # TODO dask?
    else:
      self.raiseAnError(NotImplementedError,'Unrecognized write style: "{}"'.format(style))
    return len(self) # so that other entities can track which realization we've written

  ### INITIALIZATION ###
  # These are the necessary functions to construct and initialize this data object
  def __init__(self):#, in_vars, out_vars, meta_vars=None, dynamic=False, var_dims=None,cacheSize=100,prealloc=False):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    DataObject.__init__(self)
    self.name      = 'DataSet'
    self.type      = 'DataSet'
    self.printTag  = self.name
    self.defaultDtype = object
    self._scaleFactors = {}     # mean, sigma for data for matching purposes

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
  def vars(self):
    """
      Property to access all the pointwise variables being controlled by this data object.
      As opposed to "self._allvars", returns the variables clustered by subset (inp, out, meta) instead of order added
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
    return self._pivotParams.keys()

  ### INTERNAL USE FUNCTIONS ###
  def _collapseNDtoDataArray(self,data,var,labels=None):
    """
      Converts a row of numpy samples (float or xr.DataArray) into a single DataArray suitable for a xr.Dataset.
      @ In, data, np.ndarray, array of either float or xr.DataArray; array must be single-dimension
      @ In, var, str, name of the variable being acted on
      @ In, labels, list, list of labels to use for collapsed array under self.sampleTag title
      @ Out, DataArray, xr.DataArray, single dataarray object
    """
    assert(isinstance(data,np.ndarray))
    assert(len(data.shape) == 1)
    if labels is None:
      labels = range(len(data))
    else:
      assert(len(labels) == len(data))
    #method = 'once' # see below, parallelization is possible but not implemented
    # first case: single entry per node: floats, strings, ints, etc
    if isinstance(data[0],(float,str,unicode,int)):
      array = xr.DataArray(data,
                           dims=[self.sampleTag],
                           coords={self.sampleTag:labels},
                           name=var) # THIS is very fast
    # second case: ND set (history set or higher dimension) --> CURRENTLY should be unused
    elif type(data[0]) == xr.DataArray:
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

  def _convertArrayListToDataset(self,array,action=None):
    """
      Converts a 1-D array of xr.DataArrays into a xr.Dataset, then takes action on self._data:
      action=='replace': replace self._data with the new dataset
      action=='extend' : add new dataset to self._data using merge
      else             : only return new dataset
      @ In, array, list(xr.DataArray), list of variables as samples to turn into dataset
      @ In, action, str, optional, can be used to specify the action to take with the new dataset
      @ Out, new, xr.Dataset, single data entity
    """
    try:
      new = xr.Dataset(array)
    except ValueError as e:
      self.raiseAnError(RuntimeError,'While trying to create a new Dataset, a variable has itself as an index!  Error: ' +str(e))
    if action == 'replace': #self._data is None:
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
      self._data.merge(new,inplace=True)
    # set up scaling factors
    self._setScalingFactors()
    return new

  def _convertFinalizedDataRealizationToDict(self,rlz):
    """
      After collapsing into xr.Dataset, all entries are stored as xr.DataArrays.
      This converts them into a dictionary like the realization sent in.
      @ In, rlz, dict(varname:xr.DataArray), "row" from self._data
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
        for dim in v.dims[:]:
          v = v.dropna(dim)
        new[k] = v
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
    ## initialize with np arrays of objects
    dataDict['data'] = dict((var,np.zeros(self.size,dtype=object)) for var in self.vars+self.indexes)
    ## loop over realizations to get distinct values without NaNs
    for var in self.vars:
      # how we get and store variables depends on the dimensionality of the variable
      dims=self.getDimensions(var)[var]
      # if scalar (no dims and not an index), just grab the values
      if len(dims)==0 and var not in self.indexes:
        dataDict['data'][var] = self.asDataset()[var].values
        continue
      # otherwise, need to remove NaNs, so loop over slices
      for s,rlz in enumerate(self.sliceByIndex(self.sampleTag)):
        # get data specific to this var for this realization (slice)
        data = rlz[var]
        # need to drop indexes for which no values are present
        for index in dims:
          data = data.dropna(index)
          #if dataDict['data'][index][s] == 0:
          dataDict['data'][index][s] = data[index].values
        dataDict['data'][var][s] = data.values
    return dataDict

  def _convertToXrDataset(self):
    """
      Casts this dataobject as an xr.Dataset.
      Functionally, typically collects the data from self._collector and places it in self._data.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, xarray.Dataset, all the data from this data object.
      P.S.: this was the old asDataset(self)
    """
    # TODO make into a protected method? Should it be called from outside?
    # if we have collected data, collapse it
    if self._collector is not None and len(self._collector) > 0:
      data = self._collector.getData()
      firstSample = int(self._data[self.sampleTag][-1])+1 if self._data is not None else 0
      arrs = {}
      for v,var in enumerate(self._allvars):
        # create single dataarrays
        arrs[var] = self._collapseNDtoDataArray(data[:,v],var)
        # re-index samples
        arrs[var][self.sampleTag] += firstSample
      # collect all data into dataset, and update self._data
      if self._data is None:
        self._convertArrayListToDataset(arrs,action='replace')
      else:
        self._convertArrayListToDataset(arrs,action='extend')
      # reset collector
      self._collector = self._newCollector(width=self._collector.width,dtype=self._collector.values.dtype)
    return self._data

  def _formatRealization(self,rlz):
    """
      Formats realization without truncating data
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    # TODO this could be much more efficient on the parallel (finalizeCodeOutput) than on serial
    # TODO costly for loop
    indexes = []
    for var,val in rlz.items():
      # if an index variable, skip it and mark it for removal
      if var in self._pivotParams.keys():
        indexes.append(var)
        continue
      dims = self.getDimensions(var)[var]
      ## change dimensionless to floats -> TODO use operator to collapse!
      if dims in [[self.sampleTag], []]:
        if len(val) == 1:
          rlz[var] = val[0]
      ## reshape multidimensional entries into dataarrays
      else:
        coords = dict((d,rlz[d]) for d in dims)
        rlz[var] = self.constructNDSample(val,dims,coords,name=var)
    for var in indexes:
      del rlz[var]
    return rlz

  def _fromCSV(self,fName,**kwargs):
    """
      Loads a dataset from CSV (preferably one it wrote itself, but maybe not necessarily?
      @ In, fName, str, filename to load from (not including .csv or .xml)
      @ In, kwargs, dict, optional arguments
      @ Out, None
    """
    assert(self._data is None)
    assert(self._collector is None)
    # first, try to read from csv
    try:
      panda = pd.read_csv(fName+'.csv')
    except pd.errors.EmptyDataError:
      # no data in file
      self.raiseAWarning('Tried to read data from "{}", but the file is empty!'.format(fName+'.csv'))
      return
    finally:
      self.raiseADebug('Reading data from "{}.csv"'.format(fName))
    # then, read in the XML
    # TODO what if no xml? -> read as point set?
    # TODO separate _fromCSVXML for clarity and brevity
    meta = self._fromCSVXML(fName)
    # apply findings
    self.sampleTag = meta.get('sampleTag',self.sampleTag)
    dims = meta.get('pivotParams',{}) # stores the dimensionality of each variable
    if len(dims)>0:
      self.setPivotParams(dims)
    # TODO make this into a consistency check instead?  Selective loading?
    self._inputs = meta.get('inputs',self._inputs)
    self._outputs = meta.get('outputs',self._outputs)
    self._metavars = meta.get('metavars',self._metavars)
    # replace the IO space, default to user input # TODO or should we be sticking to user's wishes?  Probably.
    self._allvars = self._inputs + self._outputs + self._metavars
    # find distinct number of samples
    try:
      samples = list(set(panda[self.sampleTag]))
    except KeyError:
      # sample ID wasn't given, so assume each row is sample
      samples = range(len(panda.index))
      panda[self.sampleTag] = samples
    # create arrays from which to create the data set
    arrays = {}
    for var in self._allvars:
      if var in dims.keys():
        data = panda[[var,self.sampleTag]+dims[var]]
        data.set_index(self.sampleTag,inplace=True)
        ndat = np.zeros(len(samples),dtype=object)
        for s,sample in enumerate(samples):
          places = data.index.get_loc(sample)
          vals = data[places].dropna().set_index(dims[var])#.set_index(dims[var]).dropna()
          #vals.drop('dim_1')
          # TODO this needs to be improved before ND will work; we need the individual sub-indices (time, space, etc)
          rlz = xr.DataArray(vals.values[:,0],dims=dims[var],coords=dict((var,vals.index.values) for var in dims[var]))
          ndat[s] = rlz
          #rlzdat = xr.DataArray(vals,dims=['time'],coords=dict((var,vals[var].values[0]) for var in dims[var]))
          #print rlzdat
        arrays[var] = self._collapseNDtoDataArray(ndat,var,labels=samples)
      else:
        # scalar example
        data = panda[[var,self.sampleTag]].groupby(self.sampleTag).first().values[:,0]
        arrays[var] = self._collapseNDtoDataArray(data,var,labels=samples)
    self._convertArrayListToDataset(arrays,action='replace')

  def _fromCSVXML(self,fName):
    """
      Loads in the XML portion of a CSV if it exists.  Returns information found.
      @ In, fName, str, filename to read as filename.xml
      @ Out, metadata, dict, metadata discovered
    """
    metadata = {}
    # check if we have anything from which to read
    try:
      meta,_ = xmlUtils.loadToTree(fName+'.xml')
    except IOError:
      haveMeta = False
    finally:
      self.raiseADebug('Reading metadata from "{}.xml"'.format(fName))
      haveMeta = True
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
      @ In, source, dict, as {var:values} with types {str:np.array}
      @ In, dims, dict, optional, ordered list of dimensions that each var depends on as {var:[list]}
      @ In, kwargs, dict, optional, additional arguments
      @ Out, None
    """
    assert(self._data is None)
    assert(self._collector is None)
    # not safe to default to dict, so if "dims" not specified set it here
    if dims is None:
      dims = {}
    # data sent in is as follows:
    #   single-entry (scalars) - np.array([val, val, val])
    #   histories              - np.array([np.array(vals), np.array(vals), np.array(vals)])
    #   etc
    ## check that all inputs, outputs required are provided
    providedVars = set(source.keys())
    requiredVars = set(self.getVars('input')+self.getVars('output'))
    ## determine what vars are metadata (the "extra" stuff that isn't output or input
    # TODO don't take "extra", check registered meta explicitly
    extra = list(e for e in providedVars - requiredVars if e not in self.indexes)
    self._metavars = extra
    ## figure out who's missing from the IO space
    missing = requiredVars - providedVars
    if len(missing) > 0:
      self.raiseAnError(KeyError,'Variables are missing from "source" that are required for this data object:',missing)
    # make a collector from scratch -> start by collecting into ndarray
    rows = len(source.values()[0])
    cols = len(self.getVars())
    data = np.zeros([rows,cols],dtype=object)
    for i,var in enumerate(itertools.chain(self._inputs,self._outputs,self._metavars)):
      values = source[var]
      # TODO consistency checking with dimensions requested by the user?  Or override them?
      #  -> currently overriding them
      varDims = dims.get(var,[])
      # format higher-than-one-dimensional variables into a list of xr.DataArray
      for dim in varDims:
        ## first, make sure we have all the dimensions for this variable
        if dim not in source.keys():
          self.raiseAnError(KeyError,'Variable "{}" depends on dimension "{}" but it was not provided to _fromDict in the "source"!'.format(var,dim))
        ## construct ND arrays
        for v,val in enumerate(values):
          ## coordinates come from each dimension, specific to the "vth" realization
          coords = dict((dim,source[dim][v]) for dim in varDims)
          ## swap-in-place the construction; this will likely error if there's inconsistencies
          values[v] = self.constructNDSample(val,varDims,coords,name=var)
        #else:
        #  pass # TODO need to make sure entries are all single entries!
      data[:,i] = values
    # set up collector as cached nd array of values
    self._collector = cached_ndarray.cNDarray(values=data,dtype=object)
    # collapse into xr.Dataset
    self.asDataset()

  def _fromNetCDF(self,fName, **kwargs):
    """
      Reads this data object from file that is netCDF.  If not netCDF4, this could be slow.
      Loads data lazily; it won't be pulled into memory until operations are attempted on the specific data
      @ In, fName, str, path/name to read file
      @ In, kwargs, dict, optional, keywords to pass to netCDF4 reading
                                    See http://xarray.pydata.org/en/stable/io.html#netcdf for options
      @ Out, None
    """
    # TODO set up to use dask for on-disk operations -> or is that a different data object?
    # TODO are these fair assertions?
    assert(self._data is None)
    assert(self._collector is None)
    self._data = xr.open_dataset(fName)
    # convert metadata back to XML files
    for key,val in self._data.attrs.items():
      self._meta[key] = pk.loads(val.encode('utf-8'))

  def _getRealizationFromCollectorByIndex(self,index):
    """
      Obtains a realization from the collector storage using the provided index.
      @ In, index, int, index to return
      @ Out, rlz, dict, realization as {var:value}
    """
    assert(self._collector is not None)
    assert(index < len(self._collector))
    rlz = dict(zip(self._allvars,self._collector[index]))
    return rlz

  def _getRealizationFromCollectorByValue(self,match,tol=1e-15):
    """
      Obtains a realization from the collector storage matching the provided index
      @ In, match, dict, elements to match
      @ In, tol, float, optional, tolerance to which match should be made
      @ Out, r, int, index where match was found OR size of data if not found
      @ Out, rlz, dict, realization as {var:value} OR None if not found
    """
    assert(self._collector is not None)
    # TODO KD Tree for faster values -> still want in collector?
    # TODO slow double loop
    lookingFor = match.values()
    for r,row in enumerate(self._collector[:,tuple(self._allvars.index(var) for var in match.keys())]):
      match = True
      for e,element in enumerate(row):
        if isinstance(element,float):
          match *= mathUtils.compareFloats(lookingFor[e],element,tol=tol)
          if not match:
            break
      if match:
        break
    if match:
      return r,self._getRealizationFromCollectorByIndex(r)
    else:
      return len(self),None

  def _getRealizationFromDataByIndex(self,index):
    """
      Obtains a realization from the data storage using the provided index.
      @ In, index, int, index to return
      @ Out, rlz, dict, realization as {var:value} where value is a DataArray with only coordinate dimensions
    """
    assert(self._data is not None)
    #assert(index < len(self._data[self.sampleTag]))
    rlz = self._data[{self.sampleTag:index}].drop(self.sampleTag).data_vars
    rlz = self._convertFinalizedDataRealizationToDict(rlz)
    return rlz

  def _getRealizationFromDataByValue(self,match,tol=1e-15):
    """
      Obtains a realization from the data storage using the provided index.
      @ In, match, dict, elements to match
      @ In, tol, float, optional, tolerance to which match should be made
      @ Out, r, int, index where match was found OR size of data if not found
      @ Out, rlz, dict, realization as {var:value} OR None if not found
    """
    assert(self._data is not None)
    # TODO this could be slow, should do KD tree instead
    mask = 1.0
    for var,val in match.items():
      # float instances are relative, others are absolute
      if isinstance(val,(float,int)):
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
    return idx,self._getRealizationFromDataByIndex(idx)

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
        else:
          keep.append(entry.split('|')[-1].strip())
    else:
      # TODO need the sampleTag meta to load histories # BY DEFAULT only keep inputs, outputs; if specifically requested, keep metadata by selection
      keep = self._inputs + self._outputs + self._metavars
    return keep

  def _getVariableIndex(self,var):
    """
      Obtains the index in the list of variables for the requested var.
      @ In, var, str, variable name (input, output, or pointwise medatada)
      @ Out, index, int, column corresponding to the variable
    """
    return self._allvars.index(var)

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

  def _resetScaling(self):
    """
      Removes the KDTree and scaling factors, usually because the data changed in some way
      @ In, None
      @ Out, None
    """
    self._scaleFactors = {}
    self._inputKDTree = None

  def _selectiveRealization(self,rlz,checkLengthBeforeTruncating=False):
    """
      Formats realization to contain the desired data
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    for var, val in rlz.items():
      if isinstance(val,np.ndarray):
        self.raiseAnError(NotImplementedError,'Variable "{}" has no dimensions but has multiple values!  Not implemented for DataSet yet.'.format(var))
    return rlz

  def _setScalingFactors(self,var=None):
    """
      Sets the scaling factors for the data (mean, scale).
      @ In, var, str, optional, if given then will only set factors for "var"
      @ Out, None
    """
    if var is None:
      # clear existing factors and set list to "all"
      self._scaleFactors = {}
      varList = self._allvars
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
      #if (not isinstance(first,(float,int))) or np.isnan(first):# or self._data[var].isnull().all():
      #  continue
      try:
        mean = float(self._data[var].mean())
        scale = float(self._data[var].std())
        self._scaleFactors[var] = (mean,scale)
      except TypeError:
        pass

  def _toCSV(self,fName,start=0,**kwargs):
    """
      Writes this data object to CSV file (except the general metadata, see _toCSVXML)
      @ In, fName, str, path/name to write file
      @ In, start, int, optional, first realization to start printing from (if > 0, implies append mode)
      @ In, kwargs, dict, optional, keywords for options
      @ Out, None
    """
    filenameLocal = fName # TODO path?
    keep = self._getRequestedElements(kwargs)
    toDrop = list(var for var in self._allvars if var not in keep)
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

  def _toCSVXML(self,fName,**kwargs):
    """
      Writes the general metadata of this data object to XML file
      @ In, fName, str, path/name to write file
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

    self.raiseADebug('Printing metadata XML: "{}"'.format(fName+'.xml'))
    with open(fName+'.xml','w') as ofile:
      #header
      ofile.writelines('<DataObjectMetadata name="{}">\n'.format(self.name))
      for name,target in meta.items():
        xml = target.writeFile(asString=True,startingTabs=1,addRavenNewlines=False)
        ofile.writelines('  '+xml+'\n')
      ofile.writelines('</DataObjectMetadata>\n')

  def _toNetCDF(self,fName,**kwargs):
    """
      Writes this data object to file in netCDF4.
      @ In, fName, str, path/name to write file
      @ In, kwargs, dict, optional, keywords to pass to netCDF4 writing
                                    One good option is format='NETCDF4' to assure netCDF4 is used
                                    See http://xarray.pydata.org/en/stable/io.html#netcdf for options
      @ Out, None
    """
    # TODO set up to use dask for on-disk operations -> or is that a different data object?
    # convert metadata into writeable
    self._data.attrs = dict((key,pk.dumps(val)) for key,val in self._meta.items())
    self._data.to_netcdf(fName,**kwargs)

  def _usePandasWriteCSV(self,fName,data,ordered,keepSampleTag=False,keepIndex=False,mode='w'):
    """
      Uses Pandas to write a CSV.
      @ In, fName, str, path/name to write file
      @ In, data, xr.Dataset, data to write (with only "keep" vars included, plus self.sampleTag)
      @ In, ordered, list(str), ordered list of headers
      @ In, keepSampleTag, bool, optional, if True then keep the samplerTag in the CSV
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
      data.to_csv(fName+'.csv',mode=mode,header=header)
    else:
      # if other multiindices included, don't omit them #for ND DataSets only
      if isinstance(data.index,pd.MultiIndex):
        # if we have just the self.sampleTag index (we can not drop it otherwise pandas fail). We use index=False (a.a.)
        indexx = False if len(data.index.names) == 1 else True
        if indexx:
          data.index = data.index.droplevel(self.sampleTag)
        data.to_csv(fName+'.csv',mode=mode,header=header, index=indexx)
      # if keepIndex, then print as is
      elif keepIndex:
        data.to_csv(fName+'.csv',mode=mode,header=header)
      # if only index was sampleTag and we don't want it, index = False takes care of that
      else:
        data.to_csv(fName+'.csv',index=False,mode=mode,header=header)
    #raw_input('Just wrote to CSV "{}.csv", press enter to continue ...'.format(fName))
