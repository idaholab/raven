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
  def profile(func): return func

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
    # check consistency, but make it an assertion so it can be passed over
    assert self._checkRealizationFormat(rlz),'Realization was not formatted correctly! See warnings above.'
    # first, update realization with selectors
    rlz = self._formatRealization(rlz)
    rlz = self._selectiveRealization(rlz)
    # if collector/data not yet started, expand entries that aren't I/O as metadata
    if self._data is None and self._collector is None:
      unrecognized = set(rlz.keys()).difference(set(self._allvars))
      if len(unrecognized) > 0:
        self._metavars = list(unrecognized)
        self._allvars += self._metavars
    # check and order data to be stored
    try:
      newData = np.asarray([list(rlz[var] for var in self._allvars)],dtype=object)
    except KeyError as e:
      self.raiseAnError(KeyError,'Provided realization does not have all requisite values: "{}"'.format(e.args[0]))
    # if data storage isn't set up, set it up
    if self._collector is None:
      self._collector = self._newCollector(width=len(rlz))
    self._collector.append(newData)
    # reset scaling factors, kd tree
    self._resetScaling()

  def asDataset(self):
    """
      Casts this dataobject as an xr.Dataset.
      Functionally, typically collects the data from self._collector and places it in self._data.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, xarray.Dataset, all the data from this data object.
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
    toRemove = []
    # set up index vars for removal
    allDims = self.getDimensions()
    for var in rlz.keys():
      if var in allDims:
        toRemove.append(var)
    # dimensionalize outputs
    for var in self._outputs:
      # TODO are they always all there?
      # TODO check the right dimensional shape and order
      vals = rlz[var]
      dims = self.getDimensions(var)
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

  def getDimensions(self,var):
    """
      Provides the independent dimensions that this variable depends on.
      To get all dimensions at once, use self.indexes property.
      @ In, var, str, name of variable (if None, give all)
      @ Out, dims, dict, {name:values} of independent dimensions
    """
    # TODO add unit tests
    # TODO allow several variables requested at once?
    dims = list(key for key in self._pivotParams.keys() if var in self._pivotParams[key])
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
          meta[var] = self.asDataset()[var]#[self._allvars.index(var),:]
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
    # TODO have to convert here?
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


    ##### FIXME OLD
    #if readCollector:
    #  if self._collector is None or len(self._collector)==0:
    #    if matchDict is not None:
    #      return 0,None
    #    else:
    #      return None
    #elif self._data is None or len(self._data)==0:
    #  return 0,None
    #if index is not None:
    #  if readCollector:
    #    rlz = self._getRealizationFromCollectorByIndex(index)
    #  else:
    #    rlz = self._getRealizationFromDataByIndex(index)
    #  return rlz
    #else: #because of check above, this means matchDict is not None
    #  if readCollector:
    #    # TODO scaling factors and collector
    #    index,rlz = self._getRealizationFromCollectorByValue(matchDict,tol=tol)
    #  else:
    #    if self._scaleFactors is None:
    #      self._setScalingFactors()
    #    index,rlz = self._getRealizationFromDataByValue(matchDict,tol=tol)
    #  return index,rlz

  def load(self,fname,style='netCDF',**kwargs):
    """
      Reads this dataset from disk based on the format.
      @ In, fname, str, path and name of file to read
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to reading function
      @ Out, None
    """
    style = style.lower()
    if style == 'netcdf':
      self._fromNetCDF(fname,**kwargs)
    elif style == 'csv':
      self._fromCSV(fname,**kwargs)
    # TODO dask
    else:
      self.raiseAnError(NotImplementedError,'Unrecognized read style: "{}"'.format(style))

  def remove(self,realization=None,variable=None):
    """
      Used to remove either a realization or a variable from this data object.
      @ In, realization, dict or int, optional, (matching or index of) realization to remove
      @ In, variable, str, optional, name of "column" to remove
      @ Out, None
    """
    if self._data is None or len(self._data) == 0: #TODO what about collector?
      return
    assert(not (realization is None and variable is None))
    assert(not (realization is not None and variable is not None))
    assert(self._data is not None)
    # TODO what about removing from collector?
    if realization is not None:
      # TODO reset scaling factors
      self.raiseAnError(NotImplementedError,'TODO')
    elif variable is not None:
      self._data.drop(variable)
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

  def write(self,fname,style='netCDF',**kwargs):
    """
      Writes this dataset to disk based on the format.
      @ In, fname, str, path and name of file to write
      @ In, style, str, optional, options are enumerated below
      @ In, kwargs, dict, optional, additional arguments to pass to writing function
      @ Out, None
    """
    if style.lower() == 'netcdf':
      self._toNetCDF(fname,**kwargs)
    elif style.lower() == 'csv':
      self.asDataset()
      if self._data is None or len(self._data)==0: #TODO what if it's just metadata?
        self.raiseAWarning('Nothing to write!')
        return
      #first write the CSV
      self._toCSV(fname,**kwargs)
      # then the metaxml
      self._toCSVXML(fname,**kwargs)
    # TODO dask?
    else:
      self.raiseAnError(NotImplementedError,'Unrecognized write style: "{}"'.format(style))

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
      dims = self.getDimensions(var)
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

  def _fromCSV(self,fname,**kwargs):
    """
      Loads a dataset from CSV (preferably one it wrote itself, but maybe not necessarily?
      @ In, fname, str, filename to load from (not including .csv or .xml)
      @ In, kwargs, dict, optional arguments
      @ Out, None
    """
    assert(self._data is None)
    assert(self._collector is None)
    # first, try to read from csv
    try:
      panda = pd.read_csv(fname+'.csv')
    except pd.errors.EmptyDataError:
      # no data in file
      self.raiseAWarning('Tried to read data from "{}", but the file is empty!'.format(fname+'.csv'))
      return
    finally:
      self.raiseADebug('Reading data from "{}.csv"'.format(fname))
    # then, read in the XML
    # TODO what if no xml? -> read as point set?
    # TODO separate _fromCSVXML for clarity and brevity
    try:
      meta,_ = xmlUtils.loadToTree(fname+'.xml')
      self.raiseADebug('Reading metadata from "{}.xml"'.format(fname))
      haveMeta = True
    except IOError:
      haveMeta = False
    dims = {} # stores the dimensionality of each variable
    # confirm we have usable meta
    if haveMeta:
      # get the sample tag
      tagNode = xmlUtils.findPath(meta,'DataSet/general/sampleTag')
      if tagNode is not None:
        self.sampleTag = tagNode.text
      #else:
      #  pass # use default
    # collect essential data from the meta
    alldims = set([])
    if haveMeta:
      # unwrap the dimensions
      #try:
      dimsNode = xmlUtils.findPath(meta,'DataSet/dims')
      if dimsNode is not None:
        for child in dimsNode:
          new = child.text.split(',')
          dims[child.tag] = new
          alldims.update(new)
        self.setPivotParams(dims)
    # replace the IO space, default to user input # TODO or should we be sticking to user's wishes?  Probably.
    if haveMeta:
      # TODO make this into a consistency check instead
      inputsNode = xmlUtils.findPath(meta,'DataSet/general/inputs')
      if inputsNode is not None:
        self._inputs = inputsNode.text.split(',')
      outputsNode = xmlUtils.findPath(meta,'DataSet/general/outputs')
      if outputsNode is not None:
        self._outputs = outputsNode.text.split(',')
      # these DO have to be read from meta if present
      metavarsNode = xmlUtils.findPath(meta,'DataSet/general/pointwise_meta')
      if metavarsNode is not None:
        self._metavars = metavarsNode.text.split(',')
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
    # TODO this is common with "asDataset()", so make a function for this!
    self._convertArrayListToDataset(arrays,action='replace')

  def _fromNetCDF(self,fname, **kwargs):
    """
      Reads this data object from file that is netCDF.  If not netCDF4, this could be slow.
      Loads data lazily; it won't be pulled into memory until operations are attempted on the specific data
      @ In, fname, str, path/name to read file
      @ In, kwargs, dict, optional, keywords to pass to netCDF4 reading
                                    See http://xarray.pydata.org/en/stable/io.html#netcdf for options
      @ Out, None
    """
    # TODO set up to use dask for on-disk operations -> or is that a different data object?
    # TODO are these fair assertions?
    assert(self._data is None)
    assert(self._collector is None)
    self._data = xr.open_dataset(fname)
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
        except IndexError:
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
      keep = self._inputs + self._outputs
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
    self._scaleFactors = None
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

  def _setScalingFactors(self):
    """
      Sets the scaling factors for the data (mean, scale).
      @ In, None
      @ Out, None
    """
    # TODO someday make KDTree too!
    assert(self._data is not None) # TODO check against collector entries?
    self._scaleFactors = {}
    for var in self._allvars:
      # if not a float or int, don't scale it
      # TODO this check is pretty convoluted; there's probably a better way to figure out the type of the variable
      first = self._data.groupby(var).first()[var].item(0)
      if (not isinstance(first,(float,int))) or np.isnan(first):# or self._data[var].isnull().all():
        continue
      mean = float(self._data[var].mean())
      scale = float(self._data[var].std())
      self._scaleFactors[var] = (mean,scale)

  def _toCSV(self,fname,**kwargs):
    """
      Writes this data object to CSV file (except the general metadata, see _toCSVXML)
      @ In, fname, str, path/name to write file
      @ In, kwargs, dict, optional, keywords for options
      @ Out, None
    """
    # TODO only working for point sets
    self.asDataset()
    filenameLocal = fname # TODO path?
    keep = self._getRequestedElements(kwargs)
    data = self._data
    for var in self._allvars:
      if var not in keep:
        data = data.drop(var)
    self.raiseADebug('Printing data to CSV: "{}"'.format(filenameLocal+'.csv'))
    # get the list of elements the user requested to write
    # make a pandas dataframe, they write to CSV very well
    data = data.to_dataframe()
    # order data according to user specs # TODO might be time-inefficient, allow user to skip
    ordered = list(i for i in self._inputs if i in keep)
    ordered += list(o for o in self._outputs if o in keep)
    ordered += list(m for m in self._metavars if m in keep)
    data = data[ordered]
    # write CSV; changes depending on if sampleTag is kept or not.
    # TODO sampleTag is critical for reading time histories
    if self.sampleTag not in keep:
      # keep other indices if multiindex
      if isinstance(data.index,pd.MultiIndex):
        data.index = data.index.droplevel(self.sampleTag)
        data.to_csv(filenameLocal+'.csv')#,index=False)
      # otherwise just don't print index
      else:
        data.to_csv(filenameLocal+'.csv',index=False)
    else:
      data.to_csv(filenameLocal+'.csv')

  def _toCSVXML(self,fname,**kwargs):
    """
      Writes the general metadata of this data object to XML file
      @ In, fname, str, path/name to write file
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
        dimsNode.remove(child)
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

    self.raiseADebug('Printing metadata XML: "{}"'.format(fname+'.xml'))
    with file(fname+'.xml','w') as ofile:
      #header
      ofile.writelines('<DataObjectMetadata name="{}">\n'.format(self.name))
      for name,target in meta.items():
        xml = target.writeFile(asString=True,startingTabs=1,addRavenNewlines=False)
        ofile.writelines('  '+xml+'\n')
      ofile.writelines('</DataObjectMetadata>\n')

  def _toNetCDF(self,fname,**kwargs):
    """
      Writes this data object to file in netCDF4.
      @ In, fname, str, path/name to write file
      @ In, kwargs, dict, optional, keywords to pass to netCDF4 writing
                                    One good option is format='NETCDF4' to assure netCDF4 is used
                                    See http://xarray.pydata.org/en/stable/io.html#netcdf for options
      @ Out, None
    """
    # TODO set up to use dask for on-disk operations -> or is that a different data object?
    self.asDataset() #just in case there is stuff left in the collector
    # convert metadata into writeable
    self._data.attrs = dict((key,pk.dumps(val)) for key,val in self._meta.items())
    self._data.to_netcdf(fname,**kwargs)
