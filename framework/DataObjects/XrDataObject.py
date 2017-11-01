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
class DataObjectsCollection(InputData.ParameterInput):
  """
    Class for reading in a collection of data objects.
  """
DataObjectsCollection.createClass("DataObjects")

class DataObject(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  """
    Base class.  Data objects are RAVEN's method for storing data internally and passing it from one
    RAVEN entity to another.  Fundamentally, they consist of a collection of realizations, each of
    which contains inputs, outputs, and pointwise metadata.  In addition, the data object has global
    metadata.  The pointwise inputs and outputs could be floats, time-dependent, or ND-dependent variables.
  """
  ### INPUT SPECIFICATION ###
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    inputSpecification = super(DataObject,cls).getInputSpecification()
    inputSpecification.addParam('hierarchical',InputData.StringType)
    inputSpecification.addParam('inputTs',InputData.StringType)
    inputSpecification.addParam('historyName',InputData.StringType)

    inputInput = InputData.parameterInputFactory('Input',contentType=InputData.StringType) #TODO list
    inputSpecification.addSub(inputInput)

    outputInput = InputData.parameterInputFactory('Output', contentType=InputData.StringType) #TODO list
    inputSpecification.addSub(outputInput)

    optionsInput = InputData.parameterInputFactory('options')
    for option in ['inputRow','inputPivotValue','outputRow','outputPivotValue','operator','pivotParameter']:
      optionSubInput = InputData.parameterInputFactory(option,contentType=InputData.StringType)
      optionsInput.addSub(optionSubInput)
    # TODO "operator" has finite options (max, min, average)
    inputSpecification.addSub(optionsInput)

    #inputSpecification.addParam('type', param_type = InputData.StringType, required = False)
    #inputSpecification.addSub(InputData.parameterInputFactory('Input',contentType=InputData.StringType))
    #inputSpecification.addSub(InputData.parameterInputFactory('Output',contentType=InputData.StringType))
    #inputSpecification.addSub(InputData.parameterInputFactory('options',contentType=InputData.StringType))
    return inputSpecification

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)
    self._inputs   = []     # list(str) if input variables
    self._outputs  = []     # list(str) of output variables
    self._metavars = []     # list(str) of POINTWISE metadata variables
    self._allvars  = []     # list(str) of vars IN ORDER of their index

    self._data         = None   # underlying data structure
    self._collector    = None   # object used to collect samples
    self._meta         = {}     # dictionary to collect meta until data is collapsed
    self._heirarchal   = False  # if True, non-traditional format (not yet implemented)
    self._selectInput  = None   # if not None, describes how to collect input data from history
    self._selectOutput = None   # if not None, describes how to collect output data from history
    self._pivotParam   = 'time' # FIXME should deprecate or expand for ND; pivot parameter for data selection
    self._aliases      = {}     # variable aliases

    self.name      = 'BaseDataObject'
    self.printTag  = self.name
    self.sampleTag = 'RAVEN_sample_ID' # column name to track samples

  def _readMoreXML(self,xmlNode):
    """
      Initializes data object based on XML input
      @ In, xmlNode, xml.etree.ElementTree.Element or InputData.ParameterInput specification, input information
      @ Out, None
    """
    if isinstance(xmlNode,InputData.ParameterInput):
      inp = xmlNode
    else:
      inp = DataObject.getInputSpecification()()
      inp.parseNode(xmlNode)
    for child in inp.subparts:
      # TODO check for repeats, "notAllowdInputs", names in both input and output space
      if child.getName() == 'Input':
        self._inputs.extend(list(x for x in child.value.split(',') if x.strip()!=''))
      elif child.getName() == 'Output':
        self._outputs.extend(list(x for x in child.value.split(',') if x.strip()!=''))
      # options node
      elif child.getName() == 'options':
        duplicateInp = False # if True, then multiple specification options were used for input
        duplicateOut = False # if True, then multiple specification options were used for output
        for cchild in child.subparts:
          # pivot
          if cchild.getName() == 'pivotParameter':
            self._pivotParam = cchild.value.strip()
          # input pickers
          elif cchild.getName() == 'inputRow':
            if self._selectInput is None:
              self._selectInput = ('inputRow',int(cchild.value))
            else:
              duplicateInp = True
          elif cchild.getName() == 'inputPivotValue':
            if self._selectInput is None:
              self._selectInput = ('inputPivotValue',float(cchild.value))
            else:
              duplicateInp = True
          # output pickers
          elif cchild.getName() == 'outputRow':
            if self._selectOutput is None:
              self._selectOutput = ('outputRow',int(cchild.value))
            else:
              duplicateOut = True
          elif cchild.getName() == 'operator':
            if self._selectOutput is None:
              self._selectOutput = ('operator',cchild.value.strip().lower())
            else:
              duplicateOut= True
          elif cchild.getName() == 'outputPivotValue':
            if self._selectOutput is None:
              self._selectOutput = ('outputPivotValue',float(cchild.value)) # TODO HistSet can be list of floats
            else:
              duplicateOut = True
        # TODO check this in the input checker instead of here?
        if duplicateInp:
          self.raiseAnError(IOError,'Multiple options were given to specify the input row to read!  Please choose one.')
        if duplicateOut:
          self.raiseAnError(IOError,'Multiple options were given to specify the output row to read!  Please choose one.')
      # end options node
    # end input reading
    self._allvars = self._inputs + self._outputs

#
#
#
#
class DataSet(DataObject):
  """
    DataObject developed Oct 2017 to obtain linear performance from data objects when appending, over
    thousands of variables and millions of samples.  Wraps np.ndarray for collecting and uses xarray.Dataset
    for final form.
  """
  ############################################################################################
  ### NEW API
  ############################################################################################

  ### EXTERNAL API ###
  # These are the methods that RAVEN entities should call to interact with the data object

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
      This is the preferred method to add data to this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" can be either a float (pointset) or xr.DataArray object (ndset)
      @ Out, None
    """
    # first, update realization with selectors
    rlz = self._selectiveRealization(rlz)
    # TODO more error check on realization length, contents
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
      self._collector = cached_ndarray.cNDarray(width=len(rlz),dtype=object)
    self._collector.append(newData)

  def asDataset(self):
    """
      Casts this dataobject as an xr.Dataset.
      Functionally, typically collects the data from self._collector and places it in self._data.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, xarray.Dataset, all the data from this data object.
    """
    # if we have collected data, collapse it
    if self._collector is not None and len(self._collector) > 0:
      data = self._collector.getData()
      method = 'once' # internal flag to switch method.  "once" is generally faster, but "split" can be parallelized.
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
      self._collector = cached_ndarray.cNDarray(width=self._collector.width,length=10,dtype=self._collector.values.dtype)
    return self._data

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

  def getOutputs():
    raise NotImplementedError

  def getVarValues(self,var):
    """
      Returns the sampled values of "var"
      @ In, var, str, name of variable
      @ Out, res, xr.DataArray, samples
    """
    # TODO have to convert here?
    self.asDataset()
    res = self._data[var]
    #for dim in res.dims:
    #  res = res.dropna(dim)
    return res

  def realization(self,index=None,matchDict=None,readCollector=False):
    """
      Method to obtain a realization from the data, either by index or matching value.
      Either "index" or "matchDict" must be supplied.
      If matchDict and no match is found, will return (len(self),None) after the pattern of numpy, scipy
      @ In, index, int, optional, number of row to retrieve (by index, not be "sample")
      @ In, matchDict, dict, optional, {key:val} to search for matches
      @ In, readCollector, bool, if True then read out of collector instead of data
      @ Out, index, int, optional, index where found (or len(self) if not found), only returned if matchDict
      @ Out, rlz, dict, realization requested (None if not found)
    """
    # TODO convert input space to KD tree for faster searching -> XArray.DataArray has this built in
    # TODO option to read both collector and data for matches/indices
    if (index is None and matchDict is None) or (index is not None and matchDict is not None):
      self.raiseAnError(TypeError,'Either "index" OR "matchDict" (not both) must be specified to use "realization!"')
    if index is not None:
      if readCollector:
        rlz = self._getRealizationFromCollectorByIndex(index)
      else:
        rlz = self._getRealizationFromDataByIndex(index)
      return rlz
    else: #because of check above, this means matchDict is not None
      if readCollector:
        index,rlz = self._getRealizationFromCollectorByValue(matchDict)
      else:
        index,rlz = self._getRealizationFromDataByValue(matchDict)
      return index,rlz

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

  def remove():
    raise NotImplementedError

  def reset(self):
    """
      Sets this object back to its initial state.
      @ In, None
      @ Out, None
    """
    self._data = None
    self._collector = None
    self._meta = None
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
    """
    DataObject.__init__(self)
    self.name      = 'DataSet'
    self.printTag  = self.name

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
    # second case: ND set (history set or higher dimension)
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
      self.raiseAnError(TypeError,'Unrecognized data type for var "{}": "{}"'.format(var,type(data[0,v])))
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
    new = xr.Dataset(array)
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

  def _fromCSV(self,fname,**kwargs):
    """
      Loads a dataset from CSV (preferably one it wrote itself, but maybe not necessarily?
      @ In, fname, str, filename to load from (not including .csv or .xml)
      @ In, kwargs, dict, optional arguments
      @ Out, None
    """
    assert(self._data is None)
    assert(self._collector is None)
    # first, read in the XML
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
      try:
        self.sampleTag = xmlUtils.findPath(meta,'DataSet/general/sampleTag').text
      except TypeError:
        pass # use default
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
    # read from csv
    panda = pd.read_csv(fname+'.csv')
    self.raiseADebug('Reading data from "{}.csv"'.format(fname))
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

  def _getRequestedElements(self,options):
    """
      Obtains a list of the elements to be written, based on defaults and options[what]
      @ In, options, dict, general list of options for writing output files
      @ Out, keep, list(str), list of variables that will be written to file
    """
    if 'what' in options.keys():
      keep = list(v.split('|')[-1].strip() for v in options['what'].split(','))
    else:
      # TODO need the sampleTag meta to load histories # BY DEFAULT only keep inputs, outputs; if specifically requested, keep metadata by selection
      keep = self._inputs + self._outputs
    return keep

  def _getRealizationFromCollectorByIndex(self,index):
    """
      Obtains a realization from the collector storage using the provided index.
      @ In, index, int, index to return
      @ Out, rlz, dict, realization as {var:value}
    """
    assert(self._collector is not None)
    assert(index < len(self._collector))
    return dict(zip(self._allvars,self._collector[index]))

  def _getRealizationFromCollectorByValue(self,match):
    """
      Obtains a realization from the collector storage matching the provided index
      @ In, match, dict, elements to match
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
          # TODO use math util
          # TODO arbitrary tolerance
          match *= mathUtils.compareFloats(lookingFor[e],element,tol=1e-10)
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
    assert(index < len(self._data[self.sampleTag]))
    rlz = self._data[{self.sampleTag:index}].drop(self.sampleTag).data_vars
    rlz = self._convertFinalizedDataRealizationToDict(rlz)
    return rlz

  def _getRealizationFromDataByValue(self,match):
    """
      Obtains a realization from the data storage using the provided index.
      @ In, match, dict, elements to match
      @ Out, r, int, index where match was found OR size of data if not found
      @ Out, rlz, dict, realization as {var:value} OR None if not found
    """
    assert(self._data is not None)
    #print('*'*80)
    #print('Looking for',match)
    #print(self._data)
    # TODO this could be slow, should do KD tree instead
    mask = 1.0
    for var,val in match.items():
      mask *= abs(self._data[var] - val) < 1e-10
    print('mask:')
    print(mask)
    rlz = self._data.where(mask,drop=True)
    try:
      idx = rlz[self.sampleTag].item(0)
    except IndexError:
      return len(self),None
    return idx,self._getRealizationFromDataByIndex(idx)

  def _getVariableIndex(self,var):
    """
      Obtains the index in the list of variables for the requested var.
      @ In, var, str, variable name (input, output, or pointwise medatada)
      @ Out, index, int, column corresponding to the variable
    """
    return self._allvars.index(var)

  def _selectiveRealization(self,rlz):
    """
      Uses "options" parameters from input to select part of the collected data
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    # TODO this would be much more efficient on the parallel (finalizeCodeOutput) than on serial
    # TODO costly for loop
    for var,val in rlz.items():
      # only modify it if it 1) isn't already scalar, 2) there is a method given, and 3) inp/out classifier
      if not isinstance(val,float) and self._selectInput is not None and var in self._inputs:
        method,indic = self._selectInput
        if method == 'inputRow':
          rlz[var] = float(val[:,indic]) # TODO testme, also TODO don't case to float?
        elif method == 'inputPivotValue':
          rlz[var] = float(val.sel(**{self._pivotParam:indic, 'method':'nearest'}))
      elif not isinstance(val,float) and self._selectOutput is not None and var in self._outputs:
        method,indic = self._selectOutput
        if method == 'outputRow':
          rlz[var] = float(val[:,indic]) # TODO testme, also TODO don't case to float?
        elif method == 'outputPivotValue':
          rlz[var] = float(val.sel(**{self._pivotParam:indic, 'method':'nearest'}))
        elif method == 'operator':
          if indic == 'max':
            rlz[var] = float(val.max())
          elif indic == 'min':
            rlz[var] = float(val.min())
          elif indic in ['mean','expectedValue','average']:
            rlz[var] = float(val.mean())
      # otherwise, leave it alone
    return rlz

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


  ############################################################################################
  ### LEGACY API
  ############################################################################################
  def addOutput(self,toLoadFrom,options=None):
    """
      Function to construct a data from a source.
      @ In, toLoadFrom, string, loading source (hdf5, csv)
      @ In, options, dict, optional, options such as metadata or heirarchal information
      @ Out, None
    """
    self.deprecated('addOutput')
    if options is None:
      options = {}
    dataParams = {'inParam'       : self._inputs,
                  'outParam'      : self._outputs,
                  'pivotParameter': self._pivotParam,
                  'type'          : 'PointSet', # TODO Faking it
                  'HistorySet'    : toLoadFrom.getEndingGroupNames(),
                  'filter'        : 'whole',
                 }
    if self._selectInput is not None:
      if self._selectInput[0] == 'inputRow':
        dataParams['inputRow'] = self._selectInput[1]
      elif self._selectInput[0] == 'inputPivotValue':
        dataParams['inputPivotValue'] = self._selectInput[1]

    if self._selectOutput is not None:
      if self._selectOutput[0] == 'outputRow':
        dataParams['outputRow'] = self._selectOutput[1]
      elif self._selectOutput[0] == 'outputPivotValue':
        dataParams['outputPivotValue'] = self._selectOutput[1]
      elif self._selectOutput[0] == 'operator':
        dataParams['operator'] = self._selectOutput[1]
    self._aliases = options.get('alias',{})
    loadType = toLoadFrom.type
    self.raiseAMessage('Loading from "{}" which is a "{}"'.format(toLoadFrom.name,loadType))

    if loadType == 'HDF5':
      tupleVar = toLoadFrom.retrieveData(dataParams)
    # TODO Files.File (csv)

    # tupleVar is ({inputs},{outputs},{meta})
    # get number of realizations from input space
    numEntries = len(tupleVar[0].values()[0])
    for i in range(numEntries):
      rlz = {}
      for var in self._inputs:
        rlz[var] = tupleVar[0][var][i]
      for var in self._outputs:
        rlz[var] = tupleVar[1][var][i]
      # TODO skip meta for now
      # TODO need to modify form of time-dependent entries
      self.addRealization(rlz)

  def getAllMetadata(self,nodeId=None,serialize=False):
    """
      Function to get all the metadata
      @ In, nodeId, str, optional, id of the node if hierarchal
      @ In, serialize, bool, optional, serialize the tree if in hierarchal mode
      @ Out, dictionary, dict, return the metadata dictionary
    """
    self.deprecated('getAllMetadata')
    return self.getMeta(pointwise=True,general=True)

  def getInpParametersValues(self,nodeId=None,serialize=False,unstructuredInputs=False):
    """
      Function to get a reference to the input parameter dictionary
      @, In, nodeId, string, optional, in hierarchical mode, if nodeId is provided, the data for that node is returned,
                                  otherwise check explanation for getHierParam
      @ In, serialize, bool, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                  PLEASE check explanation for getHierParam
      @ In, unstructuredInputs, bool, optional, True if the unstructured input space needs to be returned
      @, Out, dictionary, dict, Reference to self._dataContainer['inputs'] or something else in hierarchical
    """
    self.deprecated('getInpParametersValues')
    res = dict((var,self.getVarValues(var)) for var in self._inputs)
    for key, val in res.items():
      res[key] = val.values
    return res

  def getMatchingRealization(self,requested,tol=1e-15):
    """
      Finds first appropriate match within tolerance and return it.
      @ In, requested, dict, {var:val}
      @ In, tol, float, relative tolerance
      @ Out, realization, dict, match
    """
    self.deprecated('getMatchingRealization')
    self.asDataset()
    idx,match = self.realization(matchDict=requested)
    if match is None:
      # no match found
      return None
    realization = {'inputs':{},'outputs':{}}
    for key in self._inputs:
      realization['inputs'][key] = [match[key]]
    for key in self._outputs:
      realization['outputs'][key] = [match[key]]
    return realization

  def getOutParametersValues(self,nodeId=None,serialize=False):
    """
      Function to get a reference to the output parameter dictionary
      @, In, nodeId, string, optional, in hierarchical mode, if nodeId is provided, the data for that node is returned,
                                  otherwise check explanation for getHierParam
      @ In, serialize, bool, optional, in hierarchical mode, if serialize is provided and is true a serialized data is returned
                                  PLEASE check explanation for getHierParam
      @ In, unstructuredInputs, bool, optional, True if the unstructured input space needs to be returned
      @, Out, dictionary, dict, Reference to self._dataContainer['inputs'] or something else in hierarchical
    """
    self.deprecated('getInpParametersValues')
    return dict((var,self.getVarValues(var)) for var in self._outputs)

  def getParaKeys(self,typePara):
    """
      Function to get the parameter keys
      @ In, typePara, string, variable type (input, output, or metadata)
      @ Out, keys, list, list of requested keys
    """
    self.deprecated('getParaKeys')
    typePara = typePara.strip().lower().rstrip('s')
    if typePara in ['input','inp']:
      return self.getVars('input')
    elif typePara in ['output','out']:
      return self.getVars('output')
    elif typePara in ['meta','metadata']:
      return self.getVars('meta')

  def getParam(self,typeVar,keyword,**kwargs):
    """
      Gets a reference to an input or output parameter.
      @ In, typeVar, str, var in 'input', 'unstructuredInput', 'output'
      @ In, keyword, string, keyword
      @ In, kwargs, dict, arbitrary additional keywords
      @ Out, getParam, list, reference to parameter
    """
    return np.asarray(self.getVarValues(keyword))

  def getParametersValues(self,typeVar,*args,**kwargs):
    """
      Gets parameter values.
      @ In, typeVar, string, variable type (input, unstructuredInput, output)
      @ In, args, list, unneeded compatability args
      @ In, kwargs, dict, unneeded compatability args
      @ Out, dictionary, dict, dict of parmater values
    """
    typeVar = typeVar.lower()
    if typeVar in 'inputs':
      return self.getInpParametersValues()
    elif typeVar in 'unstructuredinputs':
      return self.getInpParametersValues(unstructuredInputs=True)
    elif typeVar in 'outputs':
      return self.getOutParametersValues()
    else:
      self.raiseAnError(RuntimeError, 'type "{}" is not a valid type.'.format(typeVar))

  def getRealization(self,index):
    """
      Returns the indexed entry of inputs and outputs
      @ In, index, int, index to retrieve
      @ Out, realization, dict {'inputs':{var:val},'outputs':{var:val}}
    """
    self.deprecated('getRealization')
    self.asDataset()
    match = self.realization(index=index)
    realization = {'inputs':{},'outputs':{}}
    for key in self._inputs:
      realization['inputs'][key] = [match[key]]
    for key in self._outputs:
      realization['outputs'][key] = [match[key]]
    return realization

  def isItEmpty(self):
    """
      Determines if any samples have been taken.
      @ In, None
      @ Out, empty, bool, True if no samples have been taken
    """
    self.deprecated('isItEmpty')
    return True if self.size == 0 else False

  def loadXMLandCSV(self,fpath,options=None):
    """
      Function to load the xml additional file of the csv for data
      @ In, fpath, str, file name root
      @ In, options, dict, dictionary with loading options
      @ Out, None
    """
    self.deprecated('loadXMLandCSV')
    if options is not None and 'fileToLoad' in options.keys():
      name = os.path.join(options['fileToLoad'].getPath(),options['fileToLoad'].getBase())
    else:
      name = self.name
    fname = os.path.join(fpath,name)
    self.load(fname,style='CSV')

  def getNumAdditionalLoadPoints(self):
    """
      Tracks the number of expected samples in the set.
      @ In, None
      @ Out, num, int, points
    """
    self.deprecated('getNumAdditionalLoadPoints')
    return 0 #not needed

  def setNumAdditionalLoadPoints(self,value):
    """
      Tracks the number of expected samples in the set.
      @ In, value, int, new value
      @ Out, None
    """
    self.deprecated('setNumAdditionalLoadPoints')
    return #not needed

  numAdditionalLoadPoints = property(getNumAdditionalLoadPoints,setNumAdditionalLoadPoints)

  def printCSV(self,options=None):
    """
      Dump to CSV
      @ In, options, dict, optional, dictionary of options such as filename, parameters, etc
      @ Out, None
    """
    self.deprecated('printCSV')
    if options is None:
      options = {}
    fname = options.pop('filenameroot',self.name+'_dump')
    self.write(fname,style='CSV',**options)

  def resetData(self):
    """
      Resets.
      @ In, None
      @ Out, None
    """
    self.reset()

  def updateInputValue(self,name,value,options=None):
    """
      Function to update a value from the input dictionary
      @ In, name, string, parameter name
      @ In, value, float, the new value
      @ In, options, dict, optional, the dictionary of options to update the value (e.g. parentId, etc.)
      @ Out, None
    """
    print('DEBUGG updating {} with {}'.format(name,value))
    self.deprecated('updateInputValue')
    if self._collector is None or len(self._collector)==0:
      self._collector = cached_ndarray.cNDarray(width = len(self.vars),length=4,dtype=object)
      self._collector.size = 1
      self._collector.width = len(self.vars)
    column = self._getVariableIndex(name)
    try:
      self._collector._addOneEntry(column,value[0]) #sometimes "value" is just a scalar though
    except IndexError:
      self._collector._addOneEntry(column,value)
    except TypeError:
      self._collector._addOneEntry(column,value)

  def updateMetadata(self,name,value,options=None):
    """
      Function to update a value from the dictionary metadata
      @ In, name, string, parameter name
      @ In, value, float, the new value
      @ In, options, dict, optional, dictionary of options
      @ Out, None
    """
    self.deprecated('updateMetadata')
    # global
    if name in ['SamplerType','crowDist'] and len(self)<2:
      meta = {'general':{name:value}}
      self.addMeta('Sampler',meta)
    # pointwise
    elif name in ['ProbabilityWeight','prefix']: #TODO only add prefix if it's needed, don't default
      try:
        column = self._getVariableIndex(name)
        self._collector._addOneEntry(column,value)
      except ValueError:
        self._collector.addEntity([np.array([[value]])]) #WHY should there be an extra [] in here....
        # FIXME this could be a costly check (not necessary in non-deprecated API)
        if name not in self._metavars:
          self._allvars.append(name)
          self._metavars.append(name)
    # unneeded
    elif name in ['SampledVarsPb','PointProbability']:
      pass

  def updateOutputValue(self,name,value,options=None):
    """
      Function to update a value from the output dictionary
      @ In, name, string, parameter name
      @ In, value, float, the new value
      @ In, options, dict, optional, the dictionary of options to update the value (e.g. parentId, etc.)
      @ Out, None
    """
    self.deprecated('updateOutputValue')
    try:
      column = self._getVariableIndex(name)
      self._collector._addOneEntry(column,value)
    except ValueError:
      self._collector.addEntity([np.array([[value]])]) #WHY should there be an extra [] in here....
      # FIXME this could be a costly check (not necessary in non-deprecated API)
      if name not in self._outputs:
        self._allvars.append(name)
        self._outputs.append(name)

