import sys,os
import __builtin__
import functools

import abc
import numpy as np
import pandas as pd
import xarray as xr

from BaseClasses import BaseType
from utils import utils, cached_ndarray, InputData, xmlUtils

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
    inputSpecification.addParam('type', param_type = InputData.StringType, required = False)
    inputSpecification.addSub(InputData.parameterInputFactory('Input',contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory('Output',contentType=InputData.StringType))
    return inputSpecification
    # TODO on-disk, etc

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

    self._data        = None   # underlying data structure
    self._collector   = None   # object used to collect samples
    self._collectMeta = {}     # dictionary to collect meta until data is collapsed
    self._heirarchal  = False  # if True, non-traditional format (not yet implemented)

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
      if child.getName() == 'Input':
        self._inputs.extend(list(x for x in child.value.split(',')))
      elif child.getName() == 'Output':
        self._outputs.extend(list(x for x in child.value.split(',')))
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
  def addMeta(self,**kwargs):
    """
      Adds general (not pointwise) metadata to this data object.  If _data is None, collects it locally
      until conversion.  Otherwise, directly updates _data attributes.
      @ In, kwargs, dict, {xpath:value} pairs to add to metadata.  "value" should be string-like
      @ Out, None
    """
    if self._data is None:
      destination = self._collectMeta
    else:
      destination = self._data.attrs
    for key,val in kwargs.items():
      # TODO check valid xpath for key?
      assert(isinstance(val,(str,unicode))) # FIXME convert through repr instead?
      if key in destination.keys():
        # if the same entry is already there, don't replace it
        if val == destination[key]:
          continue
        else:
          self.raiseAWarning('Multiple general metadata have the same key:',key)
          # multiple entries is probably bad, but we can accomodate
          destination[key] += ','+val
      else:
        destination[key] = val

  def addRealization(self,rlz):
    """
      Adds a "row" (or "sample") to this data object.
      This is the preferred method to add data to this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" can be either a float (pointset) or xr.DataArray object (ndset)
      @ Out, None
    """
    # TODO more error check on realization length, contents
    # if collector/data not yet started, expand entries that aren't I/O as metadata
    if self._data is None and self._collector is None:
      unrecognized = set(rlz.keys()).difference(set(self._allvars))
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
        # first case: single entry per node: floats, strings, ints, etc
        if isinstance(data[0,v],(float,str,unicode,int)): #TODO expand the list of types as needed, isinstance means np.float64 is covered by float
          arrs[var] = xr.DataArray(data[:,v],
                                   dims=[self.sampleTag],
                                   coords={self.sampleTag:range(len(data))},
                                   name=var) # THIS is very fast
        # second case: ND set (history set or higher dimension)
        elif type(data[0,v]) == xr.DataArray:
          # two methods: all at "once" or "split" into multiple parts.  "once" is faster, but not parallelizable.
          # ONCE #
          if method == 'once':
            val = dict((i,data[i,v]) for i in range(len(data)))
            val = xr.Dataset(data_vars=val)
            val = val.to_array(dim=self.sampleTag)
          # SPLIT # currently unused, but could be for parallel performance
          elif method == 'split':
            chunk = 150
            start = 0
            N = len(data)
            vals = []
            # TODO can be parallelized
            while start < N-1:
              stop = min(start+chunk+1,N)
              ival = dict((i,data[i,v]) for i in range(start,stop))
              ival = xr.Dataset(data_vars=ival)
              ival = ival.to_array(dim=self.sampleTag) # TODO does this end up indexed correctly?
              vals.append(ival)
              start = stop
            val = xr.concat(vals,dim=self.sampleTag)
          # END #
          arrs[var] = val
        else:
          raise IOError('Unrecognized data type for var "{}": "{}"'.format(var,type(data[0,v])))
        # re-index samples
        arrs[var][self.sampleTag] += firstSample
        arrs[var].rename(var)
      # collect all data into dataset, and update self._data
      new = xr.Dataset(arrs)
      if self._data is None:
        self._data = new
        # general metadata included if first time
        self._data.attrs.update(self._collectMeta)
        # clear meta collector
        self._collectMeta = {}
        # store sample tag
        self.addMeta(sampleTag=self.sampleTag)
      else:
        # TODO compatability check!
        self._data.merge(new,inplace=True)
      # reset collector
      self._collector = cached_ndarray.cNDarray(width=self._collector.width,dtype=self._collector.values.dtype)
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
       @ In, keys, list(str), optional, the keys (or XPath) to search for.  If None, return all.
       @ In, pointwise, bool, optional, if True then matches will be searched in the pointwise metadata
       @ In, general, bool, optional, if True then matches will be searched in the general metadata
       @ Out, meta, dict, key variables/xpaths to data object entries (column if pointwise, XML if general)
    """
    meta = {}
    if pointwise:
      # TODO slow key crawl
      for var in self._metavars:
        if keys is None or var in keys:
          # TODO if still collecting, an option to NOT freeze
          meta[var] = self.asDataset()[var]#[self._allvars.index(var),:]
    if general:
      if keys is None and attrib:
        return self._data.
    # TODO error on missing matches
    # TODO if only one variable requested, return values directly
    #if len(meta) == 1:
    #  return meta.values()
    # otherwise, return dictionary
    #else:
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
    for dim in res.dims:
      res = res.dropna(dim)
    return self._data[var]

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
    # TODO convert input space to KD tree for faster searching
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
    if style.lower() == 'netcdf':
      self._fromNetCDF(fname,**kwargs)
    # TODO CSV
    # TODO dask
    else:
      self.raiseAnError(NotImplementedError,'Unrecognized read style: "{}"'.format(style))

  def remove():
    raise NotImplementedError

  def reset():
    raise NotImplementedError

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
    # TODO CSV in its variety
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
    # TODO this could be slow, should do KD tree instead
    mask = list(self._data[var] == val for var,val in match.items())[0]#.values
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

  def _fromCSV(self,fname,**kwargs):
    raise NotImplementedError

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

  def _toCSV(self,fname,**kwargs):
    """
      Writes this data object to CSV/XML coupled files
      @ In, fname, str, path/name to write file
      @ In, kwargs, dict, optional, keywords to pass to CSV writing (as per pandas.DataFrame.to_csv)
      @ Out, None
    """
    # TODO only working for point sets
    self.asDataset()
    filenameLocal = options.get('filenameroot','_dump')
    data = self._data
    if 'what' in options.keys():
      keep = list(v.split('|')[-1].strip() for v in options['what'].split(','))
    else:
      # BY DEFAULT only keep inputs, outputs; if specifically requested, keep metadata by selection
      keep = self._inputs + self._outputs
    for var in self._allvars:
      if var not in keep:
        data = data.drop(var)
    self.raiseADebug('Printing to CSV: "{}"'.format(filenameLocal))
    data = data.to_dataframe()
    if self.sampleTag not in keep:
      data.to_csv(filenameLocal+'.csv',index=False)
    else:
      data.to_csv(filenameLocal+'.csv')
    # general XML
    tree = xmlUtils.newTree('Metadata')
    root = tree.getroot()
    for attrib,val in self.getMeta(general=True).items():
      path = attrib.split('/')

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
    self._data.to_netcdf(fname,**kwargs)


  ############################################################################################
  ### LEGACY API
  ############################################################################################
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

  def isItEmpty(self):
    """
      Determines if any samples have been taken.
      @ In, None
      @ Out, empty, bool, True if no samples have been taken
    """
    self.deprecated('isItEmpty')
    return True if self.size == 0 else False

  def updateInputValue(self,name,value,options=None):
    """
      Function to update a value from the input dictionary
      @ In, name, string, parameter name
      @ In, value, float, the new value
      @ In, options, dict, optional, the dictionary of options to update the value (e.g. parentId, etc.)
      @ Out, None
    """
    self.deprecated('updateInputValue')
    if self._collector is None:
      self._collector = cached_ndarray.cNDarray(width = len(self.vars),length=4,dtype=object)
      self._collector.size = 1
      self._collector.width = len(self.vars)
    #try:
    column = self._getVariableIndex(name)
    self._collector._addOneEntry(column,value[0])
    if False:
    #except ValueError:
        #self._data._addOneEntry(column,value[0])
        #self._data.addEntity(np.array([value]), firstEver = True)
      self._collector.addEntity([np.array([value])])
      # FIXME this could be a costly check (not necessary in non-deprecated API)
      if name not in self._inputs:
        self._allvars.append(name)
        self._inputs.append(name)

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
    if name in ['SamplerType','crowDist']:
      kwargs = {name:value}
      self.addMeta(**kwargs)
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

  def printCSV(self,options=None):
    """
      Dump to CSV
      @ In, options, dict, optional, dictionary of options such as filename, parameters, etc
      @ Out, None
    """
    self.deprecated('printCSV')
    if options is None:
      options = {}
    self.toCSV

  def loadXMLandCSV(self,fpath,options):
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

    print('fname:',fname)
    data = pd.read_csv(fname+'.csv')
    print(data)
    import sys;sys.exit()
