import sys,os
import __builtin__
import functools

import abc
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as ncDS

from BaseClasses import BaseType
from utils import utils, cached_ndarray, InputData

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
    self._inputs     = []     # list(str) if input variables
    self._outputs    = []     # list(str) of output variables
    self._metavars   = []     # list(str) of POINTWISE metadata variables
    self._allvars    = []     # list(str) of vars IN ORDER of their index

    self._data       = None   # underlying data structure
    self._collector  = None   # object used to collect samples
    self._heirarchal = False  # if True, non-traditional format (not yet implemented)

    self.name        = 'BaseDataObject'
    self.printTag    = self.name

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
  # API TRANSLATION
  #  OLD                               |    NEW
  # addOutput                          | ? load from values
  # getAllMetadata                     | ? -remove-
  # getHierParam                       | ? heirarchal only
  # getInitParams                      | ? useful?
  # getInpParametersValues             | ? getVarValues
  # getMatchingRealization             | ? getRealization
  # getMetadata                        | ? getMeta
  # getOutParametersValues             | ? getVarValues
  # getParaKeys                        | ? getInputs, getOutputs, getMetaKeys
  # getParam                           | ? getVarValues
  # getParametersValues                | ? getInputs, getOutputs, getMeta
  # getRealization                     | ? by index, by value, also asDataset or NOT (for reading)
  # isItEmpty                          | ? size
  # loadXMLandCSV                      | ? loadFromCSV
  # printCSV                           | ? writeCSV
  # _writeUnstructuredInputInXML       | ? writeMetaXML
  # remoteInputValue                   | ? removeVariable
  # removeOutputValue                  | ? removeVariable
  # resetData                          | ? reset
  # retrieveNodeInTreeMode             | ? hierarchal only
  # sizeData                           | ? size
  # updateInputValue                   | addRealization
  # updateOutputValue                  | addRealization
  # updateMetadata                     | addRealization, addGlobalMeta
  # addNodeInTreeMode                  | ? hierarchal only
  # _createXMLFile                     | ? writeMetaXML
  # _loadXMLFile                       | ? readMetaXML
  # _readMoreXML                       | same
  # _specializedInputCheck             | ? remove
  # _specializedLoadXMLandCSV          | ? loadFromCSV
  # __getVariablesToPrint              | ? remove
  # __getMetadataType                  | ? remve
  #
  #
  # BUILTINS AND PROPERTIES
  # size (property)         # number of samples
  # shape (property)        # dimensionality as (num samples, num entities)
  # _readMoreXML            # initialization from input
  # _firstSampleInit        # initialization once one sample is obtained
  #
  # SUMMARY list of new, API only:
  # addMeta                 # adds general meta (not pointwise)
  # addRealization          # add a single row
  # getInputs               # only var names
  # getMeta                 # column values in global or point
  # getOutputs              # only var names
  # getVarValues            # column values; input, output, or meta values for a variable
  # getRealization          # by keyed values or by index
  # load                    # reads netCDF, CSVs in all their variety, np.ndarrays, dicts?
  # remove                  # removes by index, value matching, or metadata matching; or variable
  # reset                   # empties data object
  # write                   # writes CSVs in all their variety
  #
  # HELPER FUNCTIONS
  # _asDataset              # if needed, converts data to finalized storage type
  # _getMetaPointwise       # get column of values from metadata
  #     -->                 # also option for reading non-finalized results
  # _getMetaGeneral         # get specific values from XML metadata
  #     -->                 # also option for reading non-finalized results
  # _getRealizationByIndex  # obtains "ith" sample row
  #     -->                 # also option for reading non-finalized results
  # _getRealizationByValue  # obtains row with matching data
  #     -->                 # also option for reading non-finalized results
  # _getRealizationByMeta   # obtains row with matching metadata (pointwise)
  #     -->                 # also option for reading non-finalized results
  # _loadCSV                # read in data from csv
  # _loadNetCDF             # read in netCDF
  # _loadValues             # read in np.ndarray, maybe dict?
  # _readMetaXML            # reads RAVEN-written XML general metadata
  # _writeCSV               # writes standard CSVs
  # _writeMetaXML           # writes meta XML to accompany CSV
  # _writeNdCSV             # writes higher-dimensional CSVs
  # _writeNetCDF            # writes netCDF

  ############################################################################################
  ### NEW API
  ############################################################################################

  ### EXTERNAL API ###
  def addMeta():
    raise NotImplementedError

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
    # check and order data to be stored
    try:
      newData = np.asarray([list(rlz[var] for var in self._allvars)],dtype=object)
    except KeyError as e:
      self.raiseAnError(KeyError,'Provided realization does not have all requisite values: "{}"'.format(e.args[0]))
    # if data storage isn't set up, set it up
    if self._collector is None:
      self._collector = cached_ndarray.cNDarray(width=len(rlz),dtype=object)
    self._collector.append(newData)

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
          meta[var] = self._asDataset()[var]#[self._allvars.index(var),:]
    if general:
      pass # TODO implement general meta
    # TODO error on missing matches
    # TODO if only one variable requested, return values directly
    #if len(meta) == 1:
    #  return meta.values()
    # otherwise, return dictionary
    #else:
    return meta

  def getOutputs():
    raise NotImplementedError

  def getVarValues():
    raise NotImplementedError

  def getRealization(self,index=None,matchDict=None,readCollector=False):
    """
      Method to obtain a realization from the data, either by index or matching value.
      Either "index" or "matchDict" must be supplied.
      @ In, index, int, optional, number of row to retrieve (by index, not be "sample")
      @ In, matchDict, dict, optional, {key:val} to search for matches
      @ In, readCollector, bool, if True then read out of collector instead of data
      @ Out, rlz, dict, realization requested (errors if not found)
    """
    # TODO convert input space to KD tree for faster searching
    # TODO option to read both collector and data for matches/indices
    if (index is None and matchDict is None) or (index is not None and matchDict is not None):
      self.raiseAnError(TypeError,'Either "index" OR "matchDict" (not both) must be specified to use "getRealization!"')
    if index is not None:
      if readCollector:
        rlz = self._getRealizationFromCollectorByIndex(index)
      else:
        rlz = self._getRealizationFromDataByIndex(index)
    else: #because of check above, this means matchDict is not None
      if readCollector:
        rlz = self._getRealizationFromCollectorByValue(matchDict)
      else:
        rlz = self._getRealizationFromDataByValue(matchDict)

  def load():
    raise NotImplementedError

  def remove():
    raise NotImplementedError

  def reset():
    raise NotImplementedError

  def write():
    raise NotImplementedError

  ### INITIALIZATION ###
  def __init__(self):#, in_vars, out_vars, meta_vars=None, dynamic=False, var_dims=None,cacheSize=100,prealloc=False):
    """
      Constructor.
    """
    DataObject.__init__(self)
    self.name      = 'DataSet'
    self.printTag  = self.name
    # change this line to change behavior of accessing the legacy API
    ## for warnings:
    self.deprecated = self._deprecatedWarning
    ## for errors:
    #self.deprecated = self._deprecatedError

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
  def __len__(self):
    """
      Overloads the len() operator.
      @ In, None
      @ Out, int, number of samples in this dataset
    """
    pass #TODO

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
    # TODO update for collected/finalized structures
    if type(self._data) == cached_ndarray.cNDarray:
      return self._data.size
    elif type(self._data) == xr.Dataset:
      return len(self._data['sample'])
    elif self._data is None:
      return 0
    else:
      self.raiseAnError(TypeError,'DataObject member "_data" is not a recognized type:',type(self._data))

  ### INTERNAL USE FUNCTIONS ###
  def _asDataset(self):
    """
      Casts this dataobject as an xr.Dataset.
      Functionally, typically collects the data from self._collector and places it in self._data.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, xarray.Dataset, all the data from this data object.
    """
    # FIXME for collector / data management system
    # if nothing to collect, do nothing TODO
    if type(self._data) != xr.Dataset:
      data = self._data.getData()
      method = 'once' # internal flag to switch method.  "once" is generally faster, but "split" can be parallelized.
      arrs = {}
      for v,var in enumerate(self._allvars):
        if isinstance(data[0,v],float) or isinstance(data[0,v],str):
          arrs[var] = xr.DataArray(data[:,v],
                                   dims=['sample'],
                                   coords={'sample':range(len(self._data))},
                                   name=var) # THIS is very fast
        elif type(data[0,v]) == xr.DataArray:
          # ONCE #
          if method == 'once':
            val = dict((i,data[i,v]) for i in range(len(self._data)))
            val = xr.Dataset(data_vars=val)
            val = val.to_array(dim='sample')
          # SPLIT # currently unused, but could be for parallel performance
          elif method == 'split':
            chunk = 150
            start = 0
            N = len(self._data)
            vals = []
            while start < N-1:
              stop = min(start+chunk+1,N)
              ival = dict((i,data[i,v]) for i in range(start,stop))
              ival = xr.Dataset(data_vars=ival)
              ival = ival.to_array(dim='sample')
              vals.append(ival)
              start = stop
            val = xr.concat(vals,dim='sample')
          # END #
          arrs[var] = val
          arrs[var].rename(var)
        else:
          raise IOError('Unrecognized data type for var "{}": "{}"'.format(var,type(data[0,v])))
      # FIXME currently MAKING not APPENDING!  This needs to be fixed.
      self._data = xr.Dataset(arrs)
    return self._data

  def _getRealizationFromCollectorByIndex(self,index):
    """
      Obtains a realization from the collector storage using the provided index.
      @ In, index, int, index to return
      @ Out, rlz, dict, realization as {var:value}
    """
    if self._collector is None:
      self.raiseAnError(IndexError,'Requested index "{}" but collector is empty!')
    if index >= len(self._collector):
      self.raiseAnError(IndexError,'Requested index "{}" but collector only has {} entries!'.format(len(self._collector)))

  def _getVariableIndex(self,var):
    """
      Obtains the index in the list of variables for the requested var.
      @ In, var, str, variable name (input, output, or pointwise medatada)
      @ Out, index, int, column corresponding to the variable
    """
    return self._allvars.index(var)

  def _toNetCDF4(self,fname,**kwargs):
    """
      Writes this data object to file in netCDF4.
      @ In, fname, str, path/name to write file
      @ In, kwargs, dict, optional, keywords to pass to netCDF4 writing
      @ Out, None
    """
    self.raiseADebug(' ... collecting dataset ...')
    self.asDataset()
    self.raiseADebug(' ... writing to file ...')
    self._data.to_netcdf(fname,**kwargs)


  ############################################################################################
  ### OLD API
  ############################################################################################
  def _deprecatedError(self,add=None):
    """
      Raises an error when deprecated methods are used.
      @ In, add, object, additional information to print
      @ Out, None
    """
    msg = 'Using DEPRECATED data object API!'
    if add is not None:
      msg += ' "{}"'.format(add)
    self.raiseAnError(SyntaxError,msg)

  def _deprecatedWarning(self,add=None):
    """
      Raises a warning when deprecated methods are used.
      @ In, add, object, additional information to print
      @ Out, None
    """
    msg = 'Using DEPRECATED data object API!'
    if add is not None:
      msg += ' "{}"'.format(add)
    self.raiseAWarning(msg)

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
    if self._data is None:
      self._data = cached_ndarray.cNDarray(width = len(self.vars),length=4,dtype=object)
      self._data.size = 1
      self._data.width = len(self.vars)
    #try:
    column = self._getVariableIndex(name)
    self._data._addOneEntry(column,value[0])
    if False:
    #except ValueError:
        #self._data._addOneEntry(column,value[0])
        #self._data.addEntity(np.array([value]), firstEver = True)
      self._data.addEntity([np.array([value])])
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
      self._data._addOneEntry(column,value)
    except ValueError:
      self._data.addEntity([np.array([[value]])]) #WHY should there be an extra [] in here....
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
      pass # TODO
    # pointwise
    elif name in ['ProbabilityWeight','prefix']: #TODO only add prefix if it's needed, don't default
      try:
        column = self._getVariableIndex(name)
        self._data._addOneEntry(column,value)
      except ValueError:
        self._data.addEntity([np.array([[value]])]) #WHY should there be an extra [] in here....
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
    res = {}

