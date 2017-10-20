import sys,os
import __builtin__

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

  def __init__(self): #TODO message handler
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)
    self._inputs     = []     # list(str) if input variables
    self._outputs    = []     # list(str) of output variables
    self._metavars   = []     # list(str) of POINTWISE metadata variables
    self._data       = None   # underlying data structure
    self._collector  = None   # object used to collect samples
    self._heirarchal = False  # if True, non-traditional format (not yet implemented)

  def _readMoreXML(self,xmlNode):
    """
      Initializes data object based on XML input
      @ In, xmlNode, xml.etree.ElementTree.Element, input information
      @ Out, None
    """
    pass

  def add_realization(self,info_dict):
    pass

  def get_data(self):
    return self._data

  def read(self,fname):
    return xr.open_dataset(fname)
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
  def __init__(self):#, in_vars, out_vars, meta_vars=None, dynamic=False, var_dims=None,cacheSize=100,prealloc=False):
    """
      Constructor.
    """
    DataObject.__init__(self)
    #self.vars = self.in_vars + self.out_vars
    #if self.dynamic:
    #  self._data = cached_ndarray.cNDarray(width=len(self.vars),dtype=object)
    #else:
    #  self._data = cached_ndarray.cNDarray(width=len(self.vars))

  @property
  def vars(self):
    """
      Property to access all the pointwise variables being controlled by this data object.
      @ In, None
      @ Out, vars, list(str), variable names list
    """
    return self._inputs + self._outputs + self._metavars

  def __len__(self):
    """
      Overloads the len() operator.
      @ In, None
      @ Out, int, number of samples in this dataset
    """
    pass #TODO

  def _readMoreXML(self,xmlNode):
    """
      Initializes data object based on XML input
      @ In, xmlNode, xml.etree.ElementTree.Element, input information
      @ Out, None
    """
    inp = DataSet.getInputSpecification()()
    print('Node:',xmlNode)
    inp.parseNode(xmlNode)
    for child in inp.subparts:
      if child.getName() == 'Input':
        self._inputs.extend(list(x for x in child.value.split(',')))
      elif child.getName() == 'Output':
        self._outputs.extend(list(x for x in child.value.split(',')))

  # API TRANSLATION
  #  OLD                               |    NEW
  # addOutput                          | ? load from values
  # getAllMetadata                     | ? -remove-
  # getHierParam                       | ? heirarchal only
  # getInitParams                      | ? useful?
  # getInpParametersValues             | ? getInputValues
  # getMatchingRealization             | ? same
  # getMetadata                        | ? getPointMeta, getGeneralMeta
  # getOutParametersValues             | ? getOutputValues
  # getParaKeys                        | ? getInputs, getOutputs, getPointMeta, getGeneralMeta
  # getParam                           | ? getVarValues
  # getParametersValues                | ? getInputs, getOutputs, getPointMeta, getGeneralMeta
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

  ### NEW API ###
  def addRealization(self,rlz):
    """
      Adds a "row" (or "sample") to this data object.
      This is the preferred method to add data to this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" can be either a float (pointset) or xr.DataArray object (ndset)
      @ Out, None
    """
    # FIXME TODO dynamic
    if self.dynamic:
      self._data.append(np.asarray([list(rlz[var] for var in self.vars)],dtype=object))
    else:
      self._data.append(np.asarray([list(rlz[var] for var in self.vars)]))

  def asDataset(self):
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
      for v,var in enumerate(self.vars):
        if type(data[0,v]) == float:
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

  def toNetCDF4(self,fname,**kwargs):
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

  ### OLD API ###
