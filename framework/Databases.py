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
Created on April 9, 2013

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import os
import abc
import gc
from scipy.interpolate import interp1d
import collections
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from h5py_interface_creator import hdf5Database as h5Data
from utils import utils
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class DatabasesCollection(InputData.ParameterInput):
  """
    Class for reading in a collection of databases
  """

DatabasesCollection.createClass("Databases")

class DateBase(BaseType):
  """
    class to handle a database,
    Used to add and retrieve attributes and values from said database
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(DateBase, cls).getInputSpecification()
    inputSpecification.addParam("directory", InputTypes.StringType)
    inputSpecification.addParam("filename", InputTypes.StringType)
    inputSpecification.addParam("readMode", InputTypes.makeEnumType("readMode","readModeType",["overwrite","read"]), True)
    inputSpecification.addSub(InputData.parameterInputFactory("variables", contentType=InputTypes.StringListType))
    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the database parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    if 'directory' in paramInput.parameterValues:
      self.databaseDir = copy.copy(paramInput.parameterValues['directory'])
      # if not absolute path, join with working directory
      if not os.path.isabs(self.databaseDir):
        self.databaseDir = os.path.abspath(os.path.join(self.workingDir,self.databaseDir))
    else:
      self.databaseDir = os.path.join(self.workingDir,'DatabaseStorage')
    if 'filename' in paramInput.parameterValues:
      self.filename = copy.copy(paramInput.parameterValues['filename'])
    else:
      self.filename = self.name+'.h5'
    # read the variables
    varNode = paramInput.findFirst("variables")
    if varNode is not None:
      self.variables =  varNode.value
    # read mode
    self.readMode = paramInput.parameterValues['readMode'].strip().lower()
    self.raiseADebug('HDF5 Read Mode is "'+self.readMode+'".')
    if self.readMode == 'overwrite':
      # check if self.databaseDir exists or create in case not
      if not os.path.isdir(self.databaseDir):
        os.makedirs(self.databaseDir, exist_ok=True)
    # get full path
    fullpath = os.path.join(self.databaseDir,self.filename)
    if os.path.isfile(fullpath):
      if self.readMode == 'read':
        self.exist = True
      elif self.readMode == 'overwrite':
        self.exist = False
      self.database = h5Data(self.name,self.databaseDir,self.messageHandler,self.filename,self.exist,self.variables)
    else:
      #file does not exist in path
      if self.readMode == 'read':
        self.raiseAnError(IOError, 'Requested to read from database, but it does not exist at:',fullpath,'; The path to the database must be either absolute or relative to <workingDir>!')
      self.exist = False
      self.database  = h5Data(self.name,self.databaseDir,self.messageHandler,self.filename,self.exist,self.variables)
    self.raiseAMessage('Database is located at:',fullpath)


  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)
    self.database = None                # Database object
    self.exist        = False           # does it exist?
    self.built       = False            # is it built?
    self.filename    = ""               # filename
    self.workingDir  = runInfoDict['WorkingDir']
    self.databaseDir = self.workingDir  # Database directory. Default = working directory.
    self.printTag = 'DATABASE'          # For printing verbosity labels
    self.variables = None               # if not None, list of specific variables requested to be stored by user

  @abc.abstractmethod
  def addGroup(self,attributes,loadFrom):
    """
      Function used to add a group to the database
      @ In, attributes, dict, options
      @ In, loadFrom, string, source of the data
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def retrieveData(self,attributes):
    """
      Function used to retrieve data from the database
      @ In, attributes, dict, options
      @ Out, data, object, the requested data
    """
    pass

#
#  *************************s
#  *  HDF5 DATABASE CLASS  *
#  *************************
#
class HDF5(DateBase):
  """
    class to handle h5py (hdf5) databases,
    Used to add and retrieve attributes and values from said database
  """

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    DateBase.__init__(self,runInfoDict)
    self.subtype   = None
    self.type      = 'HDF5'
    self._metavars = []
    self._allvars  = []
    self.printTag = 'DATABASE HDF5'


  def __getstate__(self):
    """
      Overwrite state (for pickling)
      we do not pickle the HDF5 (C++) instance
      but only the info to reload it
      @ In, None
      @ Out, state, dict, the namespace state
    """
    # capture what is normally pickled
    state = self.__dict__.copy()
    # we pop the database instance and close it
    state.pop("database")
    self.database.closeDatabaseW()
    # what we return here will be stored in the pickle
    return state

  def __setstate__(self, newstate):
    """
      Set the state (for pickling)
      we do not pickle the HDF5 (C++) instance
      but only the info to reload it
      @ In, newstate, dict, the namespace state
      @ Out, None
    """
    self.__dict__.update(newstate)
    self.exist    = True
    self.database = h5Data(self.name,self.databaseDir,self.messageHandler,self.filename,self.exist)

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = DateBase.getInitParams(self)
    paramDict['exist'] = self.exist
    return paramDict

  def getEndingGroupNames(self):
    """
    Function to retrieve all the groups' names of the ending groups
    @ In, None
    @ Out, endingGroups, list, List of the ending groups' names
    """
    endingGroups = self.database.retrieveAllHistoryNames()
    return endingGroups

  def addRealization(self,rlz):
    """
      Adds a "row" (or "sample") to this database.
      This is the method to add data to this database.
      Note that rlz can include many more variables than this database actually wants.
      Before actually adding the realization, data is formatted for this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" is either a float or a np.ndarray of values.
      @ Out, None
    """
    # realization must be a dictionary
    assert(type(rlz).__name__ == "dict")
    # prefix must be present
    if 'prefix' not in rlz:
      rlz['prefix'] = len(self.database)
    self.database.addGroup(rlz)
    self.built = True

  def addExpectedMeta(self,keys,params={}):
    """
      Registers meta to look for in realizations.
      @ In, keys, set(str), keys to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    self.database.addExpectedMeta(keys,params)
    self.addMetaKeys(keys, params)

  def provideExpectedMetaKeys(self):
    """
      Provides the registered list of metadata keys for this entity.
      @ In, None
      @ Out, meta, tuple, (set(str),dict), expected keys (empty if none) and dictionary of expected keys with respect to their indexes, i.e. {keys:[indexes]}
    """
    return self.database.provideExpectedMetaKeys()

  def initialize(self,gname,options=None):
    """
      Function to add an initial root group into the data base...
      This group will not contain a dataset but, eventually, only metadata
      @ In, gname, string, name of the root group
      @ In, options, dict, options (metadata muste be appended to the root group), Default =None
      @ Out, None
    """
    self.database.addGroupInit(gname,options)

  def returnHistory(self,options):
    """
      Function to retrieve a history from the HDF5 database
      @ In, options, dict, options (metadata muste be appended to the root group)
      @ Out, tupleVar, tuple, tuple in which the first position is a numpy aray and the second is a dictionary of the metadata
      Note:
      # DET => a Branch from the tail (group name in attributes) to the head (dependent on the filter)
      # MC  => The History named ['group'] (one run)
    """
    tupleVar = self.database.retrieveHistory(options['history'],options)
    return tupleVar

  def allRealizations(self):
    """
      Casts this database as an xr.Dataset.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, allData, list of arrays, all the data from this data object.
    """
    allRealizationNames = self.database.retrieveAllHistoryNames()
    # instead to use a OrderedDict in the database, I sort the names here (it is much faster)
    allRealizationNames.sort()
    allData = [self.realization(name) for name in allRealizationNames]
    return allData

  def realization(self,index=None,matchDict=None,tol=1e-15):
    """
      Method to obtain a realization from the data, either by index (e.g. realization number) or matching value.
      Either "index" or "matchDict" must be supplied. (NOTE: now just "index" can be supplied)
      @ In, index, int or str, optional, number of row to retrieve (by index, not be "sample") or group name
      @ In, matchDict, dict, optional, {key:val} to search for matches
      @ In, tol, float, optional, tolerance to which match should be made
      @ Out, index, int, optional, index where found (or len(self) if not found), only returned if matchDict
      @ Out, rlz, dict, realization requested (None if not found)
    """
    # matchDict not implemented for Databases
    assert (matchDict is None)
    if (not self.exist) and (not self.built):
      self.raiseAnError(Exception,'Can not retrieve a realization from Database' + self.name + '.It has not been built yet!')
    if type(index).__name__ == 'int':
      allRealizations = self.database.retrieveAllHistoryNames()
    if type(index).__name__ == 'int' and index > len(allRealizations):
      rlz = None
    else:
      rlz,_ = self.database._getRealizationByName(allRealizations[index] if type(index).__name__ == 'int' else index ,{'reconstruct':True})
    return rlz

__base                  = 'Database'
__interFaceDict         = {}
__interFaceDict['HDF5'] = HDF5
__knownTypes            = __interFaceDict.keys()

# add input specifications in DatabasesCollection
DatabasesCollection.addSub(HDF5.getInputSpecification())

def knownTypes():
  """
   Return the known types
   @ In, None
   @ Out, __knownTypes, list, the known types
  """
  return __knownTypes

needsRunInfo = True

def returnInstance(Type,runInfoDict,caller):
  """
  Function interface for creating an instance to a database specialized class (for example, HDF5)
  @ In, Type, string, class type
  @ In, runInfoDict, dict, the runInfo Dictionary
  @ In, caller, instance, the caller instance
  @ Out, returnInstance, instance, instance of the class
  """
  try:
    return __interFaceDict[Type](runInfoDict)
  except KeyError:
    caller.raiseAnError(NameError,'not known '+__base+' type '+Type)

def returnInputParameter():
  """
    Function returns the InputParameterClass that can be used to parse the
    whole collection.
    @ Out, returnInputParameter, DatabasesCollection, class for parsing.
  """
  return DatabasesCollection()

