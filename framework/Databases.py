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
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import os
import abc
import gc
from scipy.interpolate import interp1d
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from h5py_interface_creator import hdf5Database as h5Data
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------

class DateBase(BaseType):
  """
    class to handle a database,
    Used to add and retrieve attributes and values from said database
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)     # Base Class
    self.database = None        # Database object
    self.databaseDir = ''       # Database directory. Default = working directory.
    self.workingDir = ''        #
    self.printTag = 'DATABASE'  # For printing verbosity labels
    self.variables = None       # if not None, list of specific variables requested to be stored by user

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this
      specialized class and initialize variables based on the inputs received.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ Out, None
    """
    # Check if a directory has been provided
    if 'directory' in xmlNode.attrib.keys():
      self.databaseDir = copy.copy(xmlNode.attrib['directory'])
    else:
      self.databaseDir = os.path.join(self.workingDir,'DatabaseStorage')
    # Check for variables listing
    varsNode = xmlNode.find('variables')
    if varsNode is not None:
      self.variables = list(v.strip() for v in varsNode.text.split(','))

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
    DateBase.__init__(self)
    self.subtype   = None
    self.exist     = False
    self.built     = False
    self.type      = 'HDF5'
    self._metavars = []
    self._allvars  = []
    self.filename = ""
    self.printTag = 'DATABASE HDF5'
    self.workingDir = runInfoDict['WorkingDir']
    self.databaseDir = self.workingDir

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


  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this
      specialized class and initialize variables based on the input received.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ Out, None
    """
    DateBase._readMoreXML(self, xmlNode)
    # Check if database directory exist, otherwise create it
    if '~' in self.databaseDir:
      self.databaseDir = copy.copy(os.path.expanduser(self.databaseDir))
    # Determine RELATIVE location for HDF5.
    # - if a full path is given, accept it as given, else ...
    if not os.path.isabs(self.databaseDir):
      # use working dir as base
      self.databaseDir = os.path.join(self.workingDir,self.databaseDir)
    self.databaseDir = os.path.normpath(self.databaseDir)

    utils.makeDir(self.databaseDir)
    self.raiseADebug('Database Directory is:',self.databaseDir)
    # Check if a filename has been provided
    # if yes, we assume the user wants to load the data from there
    # or update it
    #try:
    self.filename = xmlNode.attrib.get('filename',self.name+'.h5')
    if 'readMode' not in xmlNode.attrib.keys():
      self.raiseAnError(IOError,'No "readMode" attribute was specified for hdf5 database',self.name)
    self.readMode = xmlNode.attrib['readMode'].strip().lower()
    readModes = ['read','overwrite']
    if self.readMode not in readModes:
      self.raiseAnError(IOError,'readMode attribute for hdf5 database',self.name,'is not recognized:',self.readMode,'.  Options are:',readModes)
    self.raiseADebug('HDF5 Read Mode is "'+self.readMode+'".')
    fullpath = os.path.join(self.databaseDir,self.filename)
    if os.path.isfile(fullpath):
      if self.readMode == 'read':
        self.exist = True
      elif self.readMode == 'overwrite':
        self.exist = False
      self.database = h5Data(self.name,self.databaseDir,self.messageHandler,self.filename,self.exist)
    else:
      #file does not exist in path
      if self.readMode == 'read':
        self.raiseAWarning('Requested to read from database, but it does not exist at:',fullpath,'; continuing without reading...')
      self.exist = False
      self.database  = h5Data(self.name,self.databaseDir,self.messageHandler,self.filename,self.exist)
    self.raiseAMessage('Database is located at:',fullpath)

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

  def getEndingGroupPaths(self):
    """
      Function to retrieve all the groups' paths of the ending groups
      @ In, None
      @ Out, histories, list, List of the ending groups' paths
    """
    histories = self.database.retrieveAllHistoryPaths()
    return histories

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
    assert('prefix' in rlz)
    self.database.addGroup(rlz)
    self.built = True    

  def addExpectedMeta(self,keys):
    """
      Registers meta to look for in realizations.
      @ In, keys, set(str), keys to register
      @ Out, None
    """
    self.database.addExpectedMeta(keys)
    self.addMetaKeys(*keys)
  
  def initialize(self,gname,options={}):
    """
      Function to add an initial root group into the data base...
      This group will not contain a dataset but, eventually, only metadata
      @ In, gname, string, name of the root group
      @ In, options, dict, options (metadata muste be appended to the root group)
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
    if type(index).__name__ == 'int': allRealizations = self.database.retrieveAllHistoryNames()
    if type(index).__name__ == 'int' and index > len(allRealizations):
      rlz = None
    else:
      rlz,_ = self.database._getRealizationByName(allRealizations[index] if type(index).__name__ == 'int' else index ,{'reconstruct':True})
    return rlz
  
__base                  = 'Database'
__interFaceDict         = {}
__interFaceDict['HDF5'] = HDF5
__knownTypes            = __interFaceDict.keys()

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
