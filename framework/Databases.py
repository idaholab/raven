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
from utils import InputData
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
    inputSpecification.addParam("directory", InputData.StringType)
    inputSpecification.addParam("filename", InputData.StringType)
    inputSpecification.addParam("readMode", InputData.makeEnumType("readMode","readModeType",["overwrite","read"]), True)
    inputSpecification.addSub(InputData.parameterInputFactory("variables", contentType=InputData.StringListType))
    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the database parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    if 'directory' in paramInput.parameterValues:
      self.databaseDir = copy.copy(paramInput.parameterValues['directory'])
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
        self.raiseAWarning('Requested to read from database, but it does not exist at:',fullpath,'; continuing without reading...')
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
    DateBase.__init__(self)
    self.subtype   = None
    self.exist     = False
    self.built     = False
    self.type      = 'HDF5'
    self._metavars = []
    self._allvars  = []
    self.filename = ""
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
    if not 'prefix' in rlz:
      self.raiseAnError(IOError,'addRealization method needs a prefix (ID) for adding a new group to a database!')
    self.database.addGroup(rlz['prefix'],rlz,rlz,False)
    self.built = True    


  # These are the methods that RAVEN entities should call to interact with the data object
  def addExpectedMeta(self,keys):
    """
      Registers meta to look for in realizations.
      @ In, keys, set(str), keys to register
      @ Out, None
    """
    # TODO add option to skip parts of meta if user wants to
    # remove already existing keys
    keys = list(key for key in keys if key not in self._metavars)
    # if no new meta, move along
    if len(keys) == 0:
      return
    # CANNOT add expected new meta after database has been used
    assert(len(self._metavars) == 0)
    self._metavars.extend(keys)
    self._allvars.extend(keys)


  def addGroup(self,attributes,loadFrom,upGroup=False):
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
    #self.database.addExpectedMeta(keys)
    self.addMetaKeys(*keys)

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
    allData = [self.realization(name) for name in allRealizationNames]
    return allData
  
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
      rlz = self.database._getRealizationByName(allRealizations[index] if type(index).__name__ == 'int' else index ,{'reconstruct':True})
    return rlz
  
  #def __retrieveDataPointSet(self,attributes):
    #"""
      #Function to retrieve a PointSet from the HDF5 database
      #@ In, attributes, dict, options (metadata must be appended to the root group)
      #@ Out, tupleVar, tuple, tuple in which the first position is a dictionary of numpy arays (input variable)
      #and the second is a dictionary of the numpy arrays (output variables).
    #"""
    ## Check the outParam variables and the outputPivotVal filters
    #inParam, outParam, inputRow, outputRow                 = attributes['inParam'], attributes['outParam'], copy.deepcopy(attributes.get('inputRow',None)), copy.deepcopy(attributes.get('outputRow',None))
    #inputPivotVal, outputPivotVal, operator                = attributes.get('inputPivotValue',None), attributes.get('outputPivotValue',None), attributes.get('operator',None)
    #pivotParameter                                         = attributes.get('pivotParameter',None)

    #if outParam == 'all':
      #allOutParam  = True
    #else:
      #allOutParam = False

    #if outputPivotVal != None:
      #if 'end' in outputPivotVal:
        #outputPivotValEnd = True
      #else:
        #outputPivotValEnd, outputPivotVal = False,  float(outputPivotVal)
    #else:
      #if operator is None:
        #outputPivotValEnd = True
      #else:
        #outputPivotValEnd = False
    #if inputRow == None and inputPivotVal == None:
      #inputRow = 0
    #if inputRow == None and inputPivotVal == None:
      #inputRow = 0
    #if inputRow != None :
      #inputRow = int(inputRow)
      #if inputRow  > 0:
        #inputRow  -= 1
    #if outputRow != None:
      #outputRow = int(outputRow)
      #if outputRow > 0:
        #outputRow -= 1

    #inDict   = {}
    #outDict  = {}
    #metaDict = {}
    #histList = attributes['HistorySet']
    ## Retrieve all the associated HistorySet and process them
    #for i in range(len(histList)):
      ## Load the data into the numpy array
      #attributes['history'] = histList[i]
      #histVar = self.returnHistory(attributes)
      ##look for pivotParameter
      #if pivotParameter != None:
        #pivotIndex = histVar[1]['outputSpaceHeaders'].index(pivotParameter) if pivotParameter in histVar[1]['outputSpaceHeaders'] else None
        #if pivotIndex == None:
          #self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in history '+ str(attributes['history']) + '!')
      #else:
        #pivotIndex = histVar[1]['outputSpaceHeaders'].index("time") if "time" in histVar[1]['outputSpaceHeaders'] else None
        ## if None...default is 0
        #if pivotIndex == None:
          #pivotIndex = 0
      #if inputRow > histVar[0][:,0].size-1 and inputRow != -1:
        #self.raiseAnError(IOError,'inputRow is greater than number of actual rows in history '+ str(attributes['history']) + '!')
      ## check metadata
      #if 'metadata' in histVar[1].keys():
        #metaDict[i] = histVar[1]['metadata']
      #else:
        #metaDict[i] = None
      #for key in inParam:
        #if 'inputSpaceHeaders' in histVar[1]:
          #inInKey = utils.keyIn(histVar[1]['inputSpaceHeaders'],key)
          #inOutKey = utils.keyIn(histVar[1]['outputSpaceHeaders'],key)
          #if inInKey != None:
            #ix = histVar[1]['inputSpaceHeaders'].index(inInKey)
            #if i == 0:
              #inDict[key] = np.zeros(len(histList))
            #inDict[key][i] = np.atleast_1d(histVar[1]['inputSpaceValues'][ix])[0]
          #elif inOutKey != None and inInKey == None:
            #ix = histVar[1]['outputSpaceHeaders'].index(inOutKey)
            #if i == 0:
              #inDict[key] = np.zeros(len(histList))
            #if inputPivotVal != None:
              #if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]):
                #self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
              #inDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))
            #else:
              #inDict[key][i] = histVar[0][inputRow,ix]
          #else:
            #self.raiseAnError(IOError,'the parameter ' + key + ' has not been found')
        #else:
          #inKey = utils.keyIn(histVar[1]['outputSpaceHeaders'],key)
          #if inKey is not None:
            #ix = histVar[1]['outputSpaceHeaders'].index(inKey)
            #if i == 0:
              #inDict[key] = np.zeros(len(histList))
            #if inputPivotVal != None:
              #if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]):
                #self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
              #inDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))
            #else:
              #inDict[key][i] = histVar[0][inputRow,ix]
          #else:
            #self.raiseAnError(IOError,'the parameter ' + key + ' has not been found in '+str(histVar[1]))
      ## outputPivotVal end case => PointSet is at the final status
      #if outputPivotValEnd:
        #if allOutParam:
          #for key in histVar[1]['outputSpaceHeaders']:
            #if i == 0:
              #outDict[key] = np.zeros(len(histList))
            #outDict[key][i] = histVar[0][-1,histVar[1]['outputSpaceHeaders'].index(key)]
        #else:
          #for key in outParam:
            #if key in histVar[1]['outputSpaceHeaders'] or \
               #utils.toBytes(key) in histVar[1]['outputSpaceHeaders']:
              #if i == 0:
                #outDict[key] = np.zeros(len(histList))
              #if key in histVar[1]['outputSpaceHeaders']:
                #outDict[key][i] = histVar[0][-1,histVar[1]['outputSpaceHeaders'].index(key)]
              #else:
                #outDict[key][i] = histVar[0][-1,histVar[1]['outputSpaceHeaders'].index(utils.toBytes(key))]
            #else:
              #self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      #elif outputRow != None:
        #if outputRow > histVar[0][:
          #,0].size-1  and outputRow != -1: self.raiseAnError(IOError,'outputRow is greater than number of actual rows in Database '+ str(self.name) + '!')
        #if allOutParam:
          #for key in histVar[1]['outputSpaceHeaders']:
            #if i == 0:
              #outDict[key] = np.zeros(len(histList))
            #outDict[key][i] = histVar[0][outputRow,histVar[1]['outputSpaceHeaders'].index(key)]
        #else:
          #for key in outParam:
            #if key in histVar[1]['outputSpaceHeaders'] or \
               #utils.toBytes(key) in histVar[1]['outputSpaceHeaders']:
              #if i == 0:
                #outDict[key] = np.zeros(len(histList))
              #if key in histVar[1]['outputSpaceHeaders']:
                #outDict[key][i] = histVar[0][outputRow,histVar[1]['outputSpaceHeaders'].index(key)]
              #else:
                #outDict[key][i] = histVar[0][outputRow,histVar[1]['outputSpaceHeaders'].index(utils.toBytes(key))]
            #else:
              #self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      #elif operator != None:
        #if operator not in ['max','min','average']:
          #self.raiseAnError(IOError,'operator unknown. Available are min,max,average')
        #if histVar[1]['outputSpaceHeaders']:
          #for key in histVar[1]['outputSpaceHeaders']:
            #if i == 0:
              #outDict[key] = np.zeros(len(histList))
            #if operator == 'max':
              #outDict[key][i] = np.max(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)])
            #if operator == 'min':
              #outDict[key][i] = np.min(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)])
            #if operator == 'average':
              #outDict[key][i] = np.average(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)])
        #else:
          #for key in outParam:
            #if key in histVar[1]['outputSpaceHeaders'] or \
               #utils.toBytes(key) in histVar[1]['outputSpaceHeaders']:
              #if i == 0:
                #outDict[key] = np.zeros(len(histList))
              #if key in histVar[1]['outputSpaceHeaders']:
                #if operator == 'max':
                  #outDict[key][i] = np.max(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)])
                #if operator == 'min':
                  #outDict[key][i] = np.min(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)])
                #if operator == 'average':
                  #outDict[key][i] = np.average(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)])
              #else:
                #if operator == 'max':
                  #outDict[key][i] = np.max(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(utils.toBytes(key))])
                #if operator == 'min':
                  #outDict[key][i] = np.min(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(utils.toBytes(key))])
                #if operator == 'average':
                  #outDict[key][i] = np.average(histVar[0][:,histVar[1]['outputSpaceHeaders'].index(utils.toBytes(key))])
            #else:
              #self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      #else:
        ## Arbitrary point in outputPivotVal case... If the requested outputPivotVal point Set does not match any of the stored ones and
        ## start_outputPivotVal <= requested_outputPivotVal_point <= end_outputPivotVal, compute an interpolated value
        #if allOutParam:
          #for key in histVar[1]['outputSpaceHeaders']:
            #if i == 0:
              #outDict[key] = np.zeros(len(histList))
            #outDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)], kind='linear')(outputPivotVal)
        #else:
          #for key in outParam:
            #if i == 0:
              #outDict[key] = np.zeros(len(histList))
            #if key in histVar[1]['outputSpaceHeaders'] or \
               #utils.toBytes(key) in histVar[1]['outputSpaceHeaders']:
              #if key in histVar[1]['outputSpaceHeaders']:
                #outDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)], kind='linear')(outputPivotVal)
              #else:
                #outDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['outputSpaceHeaders'].index(utils.toBytes(key))], kind='linear')(outputPivotVal)
            #else:
              #self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      #del histVar
    ## return tuple of PointSet
    #return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

  #def __retrieveDataHistory(self,attributes):
    #"""
      #Function to retrieve a History from the HDF5 database
      #@ In, attributes, dict, options (metadata muste be appended to the root group)
      #@ Out, tupleVar, tuple, tuple in which the first position is a dictionary of numpy arays (input variable)
      #and the second is a dictionary of the numpy arrays (output variables).
    #"""
    ## Check the outParam variables and the outputPivotVal filters

    #inParam, outParam, inputRow                 = attributes['inParam'], attributes['outParam'], copy.deepcopy(attributes.get('inputRow',None))
    #inputPivotVal, outputPivotVal               = attributes.get('inputPivotValue',None), attributes.get('outputPivotValue',None)
    #pivotParameter                              = attributes.get('pivotParameter',None)
    #if 'all' in outParam:
      #allOutParam = True
    #else:
      #allOutParam = False
    #if outputPivotVal != None:
      #if 'all' in outputPivotVal:
        #outputPivotValAll = True
      #else:
        #outputPivotValAll, outputPivotVal = False,  [float(x) for x in outputPivotVal.split()]
    #else:
      #outputPivotValAll = True
    #if inputRow == None and inputPivotVal == None:
      #inputRow = 0
    #if inputRow == None and inputPivotVal == None:
      #inputRow = 0
    #if inputRow != None :
      #inputRow = int(inputRow)
      #if inputRow  > 0:
        #inputRow  -= 1
    #inDict  = {}
    #outDict = {}
    #metaDict= {}
    ## Call the function to retrieve a single history and
    ## load the data into the tuple
    #histVar = self.returnHistory(attributes)
    #if pivotParameter != None:
      #pivotIndex = histVar[1]['outputSpaceHeaders'].index(pivotParameter) if pivotParameter in histVar[1]['outputSpaceHeaders'] else None
      #if pivotIndex == None:
        #self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in Database '+ str(self.name) + '!')
    #else:
      #pivotIndex = histVar[1]['outputSpaceHeaders'].index("time") if "time" in histVar[1]['outputSpaceHeaders'] else None
      ## if None...default is 0
      #if pivotIndex == None:
        #pivotIndex = 0

    #if 'metadata' in histVar[1].keys():
      #metaDict[0] = histVar[1]['metadata']
    #else:
      #metaDict[0] = None
    ## fill input param dictionary
    #for key in inParam:
      #if 'inputSpaceHeaders' in histVar[1]:
        #inInKey = utils.keyIn(histVar[1]['inputSpaceHeaders'],key)
        #inOutKey = utils.keyIn(histVar[1]['outputSpaceHeaders'],key)
        #if inInKey != None:
          #ix = histVar[1]['inputSpaceHeaders'].index(inInKey)
          #inDict[key] = np.atleast_1d(np.array(histVar[1]['inputSpaceValues'][ix]))
        #elif inOutKey != None and inInKey == None:
          #ix = histVar[1]['outputSpaceHeaders'].index(inOutKey)
          #if inputPivotVal != None:
            #if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]):
              #self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
            #inDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))))
          #else:
            #inDict[key] = np.atleast_1d(np.array(histVar[0][inputRow,ix]))
        #else:
          #self.raiseAnError(RuntimeError,'the parameter ' + key + ' has not been found in '+str(histVar[1]['inputSpaceHeaders'])+' or '+str(histVar[1]['outputSpaceHeaders']))
      #else:
        #inOutKey = utils.keyIn(histVar[1]['outputSpaceHeaders'],key)
        #if inOutKey is not None:
          #ix = histVar[1]['outputSpaceHeaders'].index(inOutKey)
          #if inputPivotVal != None:
            #if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]):
              #self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
            #inDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))))
          #else:
            #inDict[key] = np.atleast_1d(np.array(histVar[0][inputRow,ix]))
        #else:
          #self.raiseAnError(RuntimeError,'the parameter ' + key + ' has not been found in '+str(histVar[1]['outputSpaceHeaders']))

    ##  all case => The history is completed (from startTime to end_time)
    #if outputPivotValAll:
      #if allOutParam:
        #for key in histVar[1]['outputSpaceHeaders']:
          #outDict[key] = histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)]
      #else:
        #for key in outParam:
          #inKey = utils.keyIn(histVar[1]['outputSpaceHeaders'],key)
          #if inKey:
            #outDict[key] = histVar[0][:,histVar[1]['outputSpaceHeaders'].index(inKey)]
          #else:
            #self.raiseAnError(RuntimeError,'the parameter ' + key + ' has not been found in '+str(histVar[1]['outputSpaceHeaders']))
    #else:
      #if allOutParam:
        #for key in histVar[1]['outputSpaceHeaders']:
          #outDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)], kind='linear')(outputPivotVal)))
      #else:
        #for key in outParam:
          #if key in histVar[1]['outputSpaceHeaders']:
            #outDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['outputSpaceHeaders'].index(key)], kind='linear')(outputPivotVal)))
          #else:
            #self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    ## Return tuple of dictionaries containing the HistorySet
    #return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

  #def retrieveData(self,attributes):
    #"""
      #Function interface for retrieving a Point or PointSet or History from the HDF5 database
      #@ In, attributes, dict, options (metadata muste be appended to the root group)
      #@ Out, data, tuple, tuple in which the first position is a dictionary of numpy arrays (input variable)
      #and the second is a dictionary of the numpy arrays (output variables).
    #"""
    #if attributes['type'] == 'PointSet':
      #data = self.__retrieveDataPointSet(attributes)
    #elif attributes['type'] == 'HistorySet':
      #listhistIn  = {}
      #listhistOut = {}
      #listhistMeta= {}
      #endGroupNames = self.getEndingGroupNames()
      #for index in range(len(endGroupNames)):
        #attributes['history'] = endGroupNames[index]
        #tupleVar = self.__retrieveDataHistory(attributes)
        ## dictionary of dictionary key = i => ith history ParameterValues dictionary
        #listhistIn[index]  = tupleVar[0]
        #listhistOut[index] = tupleVar[1]
        #listhistMeta[index]= tupleVar[2]
        #del tupleVar
      #data = (listhistIn,listhistOut,listhistMeta)
    #else:
      #self.raiseAnError(RuntimeError,'Type' + attributes['type'] +' unknown.Caller: hdf5Manager.retrieveData')
    ## return data
    #gc.collect()
    #return copy.copy(data)

>>>>>>> dfloat
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
