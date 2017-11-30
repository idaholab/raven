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
Created on Mar 25, 2013

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
from datetime import datetime
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import h5py  as h5
import numpy as np
import os
import copy
import json
import string
import difflib
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from utils import mathUtils
import MessageHandler
import Files
#Internal Modules End--------------------------------------------------------------------------------

#
#  *************************
#  *  HDF5 DATABASE CLASS  *
#  *************************
#

class hdf5Database(MessageHandler.MessageUser):
  """
    class to create a h5py (hdf5) database
  """
  def __init__(self,name, databaseDir, messageHandler,filename,exist):
    """
      Constructor
      @ In, name, string, name of this database
      @ In, databaseDir, string, database directory (full path)
      @ In, messageHandler, MessageHandler, global message handler
      @ In, filename, string, the database filename
      @ In, exist, bool, does it exist?
      @ Out, None
    """
    # database name (i.e. arbitrary name).
    # It is the database name that has been found in the xml input
    self.name       = name
    # Database type :
    # -> The structure type is "inferred" by the first group is going to be added
    # * MC  = MonteCarlo => Storing by a Parallel structure
    # * DET = Dynamic Event Tree => Storing by a Hierarchical structure
    self.type       = None
    self._metavars = []
    self._allvars  = []
    # specialize printTag (THIS IS THE CORRECT WAY TO DO THIS)
    self.printTag = 'DATABASE HDF5'
    self.messageHandler = messageHandler
    # does it exist?
    self.fileExist = exist
    # .H5 file name (to be created or read)
    # File name on disk
    self.onDiskFile = filename
    # Database directory
    self.databaseDir =  databaseDir
    # Create file name and path
    self.filenameAndPath = os.path.join(self.databaseDir,self.onDiskFile)
    # Is the file opened?
    self.fileOpen       = False
    # List of the paths of all the groups that are stored in the database
    self.allGroupPaths = []
    # Dictonary of boolean variables, true if the corresponding group in self.allGroupPaths
    # is an ending group (no sub-groups appended), false otherwise
    self.allGroupEnds = {}
    # We can create a base empty database or we open an existing one
    if self.fileExist:
      # self.h5FileW is the HDF5 object. Open the database in "update" mode
      # check if it exists
      if not os.path.exists(self.filenameAndPath):
        self.raiseAnError(IOError,'database file has not been found, searched Path is: ' + self.filenameAndPath )
      # Open file
      self.h5FileW = self.openDatabaseW(self.filenameAndPath,'r+')
      # Call the private method __createObjFromFile, that constructs the list of the paths "self.allGroupPaths"
      # and the dictionary "self.allGroupEnds" based on the database that already exists
      self.parentGroupName = b'/'
      self.__createObjFromFile()
      # "self.firstRootGroup", true if the root group is present (or added), false otherwise
      self.firstRootGroup = True
    else:
      # self.h5FileW is the HDF5 object. Open the database in "write only" mode
      self.h5FileW = self.openDatabaseW(self.filenameAndPath,'w')
      # Add the root as first group
      self.allGroupPaths.append("/")
      # The root group is not an end group
      self.allGroupEnds["/"] = False
      # The first root group has not been added yet
      self.firstRootGroup = False
      # The root name is / . it can be changed if addGroupInit is called
      self.parentGroupName = b'/'
  
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
  
  def __createObjFromFile(self):
    """
      Function to create the list "self.allGroupPaths" and the dictionary "self.allGroupEnds"
      from a database that already exists. It uses the h5py method "visititems" in conjunction
      with the private method "self.__isGroup"
      @ In, None
      @ Out, None
    """
    self.allGroupPaths = []
    self.allGroupEnds  = {}
    if not self.fileOpen:
      self.h5FileW = self.openDatabaseW(self.filenameAndPath,'a')
    self.h5FileW.visititems(self.__isGroup)
    self.raiseAMessage('TOTAL NUMBER OF GROUPS = ' + str(len(self.allGroupPaths)))

  def __isGroup(self,name,obj):
    """
      Function to check if an object name is of type "group". If it is, the function stores
      its name into the "self.allGroupPaths" list and update the dictionary "self.allGroupEnds"
      @ In, name, string, object name
      @ In, obj, object, the object itself
      @ Out, None
    """
    if isinstance(obj,h5.Group):
      self.allGroupPaths.append(name)
      try:
        self.allGroupEnds[name]  = obj.attrs["EndGroup"]
      except KeyError:
        self.allGroupEnds[name]  = True
      if "rootname" in obj.attrs:
        self.parentGroupName = name
        self.raiseAWarning('not found attribute EndGroup in group ' + name + '.Set True.')
    return

  def addGroup(self,rlz): # groupName,attributes,source,upGroup=False):
    """
      Function to add a group into the database
      @ In, groupName, string, group name
      @ In, attributes, dict, dictionary of attributes that must be added as metadata
      @ In, source, File object, data source (for example, csv file)
      @ In, upGroup, bool, optional, updated group?
      @ Out, None
    """
    parentID  = rlz.get("parentID",None)
    groupName = rlz.get("prefix")
    
    if parentID:
      #If Hierarchical structure, firstly add the root group
      if not self.firstRootGroup or parentID == 'root':        
        self.__addGroupRootLevel(groupName,attributes,source,upGroup)
        self.firstRootGroup = True
        self.type = 'DET'
      else:
        # Add sub group in the Hierarchical structure
        self.__addSubGroup(groupName,attributes,source)
      endif
    else:
      # Parallel structure (always root level)
      self.__addGroupRootLevel(groupName,rlz)
      self.firstRootGroup = True
      self.type = 'MC'      
    endif
    self.h5FileW.flush()
    

  def addGroupInit(self,groupName,attributes=None,upGroup=False):
    """
      Function to add an empty group to the database
      This function is generally used when the user provides a rootname in the input.
      It uses the groupName + it appends the date and time.
      @ In, groupName, string, group name
      @ In, attributes, dict, optional, dictionary of attributes that must be added as metadata
      @ In, upGroup, bool, optional, updated group?
      @ Out, None
    """
    groupNameInit = groupName+"_"+datetime.now().strftime("%m-%d-%Y-%H")
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        splittedPath=comparisonName.split('/')
        if len(splittedPath) > 0:
          if groupNameInit in splittedPath[0]:
            alphabetCounter, movingCounter = 0, 0
            asciiAlphabet   = list(string.ascii_uppercase)
            prefixLetter          = ''
            while True:
              testGroup = groupNameInit +"_"+prefixLetter+asciiAlphabet[alphabetCounter]
              if testGroup not in self.allGroupPaths:
                groupNameInit = testGroup
                break
              alphabetCounter+=1
              if alphabetCounter >= len(asciiAlphabet):
                prefix = asciiAlphabet[movingCounter]
                alphabetCounter = 0
                movingCounter  += 1
            break
            # self.raiseAnError(IOError,"Group named " + groupName + " already present as root group in database " + self.name + ". new group " + groupName + " is equal to old group " + splittedPath[0])
    self.parentGroupName = "/" + groupNameInit
    # Create the group
    grp = self.h5FileW.create_group(groupNameInit)
    # Add metadata
    if attributes:
      for key in attributes.keys():
        grp.attrs[key] = attributes[key]
    grp.attrs['rootname'] = True
    grp.attrs['EndGroup'] = False
    self.allGroupPaths.append("/" + groupNameInit)
    self.allGroupEnds["/" + groupNameInit] = False
    self.h5FileW.flush()

  def __addGroupRootLevel(self,groupName,rlz):
    """
      Function to add a group into the database (root level)
      @ In, groupName, string, group name
      @ In, attributes, dict, dictionary of attributes that must be added as metadata
      @ In, source, File object, source file
      @ In, upGroup, bool, optional, updated group?
      @ Out, None
    """
    # Check in the "self.allGroupPaths" list if a group is already present...
    # If so, error (Deleting already present information is not desiderable)
    for index in xrange(len(self.allGroupPaths)):
      comparisonName = self.allGroupPaths[index]
      splittedPath=comparisonName.split('/')
      for splgroup in splittedPath:
        if groupName == splgroup and splittedPath[0] == self.parentGroupName:
          found = True
          while found:
            if groupName in splittedPath:
              found = True
            else:
              found = False
            groupName = groupName + "_"+ str(index)
            #self.raiseAnError(IOError,"Group named " + groupName + " already present in database " + self.name + ". new group " + groupName + " is equal to old group " + comparisonName)
    parentName = self.parentGroupName.replace('/', '')
    # Create the group
    if parentName != '/':
      parentGroupName = self.__returnParentGroupPath(parentName)
      # Retrieve the parent group from the HDF5 database
      if parentGroupName in self.h5FileW:
        parentGroupObj = self.h5FileW.require_group(parentGroupName)
      else:
        self.raiseAnError(ValueError,'NOT FOUND group named ' + parentGroupObj)
    else:
      parentGroupObj = self.h5FileW
      
    # create the group
    groups = parentGroupObj.create_group(groupName)
    
    groups.attrs[b'mainClass' ] = b'PythonType'
    groups.attrs[b'sourceType'] = b'Dictionary'
    # I keep this structure here because I want to maintain the possibility to add a whatever dictionary even if not prepared and divided into output and input sub-sets. A.A.
    # use ONLY the subset of variables if requested
    
    
    
    metaValues = {}
    for key in self._metavars:
      metaValues[key] = rlz[key]
    # add pointwise metadata (in this case, they are group-wise)
    groups.attrs[b'point-wise-metadata'] = metaValues
    for var,item in rlz.items():
      pass
    
    for rl in rlz:
      print(type(rlz[rl]))
      print(rl)
    
    
    
    if set(['inputSpaceParams']).issubset(set(source['name'].keys())):
      sourceInputs = source['name']['inputSpaceParams'].keys()
      if specificVars is not None:
        inputHeaders = list(var for var in sourceInputs if var in specificVars)
      else:
        inputHeaders = sourceInputs
      inputHeaders = list(utils.toBytesIterative(inputHeaders))
      groups.attrs[b'inputSpaceHeaders' ] = inputHeaders

      if specificVars is not None:
        inputValues = list(source['name']['inputSpaceParams'][var] for var in sourceInputs if var in specificVars)
      else:
        inputValues = source['name']['inputSpaceParams'].values()
      inputValues = json.dumps(list(utils.toListFromNumpyOrC1arrayIterative(list(utils.toBytesIterative(inputValues)))))
      groups.attrs[b'inputSpaceValues'  ] = inputValues

    if set(['outputSpaceParams']).issubset(set(source['name'].keys())):
      if specificVars is not None:
        outDict = dict((k,v) for k,v in source['name']['outputSpaceParams'].items() if k in specificVars)
      else:
        outDict = source['name']['outputSpaceParams']
    else:
      if specificVars is not None:
        outDict = dict((key,value) for (key,value) in source['name'].iteritems() if key not in ['inputSpaceParams'] and key in specificVars)
      else:
        outDict = dict((key,value) for (key,value) in source['name'].iteritems() if key not in ['inputSpaceParams'])
    outHeaders = utils.toBytesIterative(list(outDict.keys()))
    outValues  = utils.toBytesIterative(list(outDict.values()))
    groups.attrs[b'nParams'   ] = len(outHeaders)
    groups.attrs[b'outputSpaceHeaders'] = outHeaders
    groups.attrs[b'EndGroup'   ] = True
    groups.attrs[b'parentID'  ] = parentName
    maxSize = 0
    for value in outValues:
      if type(value) == np.ndarray or type(value).__name__ == 'c1darray':
        if maxSize < value.size:
          actualOne = np.asarray(value).size
      elif type(value) in [int,float,bool,np.float64,np.float32,np.float16,np.int64,np.int32,np.int16,np.int8,np.bool8]:
        actualOne = 1
      else:
        self.raiseAnError(IOError,'The type of the dictionary parameters must be within float,bool,int,numpy.ndarray.Got '+type(value).__name__)
      if maxSize < actualOne:
        maxSize = actualOne
    groups.attrs[b'nTimeSteps'  ] = maxSize
    dataout = np.zeros((maxSize,len(outHeaders)))
    for index in range(len(outHeaders)):
      if type(outValues[index]) == np.ndarray or type(value).__name__ == 'c1darray':
        dataout[0:outValues[index].size,index] =  np.ravel(outValues[index])[:]
      else:
        dataout[0,index] = outValues[index]
    # create the data set
    groups.create_dataset(groupName + "_data", dtype="float", data=dataout)
    # add metadata if present
    for attr in attributes.keys():
      objectToConvert = mathUtils.convertNumpyToLists(attributes[attr])
      converted = json.dumps(objectToConvert)
      if converted and attr != 'name':
        groups.attrs[utils.toBytes(attr)]=converted
    if parentGroupName != "/":
      self.allGroupPaths.append(parentGroupName + "/" + groupName)
      self.allGroupEnds[parentGroupName + "/" + groupName] = True
    else:
      self.allGroupPaths.append("/" + groupName)
      self.allGroupEnds["/" + groupName] = True





    if source['type'] == 'csv':
      # Source in CSV format
      f = open(source['name'],'rb')
      # Retrieve the headers of the CSV file
      firstRow = f.readline().strip(b"\r\n")
      #firstRow = f.readline().translate(None,"\r\n")
      headers = firstRow.split(b",")
      # if there is the alias system, replace the variable name
      if 'alias' in attributes.keys():
        for aliasType in attributes['alias'].keys():
          for var in attributes['alias'][aliasType].keys():
            if attributes['alias'][aliasType][var].strip() in headers:
              headers[headers.index(attributes['alias'][aliasType][var].strip())] = var.strip()
            else:
              metadataPresent = True if 'metadata' in attributes.keys() and 'SampledVars' in attributes['metadata'].keys() else False
              if not (metadataPresent and var.strip() in attributes['metadata']['SampledVars'].keys()):
                self.raiseAWarning('the ' + aliasType +' alias"'+var.strip()+'" has been defined but has not been found among the variables!')
      # Load the csv into a numpy array(n time steps, n parameters)
      data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
      # First parent group is the root name
      parentName = self.parentGroupName.replace('/', '')
      # Create the group
      if parentName != '/':
        parentGroupName = self.__returnParentGroupPath(parentName)
        # Retrieve the parent group from the HDF5 database
        if parentGroupName in self.h5FileW:
          rootgrp = self.h5FileW.require_group(parentGroupName)
        else:
          self.raiseAnError(ValueError,'NOT FOUND group named "' + parentGroupName+'" for loading file '+str(source['name']))
        if upGroup:
          grp = rootgrp.require_group(groupName)
          del grp[groupName+"_data"]
        else:
          grp = rootgrp.create_group(groupName)
      else:
        if upGroup:
          grp = self.h5FileW.require_group(groupName)
        else:
          grp = self.h5FileW.create_group(groupName)
      self.raiseAMessage('Adding group named "' + groupName + '" in DataBase "'+ self.name +'"')
      # Create dataset in this newly added group
      grp.create_dataset(groupName+"_data", dtype="float", data=data)
      # Add metadata
      grp.attrs["outputSpaceHeaders"     ] = headers
      grp.attrs["nParams"                ] = data[0,:].size
      grp.attrs["parentID"               ] = "root"
      grp.attrs["startTime"              ] = data[0,0]
      grp.attrs["end_time"               ] = data[data[:,0].size-1,0]
      grp.attrs["nTimeSteps"             ] = data[:,0].size
      grp.attrs["EndGroup"               ] = True
      grp.attrs["sourceType"             ] = source['type']
      if source['type'] == 'csv':
        grp.attrs["sourceFile"] = source['name']
      for attr in attributes.keys():
        if attr == 'metadata':
          if 'SampledVars' in attributes['metadata'].keys():
            inpHeaders = []
            inpValues  = []
            for inkey, invalue in attributes['metadata']['SampledVars'].items():
              if inkey not in headers:
                inpHeaders.append(utils.toBytes(inkey))
                inpValues.append(invalue)
            if len(inpHeaders) > 0:
              grp.attrs[b'inputSpaceHeaders'] = inpHeaders
              grp.attrs[b'inputSpaceValues' ] = json.dumps(list(utils.toListFromNumpyOrC1arrayIterative(list( inpValues))))
        objectToConvert = mathUtils.convertNumpyToLists(attributes[attr])
        for o,obj in enumerate(objectToConvert):
          if isinstance(obj,Files.File):
            objectToConvert[o]=obj.getFilename()
        converted = json.dumps(objectToConvert)
        if converted and attr != 'name':
          grp.attrs[utils.toBytes(attr)]=converted
        #decoded = json.loads(grp.attrs[utils.toBytes(attr)])
      if "inputFile" in attributes.keys():
        grp.attrs[utils.toString("inputFile")] = utils.toString(" ".join(attributes["inputFile"])) if type(attributes["inputFile"]) == type([]) else utils.toString(attributes["inputFile"])
    else:
      self.raiseAnError(ValueError,source['type'] + " unknown!")
    # Add the group name into the list "self.allGroupPaths" and
    # set the relative bool flag into the dictionary "self.allGroupEnds"
    if parentGroupName != "/":
      self.allGroupPaths.append(parentGroupName + "/" + groupName)
      self.allGroupEnds[parentGroupName + "/" + groupName] = True
    else:
      self.allGroupPaths.append("/" + groupName)
      self.allGroupEnds["/" + groupName] = True

  def addGroupDataObjects(self,groupName,attributes,source,upGroup=False,specificVars=None):
    """
      Function to add a data (class DataObjects) or Dictionary into the Database
      @ In, groupName, string, group name
      @ In, attributes, dict, dictionary of attributes that must be added as metadata
      @ In, source, dataObject, source data
      @ In, upGroup, bool, optional, updated group?
      @ In, specificVars, list(str), if not None then indicates a selective list of variables to include in DB
      @ Out, None
    """
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        splittedPath=comparisonName.split('/')
        for splgroup in splittedPath:
          if groupName == splgroup and splittedPath[0] == self.parentGroupName:
            found = True
            while found:
              if groupName in splittedPath:
                found = True
              else:
                found = False
              groupName = groupName + "_"+ str(index)
            #self.raiseAnError(IOError,"Group named " + groupName + " already present in database " + self.name + ". new group " + groupName + " is equal to old group " + comparisonName)
    parentName = self.parentGroupName.replace('/', '')
    # Create the group
    if parentName != '/':
      parentGroupName = self.__returnParentGroupPath(parentName)
      # Retrieve the parent group from the HDF5 database
      if parentGroupName in self.h5FileW:
        parentGroupObj = self.h5FileW.require_group(parentGroupName)
      else:
        self.raiseAnError(ValueError,'NOT FOUND group named ' + parentGroupObj)
    else:
      parentGroupObj = self.h5FileW

    if type(source['name']) == dict:
      # create the group
      if upGroup:
        groups = parentGroupObj.require_group(groupName)
        del groups[groupName+"_data"]
      else:
        groups = parentGroupObj.create_group(groupName)
      groups.attrs[b'mainClass' ] = b'PythonType'
      groups.attrs[b'sourceType'] = b'Dictionary'
      # I keep this structure here because I want to maintain the possibility to add a whatever dictionary even if not prepared and divided into output and input sub-sets. A.A.
      # use ONLY the subset of variables if requested
      if set(['inputSpaceParams']).issubset(set(source['name'].keys())):
        sourceInputs = source['name']['inputSpaceParams'].keys()
        if specificVars is not None:
          inputHeaders = list(var for var in sourceInputs if var in specificVars)
        else:
          inputHeaders = sourceInputs
        inputHeaders = list(utils.toBytesIterative(inputHeaders))
        groups.attrs[b'inputSpaceHeaders' ] = inputHeaders

        if specificVars is not None:
          inputValues = list(source['name']['inputSpaceParams'][var] for var in sourceInputs if var in specificVars)
        else:
          inputValues = source['name']['inputSpaceParams'].values()
        inputValues = json.dumps(list(utils.toListFromNumpyOrC1arrayIterative(list(utils.toBytesIterative(inputValues)))))
        groups.attrs[b'inputSpaceValues'  ] = inputValues

      if set(['outputSpaceParams']).issubset(set(source['name'].keys())):
        if specificVars is not None:
          outDict = dict((k,v) for k,v in source['name']['outputSpaceParams'].items() if k in specificVars)
        else:
          outDict = source['name']['outputSpaceParams']
      else:
        if specificVars is not None:
          outDict = dict((key,value) for (key,value) in source['name'].iteritems() if key not in ['inputSpaceParams'] and key in specificVars)
        else:
          outDict = dict((key,value) for (key,value) in source['name'].iteritems() if key not in ['inputSpaceParams'])
      outHeaders = utils.toBytesIterative(list(outDict.keys()))
      outValues  = utils.toBytesIterative(list(outDict.values()))
      groups.attrs[b'nParams'   ] = len(outHeaders)
      groups.attrs[b'outputSpaceHeaders'] = outHeaders
      groups.attrs[b'EndGroup'   ] = True
      groups.attrs[b'parentID'  ] = parentName
      maxSize = 0
      for value in outValues:
        if type(value) == np.ndarray or type(value).__name__ == 'c1darray':
          if maxSize < value.size:
            actualOne = np.asarray(value).size
        elif type(value) in [int,float,bool,np.float64,np.float32,np.float16,np.int64,np.int32,np.int16,np.int8,np.bool8]:
          actualOne = 1
        else:
          self.raiseAnError(IOError,'The type of the dictionary parameters must be within float,bool,int,numpy.ndarray.Got '+type(value).__name__)
        if maxSize < actualOne:
          maxSize = actualOne
      groups.attrs[b'nTimeSteps'  ] = maxSize
      dataout = np.zeros((maxSize,len(outHeaders)))
      for index in range(len(outHeaders)):
        if type(outValues[index]) == np.ndarray or type(value).__name__ == 'c1darray':
          dataout[0:outValues[index].size,index] =  np.ravel(outValues[index])[:]
        else:
          dataout[0,index] = outValues[index]
      # create the data set
      groups.create_dataset(groupName + "_data", dtype="float", data=dataout)
      # add metadata if present
      for attr in attributes.keys():
        objectToConvert = mathUtils.convertNumpyToLists(attributes[attr])
        converted = json.dumps(objectToConvert)
        if converted and attr != 'name':
          groups.attrs[utils.toBytes(attr)]=converted
      if parentGroupName != "/":
        self.allGroupPaths.append(parentGroupName + "/" + groupName)
        self.allGroupEnds[parentGroupName + "/" + groupName] = True
      else:
        self.allGroupPaths.append("/" + groupName)
        self.allGroupEnds["/" + groupName] = True
    else:
      # Data(structure)
      # Retrieve the headers from the data (inputs and outputs)
      inpParams = source['name'].getInpParametersValues().keys()
      outParams = source['name'].getOutParametersValues().keys()
      if specificVars is not None:
        headersIn  = list(v for v in inpParams if v in specificVars)
        headersOut = list(v for v in outParams if v in specificVars)
      else:
        headersIn  = list(inpParams)
        headersOut = list(outParams)
      # for a "HistorySet" type we create a number of groups = number of HistorySet (compatibility with loading structure)
      if specificVars is not None:
        dataIn  = list(source['name'].getInpParametersValues()[v] for v in inpParams if v in specificVars)
        dataOut = list(source['name'].getOutParametersValues()[v] for v in outParams if v in specificVars)
      else:
        dataIn  = list(source['name'].getInpParametersValues().values())
        dataOut = list(source['name'].getOutParametersValues().values())
      # FIXME unused, but left commented because I'm not sure why they're unused.  PT
      #headersInUnstructured = list(source['name'].getInpParametersValues(self,unstructuredInputs=True).keys())
      #dataInUnstructured    = list(source['name'].getInpParametersValues(self,unstructuredInputs=True).values())
      metadata = source['name'].getAllMetadata()
      if source['name'].type in ['HistorySet','PointSet']:
        groups = []
        if 'HistorySet' in source['name'].type:
          nruns = len(dataIn)
        else:
          nruns = dataIn[0].size
        for run in range(nruns):
          if upGroup:
            groups.append(parentGroupObj.require_group(groupName + b'|' +str(run)))
            if (groupName + "_data") in groups[run]:
              del groups[run][groupName+"_data"]
          else:
            groups.append(parentGroupObj.create_group(groupName + '|' +str(run)))

          groups[run].attrs[b'sourceType'] = utils.toBytes(source['name'].type)
          groups[run].attrs[b'mainClass' ] = b'DataObjects'
          groups[run].attrs[b'EndGroup'   ] = True
          groups[run].attrs[b'parentID'  ] = parentName
          if source['name'].type == 'HistorySet':
            groups[run].attrs[b'inputSpaceHeaders' ] = [utils.toBytes(list(dataIn[run].keys())[i])  for i in range(len(dataIn[run].keys()))]
            groups[run].attrs[b'outputSpaceHeaders'] = [utils.toBytes(list(dataOut[run].keys())[i])  for i in range(len(dataOut[run].keys()))]
            json.dumps(list(utils.toListFromNumpyOrC1arrayIterative(list(dataIn[run].values()))))
            groups[run].attrs[b'inputSpaceValues'  ] = json.dumps(list(utils.toListFromNumpyOrC1arrayIterative(list(dataIn[run].values()))))
            groups[run].attrs[b'nParams'            ] = len(dataOut[run].keys())
            #collect the outputs
            dataout = np.zeros((next(iter(dataOut[run].values())).size,len(dataOut[run].values())))
            for param in range(len(dataOut[run].values())):
              dataout[:,param] = list(dataOut[run].values())[param][:]
            groups[run].create_dataset(groupName +'|' +str(run)+"_data" , dtype="float", data=dataout)
            groups[run].attrs[b'nTimeSteps'                ] = next(iter(dataOut[run].values())).size
          else:
            groups[run].attrs[b'inputSpaceHeaders' ] = [utils.toBytes(headersIn[i])  for i in range(len(headersIn))]
            groups[run].attrs[b'outputSpaceHeaders'] = [utils.toBytes(headersOut[i])  for i in range(len(headersOut))]
            groups[run].attrs[b'inputSpaceValues'  ] = json.dumps([list(utils.toListFromNumpyOrC1arrayIterative(np.atleast_1d(np.array(dataIn[x][run])).tolist())) for x in range(len(dataIn))])
            groups[run].attrs[b'nParams'            ] = len(headersOut)
            groups[run].attrs[b'nTimeSteps'                ] = 1
            #collect the outputs
            dataout = np.zeros((1,len(dataOut)))
            for param in range(len(dataOut)):
              dataout[0,param] = dataOut[param][run]
            groups[run].create_dataset(groupName +'|' +str(run)+"_data", dtype="float", data=dataout)
          # add metadata if present
          for attr in attributes.keys():
            objectToConvert = mathUtils.convertNumpyToLists(attributes[attr])
            converted = json.dumps(objectToConvert)
            if converted and attr != 'name':
              groups[run].attrs[utils.toBytes(attr)]=converted
          for attr in metadata.keys():
            if len(metadata[attr]) == nruns:
              toProcess = metadata[attr][run]
            else:
              toProcess = metadata[attr]
            if type(toProcess).__name__ == 'list' and 'input' in attr.lower() and isinstance(toProcess[0],Files.File):
              objectToConvert = list(a.__getstate__() for a in toProcess)
            elif isinstance(toProcess,Files.File):
              objectToConvert =toProcess.__getstate__()
            else:
              objectToConvert = mathUtils.convertNumpyToLists(toProcess)
            converted = json.dumps(objectToConvert)
            if converted and attr != 'name':
              groups[run].attrs[utils.toBytes(attr)]=converted

          if parentGroupName != "/":
            self.allGroupPaths.append(parentGroupName + "/" + groupName + '|' +str(run))
            self.allGroupEnds[parentGroupName + "/" + groupName + '|' +str(run)] = True
          else:
            self.allGroupPaths.append("/" + groupName + '|' +str(run))
            self.allGroupEnds["/" + groupName + '|' +str(run)] = True
      else:
        self.raiseAnError(IOError,'The function addGroupDataObjects accepts Data(s) or dictionaries as inputs only!!!!!')

  def __addSubGroup(self,groupName,attributes,source):
    """
      Function to add a group into the database (Hierarchical)
      @ In, groupName, string, group name
      @ In, attributes, dict, dictionary of attributes that must be added as metadata
      @ In, source, File object, source data
      @ Out, None
    """
    for index in xrange(len(self.allGroupPaths)):
      comparisonName = self.allGroupPaths[index]
      splittedPath=comparisonName.split('/')
      for splgroup in splittedPath:
        if groupName == splgroup and splittedPath[0] == self.parentGroupName:
          self.raiseAnError(IOError,"Group named " + groupName + " already present in database " + self.name + ". new group " + groupName + " is equal to old group " + comparisonName)
    if source['type'] == 'csv':
      # Source in CSV format
      f = open(source['name'],'rb')
      # Retrieve the headers of the CSV file
      headers = f.readline().split(b",")
      # Load the csv into a numpy array(n time steps, n parameters)
      data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
      # Check if the parent attribute is not null # In this case append a subgroup to the parent group
      # Otherwise => it's the main group
      parentID = None
      if 'metadata' in attributes.keys():
        if 'parentID' in attributes['metadata'].keys():
          parentID = attributes['metadata']['parentID']
      else:
        if 'parentID' in attributes.keys():
          parentID = attributes['parentID']

      if parentID:
        parentName = parentID
      else:
        self.raiseAnError(IOError,'NOT FOUND attribute <parentID> into <attributes> dictionary')
      # Find parent group path
      if parentName != '/':
        parentGroupName = self.__returnParentGroupPath(parentName)
      else:
        parentGroupName = parentName
      # Retrieve the parent group from the HDF5 database
      if parentGroupName in self.h5FileW:
        grp = self.h5FileW.require_group(parentGroupName)
      else:
        # try to guess the parentID from the file name
        head,tail = os.path.split(os.path.dirname(source['name']))
        testParentName = self.__returnParentGroupPath(tail[:-2])
        if testParentName in self.h5FileW:
          grp = self.h5FileW.require_group(testParentName)
        else:
          closestGroup = difflib.get_close_matches(parentName, self.allGroupPaths, n=1, cutoff=0.01)
          errorString = ' NOT FOUND parent group named "' + str(parentName)+'" for loading file '+str(source['name'])
          errorString+= '\n Tried '+str(tail[:-2])+ ' but not found as well. All group paths are:\n -'+'\n -'.join(self.allGroupPaths)
          errorString+= '\n Closest parent group found is "'+str(closestGroup[0] if len(closestGroup) > 0 else 'None')+'"!'
          self.raiseAnError(ValueError,errorString)
      # The parent group is not the endgroup for this branch
      self.allGroupEnds[parentGroupName] = False
      grp.attrs["EndGroup"]   = False
      self.raiseAMessage('Adding group named "' + groupName + '" in Database "'+ self.name +'"')
      # Create the sub-group
      sgrp = grp.create_group(groupName)
      # Create data set in this new group
      sgrp.create_dataset(groupName+"_data", dtype="float", data=data)
      # Add the metadata
      sgrp.attrs["outputSpaceHeaders"   ] = headers
      sgrp.attrs["nParams"  ] = data[0,:].size
      sgrp.attrs["parent"    ] = "root"
      sgrp.attrs["startTime"] = data[0,0]
      sgrp.attrs["end_time"  ] = data[data[:,0].size-1,0]
      sgrp.attrs["nTimeSteps"      ] = data[:,0].size
      sgrp.attrs["EndGroup"  ] = True
      sgrp.attrs["sourceType"] = source['type']
      if source['type'] == 'csv':
        sgrp.attrs["sourceFile"] = source['name']
      # add metadata if present
      for attr in attributes.keys():
        if attr == 'metadata':
          if 'SampledVars' in attributes['metadata'].keys():
            inpHeaders = []
            inpValues  = []
            for inkey, invalue in attributes['metadata']['SampledVars'].items():
              if inkey not in headers:
                inpHeaders.append(utils.toBytes(inkey))
                inpValues.append(invalue)
            if len(inpHeaders) > 0:
              sgrp.attrs[b'inputSpaceHeaders'] = inpHeaders
              sgrp.attrs[b'inputSpaceValues' ] = json.dumps(list(utils.toListFromNumpyOrC1arrayIterative(list(inpValues))))
        #Files objects are not JSON serializable, so we have to cover that.
        #this doesn't cover all possible circumstance, but it covers the DET case.
        if attr == 'inputFile' and isinstance(attributes[attr][0],Files.File):
          objectToConvert = list(a.__getstate__() for a in attributes[attr])
        else:
          objectToConvert = mathUtils.convertNumpyToLists(attributes[attr])
        converted = json.dumps(objectToConvert)
        if converted and attr != 'name':
          sgrp.attrs[utils.toBytes(attr)]=converted
    else:
      pass
    # The sub-group is the new ending group
    if parentGroupName != "/":
      self.allGroupPaths.append(parentGroupName + "/" + groupName)
      self.allGroupEnds[parentGroupName + "/" + groupName] = True
    else:
      self.allGroupPaths.append("/" + groupName)
      self.allGroupEnds["/" + groupName] = True
    return

  def computeBack(self,nameFrom,nameTo):
    """
      Function to compute the number of steps back from a group to another
      @ In,  nameFrom, string, group name (from)
      @ In,  nameTo, string, group name (to)
      @ Out, back, int, number of steps back (integer)
    """
    # "listStrW", list in which the path for a particular group is stored (working variable)
    # "path", string in which the path of the "to" group is stored
    # "found", bolean variable ... I would say...self explanable :D
    listStrW = []
    path       = ''
    found      = False
    # Find the path fo the "nameTo" group
    for i in xrange(len(self.allGroupPaths)):
      listStrW = self.allGroupPaths[i].split("/")
      if listStrW[len(listStrW)-1] == nameTo:
        found = True
        path  = self.allGroupPaths[i]
        break
    if not found:
      self.raiseAnError(NameError,"Group named " + nameTo + " not found in the HDF5 database" + self.filenameAndPath)
    else:
      listGroups = path.split("/")  # Split the path in order to create a list of the groups in this history
    # Retrieve indeces of groups "nameFrom" and "nameTo" v
    fr = listGroups.index(nameFrom)
    to = listGroups.index(nameTo)
    # Compute steps back
    back = to - fr
    return back

  def retrieveAllHistoryPaths(self,rootName=None):
    """
      Function to create a list of all the HistorySet paths present in an existing database
      @ In,  rootName, string, optional, It's the root name, if present, only the groups that have this root are going to be returned
      @ Out, allHistoryPaths, list, List of the HistorySet paths
    """
    allHistoryPaths = []
    # Create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen:
      self.__createObjFromFile()
    # Check database type
    if self.type == 'MC':
      # Parallel structure => "self.allGroupPaths" already contains the HistorySet' paths
      if not rootName:
        allHistoryPaths = self.allGroupPaths
      else:
        for index in xrange(len(self.allGroupPaths)):
          if rootName in self.allGroupPaths[index].split('/')[1]:
            allHistoryPaths.append(self.allGroupPaths[index])
    else:
      # Tree structure => construct the HistorySet' paths
      for index in xrange(len(self.allGroupPaths)):
        if self.allGroupEnds[self.allGroupPaths[index]]:
          if rootName and not (rootName in self.allGroupPaths[index].split('/')[1]):
            continue
          allHistoryPaths.append(self.allGroupPaths[index])
    return allHistoryPaths

  def retrieveAllHistoryNames(self,rootName=None):
    """
      Function to create a list of all the HistorySet names present in an existing database
      @ In,  rootName, string, optional, It's the root name, if present, only the history names that have this root are going to be returned
      @ Out, workingList, list, List of the HistorySet names
    """
    if not self.fileOpen:
      self.__createObjFromFile() # Create the "self.allGroupPaths" list from the existing database
    workingList = []
    for index in xrange(len(self.allGroupPaths)):
      if self.allGroupEnds[self.allGroupPaths[index]]:
        if rootName and not (rootName in self.allGroupPaths[index].split('/')[1]):
          continue
        workingList.append(self.allGroupPaths[index].split('/')[len(self.allGroupPaths[index].split('/'))-1])
    return workingList

  def retrieveHistory(self,name,filterHist=None,attributes = None):
    """
      Function to retrieve the history whose end group name is "name"
      @ In, name, string, history name => It must correspond to a group name (string)
      @ In, filterHist, string or int, optional, filter for history retrieving
                      ('whole' = whole history,
                       integer value = groups back from the group "name",
                       or None = retrieve only the group "name". Default is None)
      @ In, attributes, dict, optional, dictionary of attributes (options)
      @ Out, (result,attrs), tuple, tuple where position 0 = 2D numpy array (history), 1 = dictionary (metadata)
    """
    listStrW = []
    listPath  = []
    path       = ''
    found      = False
    result     = None
    attrs = {}
    if attributes:
      if 'inputTs' in attributes.keys():
        inputTs  = attributes['inputTs' ]
      if 'operator' in attributes.keys():
        operator = attributes['operator']
    else:
      inputTs  = None
      operator = None

    # Check if the h5 file is already open, if not, open it
    # and create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen:
      self.__createObjFromFile()
    # Find the endgroup that coresponds to the given name
    for i in xrange(len(self.allGroupPaths)):
      listStrW = self.allGroupPaths[i].split("/")
      try:
        listStrW.remove("")
      except ValueError:
        pass
      if listStrW[len(listStrW)-1] == name:
        found = True
        path  = self.allGroupPaths[i]
        listPath = listStrW
        break
    if found:
      # Check the filter type
      if not filterHist or filterHist == 0:
        # Grep only History from group "name"
        grp = self.h5FileW.require_group(path)
        # Retrieve dataset
        dataset = grp.require_dataset(name +'_data', (int(grp.attrs['nTimeSteps']),int(grp.attrs['nParams'])), dtype='float').value
        # Get numpy array
        result = dataset[:,:]
        # Get attributes (metadata)
        attrs = grp.attrs
        for attr in attrs.keys():
          try:
            attrs[attr] = json.loads(attrs[attr])
          except:
            attrs[attr] = attrs[attr]
      elif  filterHist == 'whole':
        # Retrieve the whole history from group "name" to the root
        # Start constructing the merged numpy array
        whereList = []
        nameList  = []
        # back represents the number of groups back that need to be
        # included in the construction of the full history.
        if self.parentGroupName != '/':
          back = len(listPath)-2
        else:
          back = len(listPath)-1
        if back < 0:
          back = 0
        i=0
        #Question, should all the "" be removed, or just the first?
        listPath = list(filter(None,listPath))
        # Find the paths for the completed history
        while (i <= back):
          pathW = ''
          for j in xrange(len(listPath) - i):
            pathW = pathW + "/" + listPath[j]
          if pathW != "":
            whereList.append(pathW)
          mylist = whereList[i].split("/")
          nameList.append(mylist[len(mylist)-1])
          i = i + 1
        # get the relative groups' data
        gbRes ={}
        gbAttrs={}
        whereList.reverse()
        nameList.reverse()
        nToTS = 0
        nParams = 0
        # Retrieve every single partial history that will be merged to create the whole history
        for i in xrange(len(whereList)):
          grp = self.h5FileW.require_group(whereList[i])
          namel = nameList[i] +'_data'
          dataset = grp.require_dataset(namel, (int(grp.attrs['nTimeSteps']),int(grp.attrs['nParams'])), dtype='float').value
          if i == 0:
            nParams = int(grp.attrs['nParams'])
          if nParams != int(grp.attrs['nParams']):
            self.raiseAnError(TypeError,'Can not merge datasets with different number of parameters')
          # Get numpy array
          gbRes[i]   = dataset[:,:]
          gbAttrs[i] = copy.copy(grp.attrs   )
          nToTS = nToTS + int(grp.attrs['nTimeSteps'])
        # Create the numpy array
        result = np.zeros((nToTS,nParams))
        ts = 0
        # Retrieve the metadata
        for key in gbRes:
          arr = gbRes[key]
          result[ts:ts+arr[:,0].size,:] = arr
          ts = ts + arr[:,0].size
          # must be checked if overlapping of time (branching for example)
        try:
          attrs["outputSpaceHeaders"]   = gbAttrs[len(whereList)-1]["outputSpaceHeaders"].tolist()
        except:
          attrs["outputSpaceHeaders"]   = gbAttrs[len(whereList)-1]["outputSpaceHeaders"]
        try:
          attrs["inputSpaceHeaders"]    = gbAttrs[len(whereList)-1]["inputSpaceHeaders"].tolist()
        except:
          try:
            attrs["inputSpaceHeaders"]  = gbAttrs[len(whereList)-1]["inputSpaceHeaders"]
          except:
            pass
        try:
          attrs["inputSpaceValues"]     = list(utils.toListFromNumpyOrC1arrayIterative(list(gbAttrs[0]["inputSpaceValues"].tolist())))
        except:
          try:
            attrs["inputSpaceValues"]   = list(utils.toListFromNumpyOrC1arrayIterative(list(json.loads(gbAttrs[len(whereList)-1]["inputSpaceValues"]))))
          except:
            pass
        attrs["nParams"]        = gbAttrs[len(whereList)-1]["nParams"]
        attrs["parentID"]       = whereList[0]
        attrs["startTime"]      = result[0,0]
        attrs["end_time"]       = result[result[:,0].size-1,0]
        attrs["nTimeSteps"]     = result[:,0].size
        attrs["sourceType"]     = gbAttrs[len(whereList)-1]["sourceType"]
        attrs["inputFile"]      = []
        attrs["sourceFile"]     = []
        for key in gbRes.keys():
          for attr in gbAttrs[key].keys():
            if attr not in ["outputSpaceHeaders","inputSpaceHeaders","inputSpaceValues","nParams","parentID","startTime","end_time","nTimeSteps","sourceType"]:
              if attr not in attrs.keys():
                attrs[attr] = []
              try:
                attrs[attr].append(json.loads(gbAttrs[key][attr]))
              except:
                if type(attrs[attr]) == list:
                  attrs[attr].append(gbAttrs[key][attr])
          if attrs["sourceType"] == 'csv' and 'sourceFile' in gbAttrs[key].keys():
            attrs["sourceFile"].append(gbAttrs[key]["sourceFile"])
      else:
        # A number of groups' back have been inputted
        # Follow the same strategy used above (filterHist = whole)
        # but stop at back(th) group starting from group "name"
        if is_number(filterHist):
          back = int(filterHist) + 1
          if len(listPath) < back:
            self.raiseAnError(RuntimeError,'Number of branches back > number of actual branches in dataset for History ending with ' + name)
          if (back == len(listPath)-1) and (self.parentGroupName != '/'):
            back = back - 1
          # start constructing the merged numpy array
          whereList = []
          nameList  = []
          i=0
          listPath = list(filter(None,listPath))
          # Find the paths for the completed history
          while (i < back):
            pathW = ''
            for j in xrange(len(listPath) - i):
              pathW = pathW + '/' + listPath[j]
            if pathW != "":
              whereList.append(pathW)
            mylist = whereList[i].split("/")
            nameList.append(mylist[len(mylist)-1])
            i = i + 1
          # get the relative groups' data
          gbRes ={}
          gbAttrs={}
          whereList.reverse()
          nameList.reverse()
          nToTS = 0
          nParams = 0
          # Retrieve every single partial history that will be merged to create the whole history
          for i in xrange(len(whereList)):
            grp = self.h5FileW.require_group(whereList[i])
            namel = nameList[i] +'_data'
            dataset = grp.require_dataset(namel, (int(grp.attrs['nTimeSteps']),int(grp.attrs['nParams'])), dtype='float').value
            if i == 0:
              nParams = int(grp.attrs['nParams'])

            if nParams != int(grp.attrs['nParams']):
              self.raiseAnError(TypeError,'Can not merge datasets with different number of parameters')
            # get numpy array
            gbRes[i]   = dataset[:,:]
            gbAttrs[i] =grp.attrs
            nToTS = nToTS + int(grp.attrs['nTimeSteps'])
          # Create numpy array
          result = np.zeros((nToTS,nParams))
          ts = 0
          # Retrieve metadata
          for key in gbRes:
            arr = gbRes[key]
            result[ts:ts+arr[:,0].size,:] = arr[:,:]
            ts = ts + arr[:,0].size
            # must be checked if overlapping of time (branching for example)
          try:
            attrs["outputSpaceHeaders"]   = gbAttrs[len(whereList)-1]["outputSpaceHeaders"].tolist()
          except:
            attrs["outputSpaceHeaders"]   = gbAttrs[len(whereList)-1]["outputSpaceHeaders"]
          try:
            attrs["inputSpaceHeaders"]    = gbAttrs[len(whereList)-1]["inputSpaceHeaders"].tolist()
          except:
            try:
              attrs["inputSpaceHeaders"]  = gbAttrs[len(whereList)-1]["inputSpaceHeaders"]
            except:
              pass
          try:
            attrs["inputSpaceValues"]     = list(utils.toListFromNumpyOrC1arrayIterative(list(gbAttrs[0]["inputSpaceValues"].tolist())))
          except:
            try:
              attrs["inputSpaceValues"]   = list(utils.toListFromNumpyOrC1arrayIterative(list(json.loads(gbAttrs[0]["inputSpaceValues"]))))
            except:
              pass
          attrs["nParams"]        = gbAttrs[len(whereList)-1]["nParams"]
          attrs["parent"]         = whereList[0]
          attrs["startTime"]      = result[0,0]
          attrs["end_time"]       = result[result[:,0].size-1,0]
          attrs["nTimeSteps"]     = result[:,0].size
          attrs["sourceType"]     = gbAttrs[len(whereList)-1]["sourceType"]
          attrs["inputFile"]      = []
          attrs["sourceFile"]     = []
          for key in gbRes.keys():
            for attr in gbAttrs[key].keys():
              if attr not in ["outputSpaceHeaders","inputSpaceHeaders","inputSpaceValues","nParams","parentID","startTime","end_time","nTimeSteps","sourceType"]:
                if attr not in attrs.keys():
                  attrs[attr] = []
                try:
                  attrs[attr].append(json.loads(gbAttrs[key][attr]))
                except:
                  if type(attrs[attr]) == list:
                    attrs[attr].append(gbAttrs[key][attr])
              if attrs["sourceType"] == 'csv':
                attrs["sourceFile"].append(gbAttrs[key]["sourceFile"])

        else:
          self.raiseAnError(IOError,'Filter not recognized in hdf5Database.retrieveHistory function. Filter = ' + str(filter))
    else:
      self.raiseAnError(IOError,'History named ' + name + ' not found in database')

    return(result,attrs)

  def closeDatabaseW(self):
    """
      Function to close the database
      @ In,  None
      @ Out, None
    """
    self.h5FileW.close()
    self.fileOpen       = False
    return

  def openDatabaseW(self,filename,mode='w'):
    """
      Function to open the database
      @ In, filename, string, name of the file (string)
      @ In, mode, string, open mode (default "w=write")
      @ Out, fh5, hdf5 object, instance of hdf5
    """
    fh5 = h5.File(filename,mode)
    self.fileOpen       = True
    return fh5

  def __returnParentGroupPath(self,parentName):
    """
      Function to return a parent group Path
      @ In, parentName, string, parent ID
      @ Out, parentGroupName, string, parent group path
    """
    if parentName != '/':
      parentGroupName = '-$' # control variable
      for index in xrange(len(self.allGroupPaths)):
        testList = self.allGroupPaths[index].split('/')
        if testList[-1] == parentName:
          parentGroupName = self.allGroupPaths[index]
          break
    else:
      parentGroupName = '/'
    return parentGroupName

def is_number(s):
  """
    Function to check if s is a number
    @ In, s, object, the object to checkt
    @ Out, response, bool, is it a number?
  """
  response = False
  try:
    float(s)
    response = True
  except ValueError:
    pass
  return response
