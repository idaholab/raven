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
    if 'allGroupPaths' in self.h5FileW.attrs and 'allGroupEnds' in self.h5FileW.attrs:
      self.allGroupPaths = json.loads(self.h5FileW.attrs['allGroupPaths'])
      self.allGroupEnds  = json.loads(self.h5FileW.attrs['allGroupEnds'])
    else:
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
        self.allGroupEnds[name]  = obj.attrs["endGroup"]
      except KeyError:
        self.allGroupEnds[name]  = True
      if "rootname" in obj.attrs:
        self.parentGroupName = name
        self.raiseAWarning('not found attribute endGroup in group ' + name + '.Set True.')
    return

  def addGroup(self,rlz):
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
        self.__addGroupRootLevel(groupName,rlz)
        self.firstRootGroup = True
        self.type = 'DET'
      else:
        # Add sub group in the Hierarchical structure
        self.__addSubGroup(groupName,rlz)
      endif
    else:
      # Parallel structure (always root level)
      self.__addGroupRootLevel(groupName,rlz)
      self.firstRootGroup = True
      self.type = 'MC'      
    self.h5FileW.attrs['allGroupPaths'] = json.dumps(self.allGroupPaths)
    self.h5FileW.attrs['allGroupEnds'] = json.dumps(self.allGroupEnds)  
    self.h5FileW.flush()
    

  def addGroupInit(self,groupName,attributes={},upGroup=False):
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
    grp.attrs.update(attributes)
    grp.attrs['rootname'] = True
    grp.attrs['endGroup'] = False
    self.allGroupPaths.append("/" + groupNameInit)
    self.allGroupEnds["/" + groupNameInit] = False
    self.h5FileW.attrs['allGroupPaths'] = json.dumps(self.allGroupPaths)
    self.h5FileW.attrs['allGroupEnds'] = json.dumps(self.allGroupEnds)      
    self.h5FileW.flush()
  
  def __populateGroup(self, group, rlz):
    """
      This method is a common method between the __addGroupRootLevel and __addSubGroup
      It is used to populate the group with the info in the rlz
      @ In, group, h5py.Group, the group instance
      @ In, rlz, dict, dictionary with the data and metadata to add
      @ Out, None
    """
    # add pointwise metadata (in this case, they are group-wise)
    group.attrs[b'point_wise_metadata_keys'] = json.dumps(self._metavars)
    # get the data
    data = dict( (key, value) for (key, value) in rlz.items() if type(value) == np.ndarray )
    # get size of each data variable
    varKeys = data.keys()
    varShape = [data[key].shape for key in varKeys]
    # get data names
    group.attrs[b'data_names'] = json.dumps(varKeys)
    # get data shapes
    group.attrs[b'data_shapes'] = json.dumps(varShape)
    # get data shapes
    end   = np.cumsum(varShape)
    begin = np.concatenate(([0],end[0:-1]))
    group.attrs[b'data_begin_end'] = json.dumps((begin.tolist(),end.tolist()))    
    # get data names
    group.create_dataset(group.name + "_data", dtype="float", data=(np.concatenate( data.values()).ravel()))
    # add some info
    group.attrs[b'endGroup'   ] = True
    group.attrs[b'parentID'   ] = group.parent.name
    group.attrs[b'nVars'      ] = len(varKeys)

  
  def __addGroupRootLevel(self,groupName,rlz):
    """
      Function to add a group into the database (root level)
      @ In, groupName, string, group name
      @ In, rlz, dict, dictionary with the data and metadata to add
      @ Out, None
    """
    # Check in the "self.allGroupPaths" list if a group is already present...
    # If so, error (Deleting already present information is not desiderable)
    if self.__returnGroupPath(groupName) != '-$':
      # the group alread exists 
      groupName = groupName + "_" + groupName

    parentName = self.parentGroupName.replace('/', '')
    # Create the group
    parentGroupName = self.__returnGroupPath(parentName)
    # Retrieve the parent group from the HDF5 database
    if parentGroupName in self.h5FileW:
      parentGroupObj = self.h5FileW.require_group(parentGroupName)
    else:
      self.raiseAnError(ValueError,'NOT FOUND group named ' + parentGroupObj)
    # create and populate the group
    self.__populateGroup(parentGroupObj.create_group(groupName), rlz)
    # update lists
    self.__updateGroupLists(groupName, parentGroupName)

  def __addSubGroup(self,groupName,rlz):
    """
      Function to add a group into the database (Hierarchical)
      @ In, groupName, string, group name
      @ In, rlz, dict, dictionary with the data and metadata to add
      @ Out, None
    """
    if self.__returnGroupPath(groupName) != '-$':
      # the group alread exists 
      groupName = groupName + "_" + groupName
    
    # retrieve parentID
    parentID = rlz.get("parentID")
    parentName = parentID
  
    # Find parent group path
    if parentName != '/':
      parentGroupName = self.__returnGroupPath(parentName)
    else:
      parentGroupName = parentName
    # Retrieve the parent group from the HDF5 database
    if parentGroupName in self.h5FileW:
      parentGroupObj = self.h5FileW.require_group(parentGroupName)
    else:
      # try to guess the parentID from the file name
      closestGroup = difflib.get_close_matches(parentName, self.allGroupPaths, n=1, cutoff=0.01)
      errorOut = False
      if len(closestGroup) > 0:
        parentGroupName = closestGroup[0]   
        if parentGroupName in self.h5FileW:
          parentGroupObj = self.h5FileW.require_group(parentGroupName)   
        else:
          errorOut = True
      else:
        errorOut = True
      if errorOut:
        errorString = ' NOT FOUND parent group named "' + str(parentName)
        errorString+= '\n All group paths are:\n -'+'\n -'.join(self.allGroupPaths)
        errorString+= '\n Closest parent group found is "'+str(closestGroup[0] if len(closestGroup) > 0 else 'None')+'"!'
        self.raiseAnError(ValueError,errorString)    
    
    parentGroupObj.attrs[b'endGroup'   ] = False  
    # create the sub group
    self.raiseAMessage('Adding group named "' + groupName + '" in Database "'+ self.name +'"')
    # create and populate the group
    self.__populateGroup(parentGroupObj.create_group(groupName), rlz)
    # update lists
    self.__updateGroupLists(groupName, parentGroupName)
    
  
  def __updateGroupLists(self,groupName, parentName):
    """
      Utility method to update the group lists
      @ In, groupName, str, the new group added
      @ In, parentName, str, the parent name
      @ Out, None
    """
    if parentName != "/":
      self.allGroupPaths.append(parentName + "/" + groupName)
      self.allGroupEnds[parentName + "/" + groupName] = True
    else:
      self.allGroupPaths.append("/" + groupName)
      self.allGroupEnds["/" + groupName] = True    
    
  def retrieveAllHistoryPaths(self,rootName=None):
    """
      Function to create a list of all the HistorySet paths present in an existing database
      @ In,  rootName, string, optional, It's the root name, if present, only the groups that have this root are going to be returned
      @ Out, allHistoryPaths, list, List of the HistorySet paths
    """
    if rootName:
      rname = rootName 
    # Create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen:
      self.__createObjFromFile()
    # Check database type
    if self.type == 'MC':
      # Parallel structure => "self.allGroupPaths" already contains the HistorySet' paths
      if not rootName:
        allHistoryPaths = self.allGroupPaths
      else:
        allHistoryPaths = [k for k in self.allGroupPaths.keys() if k.endswith(rname)] 
    else:
      # Tree structure => construct the HistorySet' paths
      if not rootName:
        allHistoryPaths = [k for k, v in self.allGroupPaths.items() if v ]
      else:
        allHistoryPaths = [k for k, v in self.allGroupPaths.items() if v and k.endswith(rname)]           
    return allHistoryPaths

  def retrieveAllHistoryNames(self,rootName=None):
    """
      Function to create a list of all the HistorySet names present in an existing database
      @ In,  rootName, string, optional, It's the root name, if present, only the history names that have this root are going to be returned
      @ Out, workingList, list, List of the HistorySet names
    """
    if rootName:
      rname = rootName     
    if not self.fileOpen:
      self.__createObjFromFile() # Create the "self.allGroupPaths" list from the existing database
    if not rootName:  
      workingList = [k.split('/')[-1] for k, v in self.allGroupPaths.items() if v ]  
    else:
      workingList = [k.split('/')[-1] for k, v in self.allGroupPaths.items() if v and k.endswith(rname)]
    
    return workingList
  
  def __getListOfParentGroups(self, grp, backGroups = []):
    """
      Method to get the list of groups from the deepest to the root, given a certain group
      @ In, grp, h5py.Group, istance of the starting group
      @ InOut, backGroups, list, list of group instances (from the deepest to the root)
    """
    if grp.parent and grp.parent != grp:
      backGroups.append(grp.parent)
      self.__getListOfParentGroups(grp.parent, backGroups)
    return backGroups
    
  def _(self,name,options = {}):
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
    reconstruct = options.get("reconstruct", True)
    pivotParam  = options.get("pivotParam", "time")
    
    listStrW = []
    listPath  = []
    path       = ''
    found      = False
    result     = None
    attrs = {}
    # Check if the h5 file is already open, if not, open it
    # and create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen:
      self.__createObjFromFile()
    # Find the endGroup that coresponds to the given name
    path = self.__returnGroupPath(name)
    found = path != '-$'

    if found:
      # check the reconstruct flag
      
      # Grep only history from group "name"
      grp = self.h5FileW.require_group(path)
      # Retrieve dataset
      dataset = group[groupName + "_data"]
      # Get some variables of interest
      nVars      = json.loads(group.attrs[b'nVars'])
      varShape   = json.loads(group.attrs[b'data_shapes'])
      varKeys    = json.loads(group.attrs[b'data_names'])
      end, begin = json.loads(group.attrs[b'data_begin_end']) 
      # Reconstruct the dataset
      newData = {key : np.reshape(dataset[begin[cnt]:end[cnt]], varShape[cnt]) for cnt,key in enumerate(varKeys)} 
      # Add the attributes
      attrs = {'nVars':nVars,'varShape':varShape,'varKeys':varKeys}
      if reconstruct:
        # get list of back groups
        listGroups = self.__getListOfParentGroups(grp)
        listGroups.reverse()
        for group in listGroups:
          print(group.name)
    else:
      self.raiseAnError(IOError,'Group named ' + name + ' not found in database "'+self.name+'"!')

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

  def __returnGroupPath(self,parentName):
    """
      Function to return a group Path
      @ In, parentName, string, parent ID
      @ Out, parentGroupName, string, parent group path
    """
    parentGroupName = '-$' # control variable
    if parentName != '/':
      # this loops takes ~.2 seconds on a 100 milion list (it is accetable)
      for s in self.allGroupPaths:
        if s.endswith("/"+parentName.strip()):
          parentGroupName = s
          break
    else:
      parentGroupName = '/'
    return parentGroupName

