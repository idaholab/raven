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
    self.h5FileW.attrs['allGroupPaths'] = json.dumps(self.allGroupPaths)
    self.h5FileW.attrs['allGroupEnds'] = json.dumps(self.allGroupEnds)  
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
    grp.attrs['endGroup'] = False
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
    if self.__returnParentGroupPath(groupName) != '-$':
      # the group alread exists 
      groupName = groupName + "_" + groupName

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
    
    # I keep this structure here because I want to maintain the possibility to add a whatever dictionary even if not prepared and divided into output and input sub-sets. A.A.
    # use ONLY the subset of variables if requested
    
    # add pointwise metadata (in this case, they are group-wise)
    groups.attrs[b'point_wise_metadata_keys'] = json.dumps(self._metavars)
    
    # get the data
    data = dict( (key, value) for (key, value) in rlz.items() if type(value) == np.ndarray )
    # get size of each data variable
    varKeys = data.keys()
    varShape = [data[key].shape for key in varKeys]
    # get data names
    groups.attrs[b'data_names'] = json.dumps(varKeys)
    # get data shapes
    groups.attrs[b'data_shapes'] = json.dumps(varShape)
    # get data shapes
    end   = np.cumsum(varShape)
    begin = np.concatenate(([0],end[0:-1]))
    groups.attrs[b'data_begin_end'] = json.dumps((begin.tolist(),end.tolist()))    
    # get data names
    groups.create_dataset(groupName + "_data", dtype="float", data=(np.concatenate( data.values()).ravel()))
    # add some info
    groups.attrs[b'endGroup'   ] = True
    groups.attrs[b'parentID'   ] = parentName
    groups.attrs[b'nVars'      ] = len(varKeys)
    ## get the data back
    ##dataset = groups[groupName + "_data"]
    ### reshape them based on the shapes
    ##newData = {key : np.reshape(dataset[begin[cnt]:end[cnt]], varShape[cnt]) for cnt,key in enumerate(varKeys)} 
    if parentGroupName != "/":
      self.allGroupPaths.append(parentGroupName + "/" + groupName)
      self.allGroupEnds[parentGroupName + "/" + groupName] = True
    else:
      self.allGroupPaths.append("/" + groupName)
      self.allGroupEnds["/" + groupName] = True

  def __addSubGroup(self,groupName,attributes,source):
    """
      Function to add a group into the database (Hierarchical)
      @ In, groupName, string, group name
      @ In, attributes, dict, dictionary of attributes that must be added as metadata
      @ In, source, File object, source data
      @ Out, None
    """
    
    if self.__returnParentGroupPath(groupName) != '-$':
      # the group alread exists 
      groupName = groupName + "_" + groupName
    
    # retrieve parentID
    parentID = rlz.get("parentID")
    parentName = parentID
  
    # Find parent group path
    if parentName != '/':
      parentGroupName = self.__returnParentGroupPath(parentName)
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
        testParentName = closestGroup[0]   
        if testParentName in self.h5FileW:
          parentGroupObj = self.h5FileW.require_group(testParentName)   
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
    groups = parentGroupObj.create_group(groupName)
    # add pointwise metadata (in this case, they are group-wise)
    groups.attrs[b'point_wise_metadata_keys'] = json.dumps(self._metavars)
    # get the data
    data = dict( (key, value) for (key, value) in rlz.items() if type(value) == np.ndarray )
    # get size of each data variable
    varKeys = data.keys()
    varShape = [data[key].shape for key in varKeys]
    # get data names
    groups.attrs[b'data_names'] = json.dumps(varKeys)
    # get data shapes
    groups.attrs[b'data_shapes'] = json.dumps(varShape)
    # get data shapes
    end   = np.cumsum(varShape)
    begin = np.concatenate(([0],end[0:-1]))
    groups.attrs[b'data_begin_end'] = json.dumps((begin.tolist(),end.tolist()))    
    # get data names
    groups.create_dataset(groupName + "_data", dtype="float", data=(np.concatenate( data.values()).ravel()))
    # add some info
    groups.attrs[b'endGroup'   ] = True
    groups.attrs[b'parentID'   ] = parentName
    groups.attrs[b'nVars'      ] = len(varKeys)
    # The sub-group is the new ending group
    if parentGroupName != "/":
      self.allGroupPaths.append(parentGroupName + "/" + groupName)
      self.allGroupEnds[parentGroupName + "/" + groupName] = True
    else:
      self.allGroupPaths.append("/" + groupName)
      self.allGroupEnds["/" + groupName] = True
    return

  #def computeBack(self,nameFrom,nameTo):
    #"""
      #Function to compute the number of steps back from a group to another
      #@ In,  nameFrom, string, group name (from)
      #@ In,  nameTo, string, group name (to)
      #@ Out, back, int, number of steps back (integer)
    #"""
    ## "listStrW", list in which the path for a particular group is stored (working variable)
    ## "path", string in which the path of the "to" group is stored
    ## "found", bolean variable ... I would say...self explanable :D
    #listStrW = []
    #path       = ''
    #found      = False
    ## Find the path fo the "nameTo" group
    #for i in xrange(len(self.allGroupPaths)):
      #listStrW = self.allGroupPaths[i].split("/")
      #if listStrW[len(listStrW)-1] == nameTo:
        #found = True
        #path  = self.allGroupPaths[i]
        #break
    #if not found:
      #self.raiseAnError(NameError,"Group named " + nameTo + " not found in the HDF5 database" + self.filenameAndPath)
    #else:
      #listGroups = path.split("/")  # Split the path in order to create a list of the groups in this history
    ## Retrieve indeces of groups "nameFrom" and "nameTo" v
    #fr = listGroups.index(nameFrom)
    #to = listGroups.index(nameTo)
    ## Compute steps back
    #back = to - fr
    #return back

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
    # Find the endGroup that coresponds to the given name
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

