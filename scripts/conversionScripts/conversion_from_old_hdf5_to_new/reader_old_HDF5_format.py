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
Created on Feb 03, 2018
based on h5py_interface_creator.py (commit 3469eddfcacec2ab1dc93786ac068cebff5d6e05)
This module is a stripped version of the h5py_interface_creator.py module present in devel
(3469eddfcacec2ab1dc93786ac068cebff5d6e05). It is able to read databases generated after Jan 2015.
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
import sys
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from utils import mathUtils
#Internal Modules End--------------------------------------------------------------------------------

#
#  *************************
#  *  HDF5 DATABASE CLASS  *
#  *************************
#

class OldHDF5Database():
  """
    class to create a h5py (hdf5) database
  """
  def __init__(self,name, databaseDir,filename):
    """
      Constructor
      @ In, name, string, name of this database
      @ In, databaseDir, string, database directory (full path)
      @ In, filename, string, the database filename
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
    # self.h5FileW is the HDF5 object. Open the database in "update" mode
    # check if it exists
    if not os.path.exists(self.filenameAndPath):
      raise IOError('database file has not been found, searched Path is: ' + self.filenameAndPath )
    # Open file
    self.h5FileW = self.openDatabaseW(self.filenameAndPath,'r+')
    # Call the private method __createObjFromFile, that constructs the list of the paths "self.allGroupPaths"
    # and the dictionary "self.allGroupEnds" based on the database that already exists
    self.parentGroupName = b'/'
    self.__createObjFromFile()
    # "self.firstRootGroup", true if the root group is present (or added), false otherwise
    self.firstRootGroup = True

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
    print('TOTAL NUMBER OF GROUPS = ' + str(len(self.allGroupPaths)))

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
        print('Warning: not found attribute EndGroup in group ' + name + '.Set True.')
    return

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
          raise IOError('Filter not recognized in hdf5Database.retrieveHistory function. Filter = ' + str(filter))
    else:
      raise IOError('History named ' + name + ' not found in database')

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
