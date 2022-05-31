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
from datetime import datetime
import os
import pickle as pk
import string
import difflib
import h5py  as h5
import numpy as np

from .utils import utils, mathUtils
from .BaseClasses import InputDataUser, MessageUser

# the database version should be modified
# everytime a new modification of the internal
# structure of the data is performed
_hdf5DatabaseVersion = "v2.1"

def _dumps(val, void=True):
  """
    Method to convert an arbitary value to something h5py can store
    @ In, val, any, data to encode
    @ In, void, bool, optional, use np void to cast the pickled data?
    @ Out, _dumps, np.void, encoded data
  """
  serialized = pk.dumps(val, protocol=0)

  return np.void(serialized) if void else serialized

def _loads(val):
  """
    Method to undo what _dumps does
    @ In, val, np.void, data to decode
    @ Out, _loads, any, data decoded
  """
  if hasattr(val, 'tostring'):
    try:
      return pk.loads(val.tostring())
    except UnicodeDecodeError:
      return pk.loads(val.tostring(),errors='backslashreplace')
  else:
    try:
      return pk.loads(val)
    except UnicodeDecodeError:
      return pk.loads(val,errors='backslashreplace')

def _checkTypeHDF5(value, neg):
  """
    Local utility function to check the type
    @ In, value, object, the value to check
    @ In, neg, bool, to use the "not" or not
    @ Out, check, bool, the check
  """
  scalarNumpy = mathUtils.getNumpyTypes('float') + mathUtils.getNumpyTypes('int') + mathUtils.getNumpyTypes('uint')
  scalarBultins = mathUtils.getBuiltinTypes('float') + mathUtils.getBuiltinTypes('int')
  if neg:
    check = isinstance(value, np.ndarray) and value.dtype not in scalarNumpy and type(value) not in scalarBultins
  else:
    check = isinstance(value, np.ndarray) and value.dtype in scalarNumpy or type(value) in scalarBultins

  return check

#
#  *************************
#  *  HDF5 DATABASE CLASS  *
#  *************************
#

class hdf5Database(InputDataUser, MessageUser):
  """
    class to create a h5py (hdf5) database
  """
  def __init__(self,name, databaseDir, filename, exist, variables=None):
    """
      Constructor
      @ In, name, string, name of this database
      @ In, databaseDir, string, database directory (full path)
      @ In, filename, string, the database filename
      @ In, exist, bool, does it exist?
      @ In, variables, list, the user wants to store just some specific variables (default =None => all variables are stored)
      @ Out, None
    """
    super().__init__()
    self.name = name       # database name (i.e. arbitrary name) found in the xml input
    self.variables = variables
    self.type = None       # Database type -> "inferred" by the first group is added
                           # * MC  = MonteCarlo => Storing by a Parallel structure
                           # * DET = Dynamic Event Tree => Storing by a Hierarchical structure
    self.printTag = 'DATABASE HDF5' # specialize printTag
    self.fileExist = exist # does it exist?
    self.onDiskFile = filename # .H5 file name (to be created or read) on disk
    self.databaseDir =  databaseDir # Database directory
    self.filenameAndPath = os.path.join(self.databaseDir, self.onDiskFile)
    self.fileOpen = False
    self.allGroupPaths = [] # List of the paths of all the groups that are stored in the database
    # List of boolean variables, true if the corresponding group in self.allGroupPaths
    # is an ending group (no sub-groups appended), false otherwise
    self.allGroupEnds = []
    # We can create a base empty database or we open an existing one
    if self.fileExist:
      # self.h5FileW is the HDF5 object. Open the database in "update" mode
      # check if it exists
      if not os.path.exists(self.filenameAndPath):
        self.raiseAnError(IOError, 'database file has not been found, searched Path is: ' + self.filenameAndPath )
      # Open file
      self.h5FileW = self.openDatabaseW(self.filenameAndPath, 'r+')
      # check version
      version = self.h5FileW.attrs.get("version", "None")
      if version != _hdf5DatabaseVersion:
        self.raiseAnError(IOError, 'HDF5 RAVEN version (read mode) is outdated. ' +
                          f'Current version is "{_hdf5DatabaseVersion}". ' +
                          f'Version in HDF5 is "{version}".' +
                          'Read README file in folder ' +
                          '"raven/scripts/conversionScripts/conversion_hdf5"' +
                          ' to convert your outdated HDF5 into the new format!')

      # Call the private method __createObjFromFile, that constructs the list of the paths "self.allGroupPaths"
      # and the list "self.allGroupEnds" based on the database that already exists
      self.parentGroupName = '/'
      self.__createObjFromFile()
      # "self.firstRootGroup", true if the root group is present (or added), false otherwise
      self.firstRootGroup = True
    else:
      # self.h5FileW is the HDF5 object. Open the database in "write only" mode
      self.h5FileW = self.openDatabaseW(self.filenameAndPath,'w')
      # Add the root as first group
      self.allGroupPaths.append(b"/")
      # The root group is not an end group
      self.allGroupEnds.append(False)
      # The first root group has not been added yet
      self.firstRootGroup = False
      # The root name is / . it can be changed if addGroupInit is called
      self.parentGroupName = '/'
      self.__createFileLevelInfoDatasets()

  def __len__(self):
    """
      Overload len method
      @ In, None
      @ Out, __len__, length
    """
    return len(self.allGroupPaths)

  def __createFileLevelInfoDatasets(self):
    """
      Method to create datasets that are at the File Level and contains general info
      to recontruct HDF5 in loading mode
      @ In, None
      @ Out, None
    """
    self.h5FileW.attrs["version"] = _hdf5DatabaseVersion
    self.h5FileW.create_dataset("allGroupPaths", shape=(len(self.allGroupPaths),), dtype=h5.special_dtype(vlen=str), data=self.allGroupPaths, maxshape=(None,))
    self.h5FileW["allGroupPaths"].resize((max(len(self.allGroupPaths)*2,2000),))
    self.h5FileW.create_dataset("allGroupEnds", shape=(len(self.allGroupEnds),), dtype=bool, data=self.allGroupEnds, maxshape=(None,))
    self.h5FileW["allGroupEnds"].resize((max(len(self.allGroupPaths)*2,2000),))

  def __updateFileLevelInfoDatasets(self):
    """
      Method to create datasets that are at the File Level and contains general info
      to recontruct HDF5 in loading mode
      @ In, None
      @ Out, None
    """
    if len(self.allGroupPaths) > len(self.h5FileW["allGroupPaths"]):
      self.h5FileW["allGroupPaths"].resize((len(self.allGroupPaths)*2,))
      self.h5FileW["allGroupEnds"].resize( (len(self.allGroupPaths)*2,) )
    self.h5FileW["allGroupPaths"][len( self.allGroupPaths) - 1] = self.allGroupPaths[-1]
    self.h5FileW["allGroupEnds"][len(self.allGroupPaths) - 1] = self.allGroupEnds[-1]
    self.h5FileW.attrs["nGroups"] = len(self.allGroupPaths)

  def __createObjFromFile(self):
    """
      Function to create the list "self.allGroupPaths" and the dictionary "self.allGroupEnds"
      from a database that already exists. It uses the h5py method "visititems" in conjunction
      with the private method "self.__isGroup"
      @ In, None
      @ Out, None
    """
    self.allGroupPaths = []
    self.allGroupEnds  = []
    if len(self.h5FileW) == 0:
      # the database is empty. An error must be raised
      self.raiseAnError(IOError, 'The database '+str(self.name) + ' is empty but "readMode" is "read"!')
    if not self.fileOpen:
      self.h5FileW = self.openDatabaseW(self.filenameAndPath,'a')
    if 'allGroupPaths' in self.h5FileW and 'allGroupEnds' in self.h5FileW:
      nGroups = self.h5FileW.attrs.get("nGroups",None)
      self.allGroupPaths = utils.toBytesIterative(self.h5FileW["allGroupPaths"][:nGroups].tolist())
      self.allGroupEnds = self.h5FileW["allGroupEnds"][:nGroups].tolist()
    else:
      self.h5FileW.visititems(self.__isGroup)
      self.__createFileLevelInfoDatasets()
    self.h5FileW.attrs["nGroups"] = len(self.allGroupPaths)
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
      self.allGroupPaths.append(utils.toBytes(name))
      try:
        self.allGroupEnds.append(obj.attrs["endGroup"])
      except KeyError:
        self.allGroupEnds.append(True)
      if "rootname" in obj.attrs:
        self.parentGroupName = name
        self.raiseAWarning('not found attribute endGroup in group ' + name + '.Set True.')
    return

  def addExpectedMeta(self, keys, params={}):
    """
      Store expected metadata
      @ In, keys, set(), the metadata list
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    self.h5FileW.attrs['expectedMetadata'] = _dumps(list(keys))

  def provideExpectedMetaKeys(self):
    """
      Provides the registered list of metadata keys for this entity.
      @ In, None
      @ Out, meta, tuple, (set(str),dict), expected keys (empty if none) and dictionary of expected keys corresponding to their indexes
        i.e. {keys, [indexes]}
    """
    meta = set()
    gotMeta = self.h5FileW.attrs.get('expectedMetadata',None)
    if gotMeta is not None:
      meta = set(_loads(gotMeta))
    # FIXME, I'm not sure how to enable the HDF5 to store the time-dependent metadata,
    # or how to store the time dependent metadata in the HDF5, currently only empty dict of
    # indexes information is returned
    return meta, {}

  def addGroup(self, rlz):
    """
      Function to add a group into the database
      @ In, groupName, string, group name
      @ In, attributes, dict, dictionary of attributes that must be added as metadata
      @ In, source, File object, data source (for example, csv file)
      @ Out, None
    """
    parentID  = rlz.get("RAVEN_parentID", [None])[0]
    prefix    = rlz.get("prefix")

    groupName = str(prefix if mathUtils.isSingleValued(prefix) else prefix[0])
    if parentID:
      #If Hierarchical structure, firstly add the root group
      if not self.firstRootGroup or parentID == "None":
        self.__addGroupRootLevel(groupName,rlz)
        self.firstRootGroup = True
        self.type = 'DET'
      else:
        # Add sub group in the Hierarchical structure
        self.__addSubGroup(groupName,rlz)
    else:
      # Parallel structure (always root level)
      self.__addGroupRootLevel(groupName,rlz)
      self.firstRootGroup = True
      self.type = 'MC'
    self.__updateFileLevelInfoDatasets()
    self.h5FileW.flush()

  def addGroupInit(self, groupName, attributes=None):
    """
      Function to add an empty group to the database
      This function is generally used when the user provides a rootname in the input.
      It uses the groupName + it appends the date and time.
      @ In, groupName, string, group name
      @ In, attributes, dict, optional, dictionary of attributes that must be added as metadata (None by default)
      @ Out, None
    """
    attribs = {} if attributes is None else attributes
    groupNameInit = groupName+"_"+datetime.now().strftime("%m-%d-%Y-%H-%S")
    for index in range(len(self.allGroupPaths)):
      comparisonName = utils.toString(self.allGroupPaths[index])
      splittedPath=comparisonName.split('/')
      if len(splittedPath) > 0:
        if groupNameInit in splittedPath[0]:
          alphabetCounter, movingCounter = 0, 0
          asciiAlphabet   = list(string.ascii_uppercase)
          prefixLetter          = ''
          while True:
            testGroup = groupNameInit +"_"+prefixLetter+asciiAlphabet[alphabetCounter]
            if testGroup not in self.allGroupPaths:
              groupNameInit = utils.toString(testGroup)
              break
            alphabetCounter+=1
            if alphabetCounter >= len(asciiAlphabet):
              # prefix = asciiAlphabet[movingCounter]
              alphabetCounter = 0
              movingCounter  += 1
          break
    self.parentGroupName = "/" + groupNameInit
    # if the group exists, return it, otherwise create it
    grp = self.h5FileW.require_group(groupNameInit)
    # Add metadata
    grp.attrs.update(attribs)
    grp.attrs['rootname'  ] = True
    grp.attrs['endGroup'  ] = False
    grp.attrs[b'groupName'] = groupNameInit
    self.allGroupPaths.append(utils.toBytes("/" + groupNameInit))
    self.allGroupEnds.append(False)
    self.__updateFileLevelInfoDatasets()
    self.h5FileW.flush()

  def __populateGroup(self, group, name,  rlz):
    """
      This method is a common method between the __addGroupRootLevel and __addSubGroup
      It is used to populate the group with the info in the rlz
      @ In, group, h5py.Group, the group instance
      @ In, name, str, the group name (no path)
      @ In, rlz, dict, dictionary with the data and metadata to add
      @ Out, None
    """
    # vectorize method
    _vdumps = np.vectorize(_dumps)
    # create local dump method (no void)
    _vectDumps = lambda x: _vdumps(x,False)

    group.attrs[b'hasScalar'] = False
    group.attrs[b'hasOther'   ] = False
    if self.variables is not None:
      # check if all variables are contained in the rlz dictionary
      if not set(self.variables).issubset(rlz.keys()):
        self.raiseAnError(IOError, "Not all the requested variables have been passed in the realization. Missing are: "+
                          ",".join(list(set(self.variables).symmetric_difference(set(rlz.keys())))))
    # get the data floats or arrays
    if self.variables is None:
      dataScalar = dict( (key, np.atleast_1d(value)) for (key, value) in rlz.items()
                         if _checkTypeHDF5(value, False) )
    else:
      dataScalar = dict( (key, np.atleast_1d(value)) for (key, value) in rlz.items()
                         if _checkTypeHDF5(value, False) and key in self.variables)
    # get other dtype data (strings and objects)
    dataOther    = dict( (key, np.atleast_1d(_vectDumps(value))) for (key, value) in rlz.items() if _checkTypeHDF5(value, True) )
    # get size of each data variable (float)
    varKeysScalar = list(dataScalar.keys())
    if len(varKeysScalar) > 0:
      varShapeScalar = [dataScalar[key].shape for key in varKeysScalar]
      # get data names
      group.attrs[b'data_namesScalar'] = _dumps(varKeysScalar)
      # get data shapes
      group.attrs[b'data_shapesScalar'] = _dumps(varShapeScalar)
      # get data shapes
      end   = np.cumsum(varShapeScalar)
      begin = np.concatenate(([0],end[0:-1]))
      group.attrs[b'data_begin_endScalar'] = _dumps((begin.tolist(),end.tolist()))
      # get data names
      group.create_dataset(name + "_dataScalar", dtype="float", data=(np.concatenate( list(dataScalar.values())).ravel()))
      group.attrs[b'hasScalar'] = True
    # get size of each data variable (other type)
    varKeysOther = list(dataOther.keys())
    if len(varKeysOther) > 0:
      varShapeOther = [dataOther[key].shape for key in varKeysOther]
      # get data names
      group.attrs[b'data_namesOther'] = _dumps(varKeysOther)
      # get data shapes
      group.attrs[b'data_shapesOther'] = _dumps(varShapeOther)
      # get data shapes
      end   = np.cumsum(varShapeOther)
      begin = np.concatenate(([0],end[0:-1]))
      group.attrs[b'data_begin_endOther'] = _dumps((begin.tolist(),end.tolist()))
      # construct single data array
      vals = np.concatenate( list(dataOther.values())).ravel()
      # create dataset
      group.create_dataset(name + '_dataOther', dtype=vals.dtype,  data=vals)
      group.attrs[b'hasOther'] = True
    # add some info
    group.attrs[b'groupName'     ] = name
    group.attrs[b'endGroup'      ] = True
    group.attrs[b'RAVEN_parentID'] = group.parent.name
    group.attrs[b'nVarsScalar' ] = len(varKeysScalar)
    group.attrs[b'nVarsOther'    ] = len(varKeysOther)

  def __addGroupRootLevel(self,groupName,rlz):
    """
      Function to add a group into the database (root level)
      @ In, groupName, string, group name
      @ In, rlz, dict, dictionary with the data and metadata to add
      @ Out, None
    """
    # Check in the "self.allGroupPaths" list if a group is already present...
    # If so, error (Deleting already present information is not desiderable)
    while self.__returnGroupPath(groupName) != '-$':
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
    grp = parentGroupObj.create_group(groupName)
    self.__populateGroup(grp, groupName, rlz)
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
    parentID = rlz.get("RAVEN_parentID")[0]
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

    parentGroupObj.attrs[b'endGroup' ] = False
    # create the sub group
    self.raiseAMessage('Adding group named "' + groupName + '" in Database "'+ self.name +'"')
    # create and populate the group
    grp = parentGroupObj.create_group(groupName)
    self.__populateGroup(grp, groupName, rlz)
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
      self.allGroupPaths.append(utils.toBytes(parentName) + b"/" + utils.toBytes(groupName))
      self.allGroupEnds.append(True)
    else:
      self.allGroupPaths.append(b"/" + utils.toBytes(groupName))
      self.allGroupEnds.append(True)

  def retrieveAllHistoryNames(self,rootName=None):
    """
      Function to create a list of all the HistorySet names present in an existing database
      @ In,  rootName, string, optional, It's the root name, if present, only the history names that have this root are going to be returned
      @ Out, workingList, list, List of the HistorySet names
    """
    if rootName:
      rname = utils.toString(rootName)
    if not self.fileOpen:
      self.__createObjFromFile() # Create the "self.allGroupPaths" list from the existing database
    if not rootName:
      workingList = [utils.toString(k).split('/')[-1] for k, v in zip(self.allGroupPaths,self.allGroupEnds) if v ]
    else:
      workingList = [utils.toString(k).split('/')[-1] for k, v in zip(self.allGroupPaths,self.allGroupEnds) if v and utils.toString(k).endswith(rname)]

    return workingList

  def __getListOfParentGroups(self, grp, backGroups = None):
    """
      Method to get the list of groups from the deepest to the root, given a certain group
      @ In, grp, h5py.Group, istance of the starting group
      @ InOut, backGroups, list, list of group instances (from the deepest to the root)
    """
    backGroups = [] if backGroups is None else backGroups
    if grp.parent and grp.parent != grp:
      parentGroup = grp.parent
      if not parentGroup.attrs.get("rootname",False):
        backGroups.append(parentGroup)
        self.__getListOfParentGroups(parentGroup, backGroups)
    backGroups = list(set(backGroups))
    return backGroups

  def __getNewDataFromGroup(self, group, name):
    """
      Get the data from the group
      @ In, group, h5py.Group, the group from which the data needs to be got
      @ In, name, str, the group name
      @ Out, newData, dict, the dictionary with the data
    """
    newData = {}
    hasScalar = group.attrs['hasScalar']
    hasOther    = group.attrs['hasOther']
    if hasScalar:
      dataSetScalar = group[name + "_dataScalar"]
      # Get some variables of interest
      varShapeScalar   = _loads(group.attrs[b'data_shapesScalar'])
      varKeysScalar    = _loads(group.attrs[b'data_namesScalar'])
      begin, end         = _loads(group.attrs[b'data_begin_endScalar'])
      # Reconstruct the dataset
      newData = {key : np.reshape(dataSetScalar[begin[cnt]:end[cnt]], varShapeScalar[cnt]) for cnt,key in enumerate(varKeysScalar)}
    if hasOther:
      unvect = np.vectorize(_loads)
      # get the "other" data
      datasetOther = group[name + "_dataOther"]
      # Get some variables of interest
      varShapeOther   = _loads(group.attrs[b'data_shapesOther'])
      varKeysOther    = _loads(group.attrs[b'data_namesOther'])
      begin, end       = _loads(group.attrs[b'data_begin_endOther'])
      # Reconstruct the dataset
      newData.update({key : unvect(np.reshape(datasetOther[begin[cnt]:end[cnt]], varShapeOther[cnt])) for cnt,key in enumerate(varKeysOther)})
    return newData

  def _getRealizationByName(self, name, options=None):
    """
      Function to retrieve the history whose end group name is "name"
      @ In, name, string, realization name => It must correspond to a group name (string)
      @ In, options, dict, dictionary of options (now, just "recunstruct" flag)
      @ In, attributes, dict, optional, dictionary of attributes (options)
      @ Out, (newData,attrs), tuple, tuple where position 0 = dict containing the realization, 1 = dictionary of some attributes
    """
    if options is None:
      options = {}
    reconstruct = options.get("reconstruct", True)
    path  = ''
    found = False
    attrs = {}
    # Check if the h5 file is already open, if not, open it
    # and create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen:
      self.__createObjFromFile()
    # Find the endGroup that coresponds to the given name
    path = self.__returnGroupPath(name)
    found = path != '-$'

    if found:
      # Grep only history from group "name"
      group = self.h5FileW.require_group(path)
      # Retrieve dataset
      newData = self.__getNewDataFromGroup(group, name)
      # Add the attributes
      attrs = {'nVars':len(newData.keys()),'varKeys':newData.keys()}
      # check the reconstruct flag
      if reconstruct:
        # get list of back groups
        listGroups = self.__getListOfParentGroups(group)
        listGroups.reverse()
        for grp in listGroups:
          # the root groups get skipped
          if grp.name not in ["/",self.parentGroupName]:
            data = self.__getNewDataFromGroup(grp, grp.attrs[b'groupName'])
            if len(data.keys()) != len(newData.keys()):
              self.raiseAnError(IOError,'Group named "' + grp.attrs[b'groupName'] + '" has an inconsistent number of variables in database "'+self.name+'"!')
            newData = {key : np.concatenate((newData[key],data[key])) for key in newData.keys()}
    else:
      self.raiseAnError(IOError,'Group named ' + name + ' not found in database "'+self.name+'"!')

    return(newData,attrs)

  def closeDatabaseW(self):
    """
      Function to close the database
      @ In,  None
      @ Out, None
    """
    self.h5FileW.close()
    self.fileOpen = False
    return

  def openDatabaseW(self,filename,mode='w'):
    """
      Function to open the database
      @ In, filename, string, name of the file (string)
      @ In, mode, string, open mode (default "w=write")
      @ Out, fh5, hdf5 object, instance of hdf5
    """
    fh5 = h5.File(filename,mode)
    self.fileOpen = True
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
        if utils.toString(s).endswith("/"+parentName.strip()):
          parentGroupName = s
          break
    else:
      parentGroupName = '/'

    return parentGroupName
