"""
Created on Mar 25, 2013

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import h5py  as h5
import numpy as np
import os
import copy
import json
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

"""
  *************************
  *  HDF5 DATABASE CLASS  *
  *************************
"""
class hdf5Database(MessageHandler.MessageUser):
  """
  class to create a h5py (hdf5) database
  """
  def __init__(self,name, databaseDir, messageHandler,filename=None):
    # database name (i.e. arbitrary name).
    # It is the database name that has been found in the xml input
    self.name       = name
    # Database type :
    # -> The structure type is "inferred" by the first group is going to be added
    # * MC  = MonteCarlo => Storing by a Parallel structure
    # * DET = Dynamic Event Tree => Storing by a Hierarchical structure
    self.type       = None
    # specialize printTag (THIS IS THE CORRECT WAY TO DO THIS)
    self.printTag = 'DATABASE HDF5'
    self.messageHandler = messageHandler
    # .H5 file name (to be created or read)
    if filename:
      # File name on disk (file exists => fileExist flag is True)
      self.onDiskFile = filename
      self.fileExist  = True
    else:
      # File name on disk (file does not exist => it will create => fileExist flag is False)
      self.onDiskFile = name + ".h5"
      self.fileExist  = False
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
      # self.h5_file_w is the HDF5 object. Open the database in "update" mode
      # check if it exists
      if not os.path.exists(self.filenameAndPath): self.raiseAnError(IOError,'database file has not been found, searched Path is: ' + self.filenameAndPath )
      # Open file
      self.h5_file_w = self.openDatabaseW(self.filenameAndPath,'r+')
      # Call the private method __createObjFromFile, that constructs the list of the paths "self.allGroupPaths"
      # and the dictionary "self.allGroupEnds" based on the database that already exists
      self.parent_group_name = b'/'
      self.__createObjFromFile()
      # "self.firstRootGroup", true if the root group is present (or added), false otherwise
      self.firstRootGroup = True
    else:
      # self.h5_file_w is the HDF5 object. Open the database in "write only" mode
      self.h5_file_w = self.openDatabaseW(self.filenameAndPath,'w')
      # Add the root as first group
      self.allGroupPaths.append("/")
      # The root group is not an end group
      self.allGroupEnds["/"] = False
      # The first root group has not been added yet
      self.firstRootGroup = False
      # The root name is / . it can be changed if addGroupInit is called
      self.parent_group_name = b'/'

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
    if not self.fileOpen: self.h5_file_w = self.openDatabaseW(self.filenameAndPath,'a')
    self.h5_file_w.visititems(self.__isGroup)
    self.raiseAMessage('TOTAL NUMBER OF GROUPS = ' + str(len(self.allGroupPaths)))

  def __isGroup(self,name,obj):
    """
    Function to check if an object name is of type "group". If it is, the function stores
    its name into the "self.allGroupPaths" list and update the dictionary "self.allGroupEnds"
    @ In, name : object name
    @ In, obj  : the object itself
    """
    if isinstance(obj,h5.Group):
      self.allGroupPaths.append(name)
      self.raiseAMessage('Accessing group named ' +name)
      if "EndGroup" in obj.attrs:
        self.allGroupEnds[name]  = obj.attrs["EndGroup"]
      else:
        self.raiseAWarning('not found attribute EndGroup in group ' + name + '.Set True.')
        self.allGroupEnds[name]  = True
      if "rootname" in obj.attrs: self.parent_group_name = name
    return

  def addGroup(self,gname,attributes,source,upGroup=False):
    """
    Function to add a group into the database
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    """

    if source['type'] == 'DataObjects':
      self.addGroupDataObjects(gname,attributes,source)
      self.h5_file_w.flush()
      return
    parent_id = None
    if 'metadata' in attributes.keys():
      if 'parent_id' in attributes['metadata'].keys(): parent_id = attributes['metadata']['parent_id']
      if 'parent_id' in attributes.keys(): parent_id = attributes['parent_id']
    else:
      if 'parent_id' in attributes.keys(): parent_id = attributes['parent_id']
    if parent_id:
      #If Hierarchical structure, firstly add the root group
      if not self.firstRootGroup or parent_id == 'root':
        self.__addGroupRootLevel(gname,attributes,source,upGroup)
        self.firstRootGroup = True
        self.type = 'DET'
      else:
        # Add sub group in the Hierarchical structure
        self.__addSubGroup(gname,attributes,source)
    else:
      # Parallel structure (always root level)
      self.__addGroupRootLevel(gname,attributes,source,upGroup)
      self.firstRootGroup = True
      self.type = 'MC'
    self.h5_file_w.flush()
    return

  def addGroupInit(self,gname,attributes=None,upGroup=False):
    """
    Function to add an empty group to the database
    This function is generally used when the user provides a rootname in the input
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, gname      : group name
    @ Out, None
    """
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        splittedPath=comparisonName.split('/')
        if len(splittedPath) > 0:
          if gname == splittedPath[0]: self.raiseAnError(IOError,"Group named " + gname + " already present as root group in database " + self.name + ". new group " + gname + " is equal to old group " + splittedPath[0])
    self.parent_group_name = "/" + gname
    # Create the group
    grp = self.h5_file_w.create_group(gname)
    # Add metadata
    if attributes:
      for key in attributes.keys(): grp.attrs[key] = attributes[key]
    grp.attrs['rootname'] = True
    grp.attrs['EndGroup'] = False
    self.allGroupPaths.append("/" + gname)
    self.allGroupEnds["/" + gname] = False
    self.h5_file_w.flush()

  def __addGroupRootLevel(self,gname,attributes,source,upGroup=False):
    """
    Function to add a group into the database (root level)
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    """
    # Check in the "self.allGroupPaths" list if a group is already present...
    # If so, error (Deleting already present information is not desiderable)
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        splittedPath=comparisonName.split('/')
        for splgroup in splittedPath:
          if gname == splgroup and splittedPath[0] == self.parent_group_name: self.raiseAnError(IOError,"Group named " + gname + " already present in database " + self.name + ". new group " + gname + " is equal to old group " + comparisonName)

    if source['type'] == 'csv':
      # Source in CSV format
      f = open(source['name'],'rb')
      # Retrieve the headers of the CSV file
      firstRow = f.readline().strip(b"\r\n")
      #firstRow = f.readline().translate(None,"\r\n")
      headers = firstRow.split(b",")
      # Load the csv into a numpy array(n time steps, n parameters)
      data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
      # First parent group is the root name
      parent_name = self.parent_group_name.replace('/', '')
      # Create the group
      if parent_name != '/':
        parent_group_name = self.__returnParentGroupPath(parent_name)
        # Retrieve the parent group from the HDF5 database
        if parent_group_name in self.h5_file_w: rootgrp = self.h5_file_w.require_group(parent_group_name)
        else: self.raiseAnError(ValueError,'NOT FOUND group named ' + parent_group_name)
        if upGroup:
          grp = rootgrp.require_group(gname)
          del grp[gname+"_data"]
        else: grp = rootgrp.create_group(gname)
      else:
        if upGroup: grp = self.h5_file_w.require_group(gname)
        else:       grp = self.h5_file_w.create_group(gname)
      self.raiseAMessage('Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
      # Create dataset in this newly added group
      grp.create_dataset(gname+"_data", dtype="float", data=data)
      # Add metadata
      grp.attrs["output_space_headers"   ] = headers
      grp.attrs["n_params"               ] = data[0,:].size
      grp.attrs["parent_id"              ] = "root"
      grp.attrs["start_time"             ] = data[0,0]
      grp.attrs["end_time"               ] = data[data[:,0].size-1,0]
      grp.attrs["n_ts"                   ] = data[:,0].size
      grp.attrs["EndGroup"               ] = True
      grp.attrs["source_type"            ] = source['type']
      if source['type'] == 'csv': grp.attrs["source_file"] = source['name']
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
              grp.attrs[b'input_space_headers'] = inpHeaders
              grp.attrs[b'input_space_values' ] = inpValues
        objectToConvert = utils.convertNumpyToLists(attributes[attr])
        converted = json.dumps(objectToConvert)
        if converted and attr != 'name': grp.attrs[utils.toBytes(attr)]=converted
        #decoded = json.loads(grp.attrs[utils.toBytes(attr)])
      if "input_file" in attributes.keys(): grp.attrs[utils.toString("input_file")] = utils.toString(" ".join(attributes["input_file"])) if type(attributes["input_file"]) == type([]) else utils.toString(attributes["input_file"])
    else: self.raiseAnError(ValueError,source['type'] + " unknown!")
    # Add the group name into the list "self.allGroupPaths" and
    # set the relative bool flag into the dictionary "self.allGroupEnds"
    if parent_group_name != "/":
      self.allGroupPaths.append(parent_group_name + "/" + gname)
      self.allGroupEnds[parent_group_name + "/" + gname] = True
    else:
      self.allGroupPaths.append("/" + gname)
      self.allGroupEnds["/" + gname] = True

  def addGroupDataObjects(self,gname,attributes,source,upGroup=False):
    """
    Function to add a data (class Datas) or Dictionary into the DataBase
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, a TimePointSet)
    @ In, upGroup    : update Group????
    @ Out, None
    """
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        splittedPath=comparisonName.split('/')
        for splgroup in splittedPath:
          if gname == splgroup and splittedPath[0] == self.parent_group_name: self.raiseAnError(IOError,"Group named " + gname + " already present in database " + self.name + ". new group " + gname + " is equal to old group " + comparisonName)
    parent_name = self.parent_group_name.replace('/', '')
    # Create the group
    if parent_name != '/':
      parent_group_name = self.__returnParentGroupPath(parent_name)
      # Retrieve the parent group from the HDF5 database
      if parent_group_name in self.h5_file_w: parentgroup_obj = self.h5_file_w.require_group(parent_group_name)
      else: self.raiseAnError(ValueError,'NOT FOUND group named ' + parentgroup_obj)
    else: parentgroup_obj = self.h5_file_w

    if type(source['name']) == dict:
      # create the group
      if upGroup:
        groups = parentgroup_obj.require_group(gname)
        del groups[gname+"_data"]
      else: groups = parentgroup_obj.create_group(gname)
      groups.attrs[b'main_class' ] = b'PythonType'
      groups.attrs[b'source_type'] = b'Dictionary'
      # I keep this structure here because I want to maintain the possibility to add a whatever dictionary even if not prepared and divided into output and input sub-sets. A.A.
      if set(['input_space_params']).issubset(set(source['name'].keys())):
        groups.attrs[b'input_space_headers' ] = list(utils.toBytesIterative(source['name']['input_space_params'].keys()))
        groups.attrs[b'input_space_values'  ] = list(utils.toBytesIterative(source['name']['input_space_params'].values()))
      if set(['output_space_params']).issubset(set(source['name'].keys())): outDict = source['name']['output_space_params']
      else: outDict = dict((key,value) for (key,value) in source['name'].iteritems() if key not in ['input_space_params'])
      out_headers = list(utils.toBytesIterative(outDict.keys()))
      out_values  = list(utils.toBytesIterative(outDict.values()))
      groups.attrs[b'n_params'   ] = len(out_headers)
      groups.attrs[b'output_space_headers'] = out_headers
      groups.attrs[b'EndGroup'   ] = True
      groups.attrs[b'parent_id'  ] = parent_name
      maxsize = 0
      for value in out_values:
        if type(value) == np.ndarray:
          if maxsize < value.size : actualone = value.size
        elif type(value) in [int,float,bool,np.float64,np.float32,np.float16,np.int64,np.int32,np.int16,np.int8,np.bool8]: actualone = 1
        else: self.raiseAnError(IOError,'The type of the dictionary parameters must be within float,bool,int,numpy.ndarray')
        if maxsize < actualone: maxsize = actualone
      groups.attrs[b'n_ts'  ] = maxsize
      dataout = np.zeros((maxsize,len(out_headers)))
      for index in range(len(out_headers)):
        if type(out_values[index]) == np.ndarray:  dataout[0:out_values[index].size,index] =  out_values[index][:]
        else: dataout[0,index] = out_values[index]
      # create the data set
      groups.create_dataset(gname + "_data", dtype="float", data=dataout)
      # add metadata if present
      for attr in attributes.keys():
        objectToConvert = utils.convertNumpyToLists(attributes[attr])
        converted = json.dumps(objectToConvert)
        if converted and attr != 'name': groups.attrs[utils.toBytes(attr)]=converted
      if parent_group_name != "/":
        self.allGroupPaths.append(parent_group_name + "/" + gname)
        self.allGroupEnds[parent_group_name + "/" + gname] = True
      else:
        self.allGroupPaths.append("/" + gname)
        self.allGroupEnds["/" + gname] = True
    else:
      # Data(structure)
      # Retrieve the headers from the data (inputs and outputs)
      headers_in  = list(source['name'].getInpParametersValues().keys())
      headers_out = list(source['name'].getOutParametersValues().keys())
      # for a "histories" type we create a number of groups = number of histories (compatibility with loading structure)
      data_in  = list(source['name'].getInpParametersValues().values())
      data_out = list(source['name'].getOutParametersValues().values())
      metadata = source['name'].getAllMetadata()
      if source['name'].type in ['Histories','TimePointSet']:
        groups = []
        if 'Histories' in source['name'].type: nruns = len(data_in)
        else:                                  nruns = data_in[0].size
        for run in range(nruns):
          if upGroup:
            groups.append(parentgroup_obj.require_group(gname + b'|' +str(run)))
            if (gname + "_data") in groups[run] : del groups[run][gname+"_data"]
          else:
            groups.append(parentgroup_obj.create_group(gname + '|' +str(run)))

          groups[run].attrs[b'source_type'] = utils.toBytes(source['name'].type)
          groups[run].attrs[b'main_class' ] = b'DataObjects'
          groups[run].attrs[b'EndGroup'   ] = True
          groups[run].attrs[b'parent_id'  ] = parent_name
          if source['name'].type == 'Histories':
            groups[run].attrs[b'input_space_headers' ] = [utils.toBytes(list(data_in[run].keys())[i])  for i in range(len(data_in[run].keys()))]
            groups[run].attrs[b'output_space_headers'] = [utils.toBytes(list(data_out[run].keys())[i])  for i in range(len(data_out[run].keys()))]
            groups[run].attrs[b'input_space_values'  ] = list(data_in[run].values())
            groups[run].attrs[b'n_params'            ] = len(data_out[run].keys())
            #collect the outputs
            dataout = np.zeros((next(iter(data_out[run].values())).size,len(data_out[run].values())))
            for param in range(len(data_out[run].values())): dataout[:,param] = list(data_out[run].values())[param][:]
            groups[run].create_dataset(gname +'|' +str(run)+"_data" , dtype="float", data=dataout)
            groups[run].attrs[b'n_ts'                ] = next(iter(data_out[run].values())).size
          else:
            groups[run].attrs[b'input_space_headers' ] = [utils.toBytes(headers_in[i])  for i in range(len(headers_in))]
            groups[run].attrs[b'output_space_headers'] = [utils.toBytes(headers_out[i])  for i in range(len(headers_out))]
            groups[run].attrs[b'input_space_values'  ] = [np.atleast_1d(np.array(data_in[x][run])) for x in range(len(data_in))]
            groups[run].attrs[b'n_params'            ] = len(headers_out)
            groups[run].attrs[b'n_ts'                ] = 1
            #collect the outputs
            dataout = np.zeros((1,len(data_out)))
            for param in range(len(data_out)): dataout[0,param] = data_out[param][run]
            groups[run].create_dataset(gname +'|' +str(run)+"_data", dtype="float", data=dataout)
          # add metadata if present
          for attr in attributes.keys():
            objectToConvert = utils.convertNumpyToLists(attributes[attr])
            converted = json.dumps(objectToConvert)
            if converted and attr != 'name': groups[run].attrs[utils.toBytes(attr)]=converted
          for attr in metadata.keys():
            if len(metadata[attr]) == nruns: objectToConvert = utils.convertNumpyToLists(metadata[attr][run])
            else                           : objectToConvert = utils.convertNumpyToLists(metadata[attr])
            converted = json.dumps(objectToConvert)
            if converted and attr != 'name': groups[run].attrs[utils.toBytes(attr)]=converted

          if parent_group_name != "/":
            self.allGroupPaths.append(parent_group_name + "/" + gname + '|' +str(run))
            self.allGroupEnds[parent_group_name + "/" + gname + '|' +str(run)] = True
          else:
            self.allGroupPaths.append("/" + gname + '|' +str(run))
            self.allGroupEnds["/" + gname + '|' +str(run)] = True
      elif source['name'].type in ['TimePoint','History']:
        if upGroup:
          groups = parentgroup_obj.require_group(gname)
          del groups[gname+"_data"]
        else: groups = parentgroup_obj.create_group(gname)
        groups.attrs[b'main_class'          ] = b'DataObjects'
        groups.attrs[b'source_type'         ] = utils.toBytes(source['name'].type)
        groups.attrs[b'n_params'            ] = len(headers_out)
        groups.attrs[b'input_space_headers' ] = [utils.toBytes(headers_in[i])  for i in range(len(headers_in))]
        groups.attrs[b'output_space_headers'] = [utils.toBytes(headers_out[i])  for i in range(len(headers_out))]
        groups.attrs[b'input_space_values'  ] = [np.array(data_in[i])  for i in range(len(data_in))]
        groups.attrs[b'source_type'         ] = utils.toBytes(source['name'].type)
        groups.attrs[b'EndGroup'            ] = True
        groups.attrs[b'parent_id'           ] = parent_name
        dataout = np.zeros((data_out[0].size,len(data_out)))
        groups.attrs[b'n_ts'  ] = data_out[0].size
        for run in range(len(data_out)): dataout[:,int(run)] = data_out[run][:]
        groups.create_dataset(gname + "_data", dtype="float", data=dataout)
        # add metadata if present
        for attr in attributes.keys():
          objectToConvert = utils.convertNumpyToLists(attributes[attr])
          converted = json.dumps(objectToConvert)
          if converted and attr != 'name': groups.attrs[utils.toBytes(attr)]=converted
        for attr in metadata.keys():
          objectToConvert = utils.convertNumpyToLists(metadata[attr])
          converted = json.dumps(objectToConvert)
          if converted and attr != 'name': groups.attrs[utils.toBytes(attr)]=converted

        if parent_group_name != "/":
          self.allGroupPaths.append(parent_group_name + "/" + gname)
          self.allGroupEnds[parent_group_name + "/" + gname] = True
        else:
          self.allGroupPaths.append("/" + gname)
          self.allGroupEnds["/" + gname] = True
      else: self.raiseAnError(IOError,'The function addGroupDataObjects accepts Data(s) or dictionaries as inputs only!!!!!')

  def __addSubGroup(self,gname,attributes,source):
    """
    Function to add a group into the database (Hierarchical)
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    """
    for index in xrange(len(self.allGroupPaths)):
      comparisonName = self.allGroupPaths[index]
      splittedPath=comparisonName.split('/')
      for splgroup in splittedPath:
        if gname == splgroup and splittedPath[0] == self.parent_group_name: self.raiseAnError(IOError,"Group named " + gname + " already present in database " + self.name + ". new group " + gname + " is equal to old group " + comparisonName)
    if source['type'] == 'csv':
      # Source in CSV format
      f = open(source['name'],'rb')
      # Retrieve the headers of the CSV file
      headers = f.readline().split(b",")
      # Load the csv into a numpy array(n time steps, n parameters)
      data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
      # Check if the parent attribute is not null # In this case append a subgroup to the parent group
      # Otherwise => it's the main group
      parent_id = None
      if 'metadata' in attributes.keys():
        if 'parent_id' in attributes['metadata'].keys(): parent_id = attributes['metadata']['parent_id']
        if 'parent_id' in attributes.keys(): parent_id = attributes['parent_id']
      else:
        if 'parent_id' in attributes.keys(): parent_id = attributes['parent_id']

      if parent_id: parent_name = parent_id
      else: self.raiseAnError(IOError,'NOT FOUND attribute <parent_id> into <attributes> dictionary')
      # Find parent group path
      if parent_name != '/':
        parent_group_name = self.__returnParentGroupPath(parent_name)
      else: parent_group_name = parent_name
      # Retrieve the parent group from the HDF5 database
      if parent_group_name in self.h5_file_w: grp = self.h5_file_w.require_group(parent_group_name)
      else:
        self.raiseAnError(ValueError,'NOT FOUND group named ' + parent_group_name)
      # The parent group is not the endgroup for this branch
      self.allGroupEnds[parent_group_name] = False
      grp.attrs["EndGroup"]   = False
      self.raiseAMessage('Adding group named "' + gname + '" in Database "'+ self.name +'"')
      # Create the sub-group
      sgrp = grp.create_group(gname)
      # Create data set in this new group
      sgrp.create_dataset(gname+"_data", dtype="float", data=data)
      # Add the metadata
      sgrp.attrs["output_space_headers"   ] = headers
      sgrp.attrs["n_params"  ] = data[0,:].size
      sgrp.attrs["parent"    ] = "root"
      sgrp.attrs["start_time"] = data[0,0]
      sgrp.attrs["end_time"  ] = data[data[:,0].size-1,0]
      sgrp.attrs["n_ts"      ] = data[:,0].size
      sgrp.attrs["EndGroup"  ] = True
      sgrp.attrs["source_type"] = source['type']
      if source['type'] == 'csv': sgrp.attrs["source_file"] = source['name']
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
              sgrp.attrs[b'input_space_headers'] = inpHeaders
              sgrp.attrs[b'input_space_values' ] = inpValues
        objectToConvert = utils.convertNumpyToLists(attributes[attr])
        #if type(attributes[attr]) in [np.ndarray]: objectToConvert = attributes[attr].tolist()
        #else:                                      objectToConvert = attributes[attr]
        converted = json.dumps(objectToConvert)
        if converted and attr != 'name': sgrp.attrs[utils.toBytes(attr)]=converted
      if "input_file" in attributes: grp.attrs[utils.toString("input_file")] = utils.toString(" ".join(attributes["input_file"])) if type(attributes["input_file"]) == type([]) else utils.toString(attributes["input_file"])
    else: pass
    # The sub-group is the new ending group
    if parent_group_name != "/":
      self.allGroupPaths.append(parent_group_name + "/" + gname)
      self.allGroupEnds[parent_group_name + "/" + gname] = True
    else:
      self.allGroupPaths.append("/" + gname)
      self.allGroupEnds["/" + gname] = True
    return

  def computeBack(self,nameFrom,nameTo):
    """
    Function to compute the number of step back from a group to another
    @ In,  nameFrom : group name (from)
    @ In,  nameTo   : group name (to)
    @ Out, back     : number of step back (integer)
    """
    # "list_str_w", list in which the path for a particular group is stored (working variable)
    # "path", string in which the path of the "to" group is stored
    # "found", bolean variable ... I would say...self explanable :D
    list_str_w = []
    path       = ''
    found      = False
    # Find the path fo the "nameTo" group
    for i in xrange(len(self.allGroupPaths)):
      list_str_w = self.allGroupPaths[i].split("/")
      if list_str_w[len(list_str_w)-1] == nameTo:
        found = True
        path  = self.allGroupPaths[i]
        break
    if not found: self.raiseAnError(NameError,"Group named " + nameTo + " not found in the HDF5 database" + self.filenameAndPath)
    else: listGroups = path.split("/")  # Split the path in order to create a list of the groups in this history
    # Retrieve indeces of groups "nameFrom" and "nameTo" v
    fr = listGroups.index(nameFrom)
    to = listGroups.index(nameTo)
    # Compute steps back
    back = to - fr
    return back

  def retrieveAllHistoryPaths(self,rootName=None):
    """
    Function to create a list of all the histories' paths present in an existing database
    @ In,  rootName (optional), It's the root name, if present, only the groups that have this root are going to be returned
    @ Out, List of the histories' paths
    """
    allHistoryPaths = []
    # Create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen: self.__createObjFromFile()
    # Check database type
    if self.type == 'MC':
      # Parallel structure => "self.allGroupPaths" already contains the histories' paths
      if not rootName: allHistoryPaths = self.allGroupPaths
      else:
        for index in xrange(len(self.allGroupPaths)):
          if rootName in self.allGroupPaths[index].split('/')[1] : allHistoryPaths.append(self.allGroupPaths[index])
    else:
      # Tree structure => construct the histories' paths
      for index in xrange(len(self.allGroupPaths)):
        if self.allGroupEnds[self.allGroupPaths[index]]:
          if rootName and not (rootName in self.allGroupPaths[index].split('/')[1]): continue
          allHistoryPaths.append(self.allGroupPaths[index])
    return allHistoryPaths

  def retrieveAllHistoryNames(self,rootName=None):
    """
    Function to create a list of all the histories' names present in an existing database
    @ In,  rootName (optional), It's the root name, if present, only the history names that have this root are going to be returned
    @ Out, List of the histories' names
    """
    if not self.fileOpen: self.__createObjFromFile() # Create the "self.allGroupPaths" list from the existing database
    workingList = []
    for index in xrange(len(self.allGroupPaths)):
      if self.allGroupEnds[self.allGroupPaths[index]]:
        if rootName and not (rootName in self.allGroupPaths[index].split('/')[1]): continue
        workingList.append(self.allGroupPaths[index].split('/')[len(self.allGroupPaths[index].split('/'))-1])
    return workingList

  def retrieveHistory(self,name,filterHist=None,attributes = None):
    """
    Function to retrieve the history whose end group name is "name"
    @ In,  name       : history name => It must correspond to a group name (string)
    @ In,  filterHist : filter for history retrieving
                    ('whole' = whole history,
                     integer value = groups back from the group "name",
                     or None = retrieve only the group "name". Defaul is None)
    @ In, attributes : dictionary of attributes (options)
    @ Out, history: tuple where position 0 = 2D numpy array (history), 1 = dictionary (metadata)
    """
    list_str_w = []
    list_path  = []
    path       = ''
    found      = False
    result     = None
    attrs = {}
    if attributes:
      if 'inputTs' in attributes.keys() : inputTs  = attributes['inputTs' ]
      if 'operator' in attributes.keys(): operator = attributes['operator']
    else:
      inputTs  = None
      operator = None

    # Check if the h5 file is already open, if not, open it
    # and create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen: self.__createObjFromFile()
    # Find the endgroup that coresponds to the given name
    for i in xrange(len(self.allGroupPaths)):
      list_str_w = self.allGroupPaths[i].split("/")
      try: list_str_w.remove("")
      except: pass
      if list_str_w[len(list_str_w)-1] == name:
        found = True
        path  = self.allGroupPaths[i]
        list_path = list_str_w
        break
    if found:
      # Check the filter type
      if not filterHist or filterHist == 0:
        # Grep only History from group "name"
        grp = self.h5_file_w.require_group(path)
        # Retrieve dataset
        dataset = grp.require_dataset(name +'_data', (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float').value
        # Get numpy array
        result = dataset[:,:]
        # Get attributes (metadata)
        attrs = grp.attrs
        for attr in attrs.keys():
          try   : attrs[attr] = json.loads(attrs[attr])
          except: attrs[attr] = attrs[attr]
      elif  filterHist == 'whole':
        # Retrieve the whole history from group "name" to the root
        # Start constructing the merged numpy array
        where_list = []
        name_list  = []
        # back represents the number of groups back that need to be
        # included in the construction of the full history.
        if self.parent_group_name != '/': back = len(list_path)-2
        else: back = len(list_path)-1
        if back < 0: back = 0
        i=0
        #Question, should all the "" be removed, or just the first?
        try: list_path.remove("")
        except ValueError:  pass #Not found.
        # Find the paths for the completed history
        while (i <= back):
          path_w = ''
          for j in xrange(len(list_path) - i):
            if list_path[j] != "": path_w = path_w + "/" + list_path[j]
          if path_w != "":  where_list.append(path_w)
          mylist = where_list[i].split("/")
          name_list.append(mylist[len(mylist)-1])
          i = i + 1
        # get the relative groups' data
        gb_res ={}
        gb_attrs={}
        where_list.reverse()
        name_list.reverse()
        n_tot_ts = 0
        n_params = 0
        # Retrieve every single partial history that will be merged to create the whole history
        for i in xrange(len(where_list)):
          grp = self.h5_file_w.require_group(where_list[i])
          namel = name_list[i] +'_data'
          dataset = grp.require_dataset(namel, (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float').value
          if i == 0: n_params = int(grp.attrs['n_params'])
          if n_params != int(grp.attrs['n_params']): self.raiseAnError(TypeError,'Can not merge datasets with different number of parameters')
          # Get numpy array
          gb_res[i]   = dataset[:,:]
          gb_attrs[i] = copy.copy(grp.attrs   )
          n_tot_ts = n_tot_ts + int(grp.attrs['n_ts'])
        # Create the numpy array
        result = np.zeros((n_tot_ts,n_params))
        ts = 0
        # Retrieve the metadata
        for key in gb_res:
          arr = gb_res[key]
          result[ts:ts+arr[:,0].size,:] = arr
          ts = ts + arr[:,0].size
          # must be checked if overlapping of time (branching for example)
        try:    attrs["output_space_headers"]   = gb_attrs[0]["output_space_headers"].tolist()
        except: attrs["output_space_headers"]   = gb_attrs[0]["output_space_headers"]
        try:    attrs["input_space_headers"]    = gb_attrs[0]["input_space_headers"].tolist()
        except:
          try:    attrs["input_space_headers"]  = gb_attrs[0]["input_space_headers"]
          except: pass
        try:    attrs["input_space_values"]     = gb_attrs[0]["input_space_values"].tolist()
        except:
          try:    attrs["input_space_values"]   = gb_attrs[0]["input_space_values"]
          except: pass
        attrs["n_params"]        = gb_attrs[0]["n_params"]
        attrs["parent_id"]       = where_list[0]
        attrs["start_time"]      = result[0,0]
        attrs["end_time"]        = result[result[:,0].size-1,0]
        attrs["n_ts"]            = result[:,0].size
        attrs["source_type"]     = gb_attrs[0]["source_type"]
        attrs["input_file"]      = []
        attrs["source_file"]     = []
        for key in gb_res.keys():
          for attr in gb_attrs[key].keys():
            if attr not in ["output_space_headers","input_space_headers","input_space_values","n_params","parent_id","start_time","end_time","n_ts","source_type"]:
              if attr not in attrs.keys(): attrs[attr] = []
              try   : attrs[attr].append(json.loads(gb_attrs[key][attr]))
              except:
                if type(attrs[attr]) == list: attrs[attr].append(gb_attrs[key][attr])
          if attrs["source_type"] == 'csv' and 'source_file' in gb_attrs[key].keys(): attrs["source_file"].append(gb_attrs[key]["source_file"])
      else:
        # A number of groups' back have been inputted
        # Follow the same strategy used above (filterHist = whole)
        # but stop at back(th) group starting from group "name"
        if is_number(filterHist):
          back = int(filterHist) + 1
          if len(list_path) < back: self.raiseAnError(RuntimeError,'Number of branches back > number of actual branches in dataset for History ending with ' + name)
          if (back == len(list_path)-1) and (self.parent_group_name != '/'): back = back - 1
          # start constructing the merged numpy array
          where_list = []
          name_list  = []
          i=0
          try: list_path.remove("")
          except ValueError: pass #don't remove if not found.
          # Find the paths for the completed history
          while (i < back):
            path_w = ''
            for j in xrange(len(list_path) - i):
              if list_path[j] != "":
                path_w = path_w + "/" + list_path[j]
            if path_w != "":
              where_list.append(path_w)
            mylist = where_list[i].split("/")
            name_list.append(mylist[len(mylist)-1])
            i = i + 1
          # get the relative groups' data
          gb_res ={}
          gb_attrs={}
          where_list.reverse()
          name_list.reverse()
          n_tot_ts = 0
          n_params = 0
          # Retrieve every single partial history that will be merged to create the whole history
          for i in xrange(len(where_list)):
            grp = self.h5_file_w.require_group(where_list[i])
            namel = name_list[i] +'_data'
            dataset = grp.require_dataset(namel, (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float').value
            if i == 0:
              n_params = int(grp.attrs['n_params'])

            if n_params != int(grp.attrs['n_params']):
              self.raiseAnError(TypeError,'Can not merge datasets with different number of parameters')
            # get numpy array
            gb_res[i]   = dataset[:,:]
            gb_attrs[i] =grp.attrs
            n_tot_ts = n_tot_ts + int(grp.attrs['n_ts'])
          # Create numpy array
          result = np.zeros((n_tot_ts,n_params))
          ts = 0
          # Retrieve metadata
          for key in gb_res:
            arr = gb_res[key]
            result[ts:ts+arr[:,0].size,:] = arr[:,:]
            ts = ts + arr[:,0].size
            # must be checked if overlapping of time (branching for example)
          try:    attrs["output_space_headers"]   = gb_attrs[0]["output_space_headers"].tolist()
          except: attrs["output_space_headers"]   = gb_attrs[0]["output_space_headers"]
          try:    attrs["input_space_headers"]    = gb_attrs[0]["input_space_headers"].tolist()
          except:
            try:    attrs["input_space_headers"]  = gb_attrs[0]["input_space_headers"]
            except: pass
          try:    attrs["input_space_values"]     = gb_attrs[0]["input_space_values"].tolist()
          except:
            try:    attrs["input_space_values"]   = gb_attrs[0]["input_space_values"]
            except: pass
          attrs["n_params"]        = gb_attrs[0]["n_params"]
          attrs["parent"]          = where_list[0]
          attrs["start_time"]      = result[0,0]
          attrs["end_time"]        = result[result[:,0].size-1,0]
          attrs["n_ts"]            = result[:,0].size
          attrs["source_type"]     = gb_attrs[0]["source_type"]
          attrs["input_file"]      = []
          attrs["source_file"]     = []
          for key in gb_res.keys():
            for attr in gb_attrs[key].keys():
              if attr not in ["output_space_headers","input_space_headers","input_space_values","n_params","parent_id","start_time","end_time","n_ts","source_type"]:
                if attr not in attrs.keys(): attrs[attr] = []
                try   : attrs[attr].append(json.loads(gb_attrs[key][attr]))
                except:
                  if type(attrs[attr]) == list: attrs[attr].append(gb_attrs[key][attr])
              if attrs["source_type"] == 'csv': attrs["source_file"].append(gb_attrs[key]["source_file"])

        else: self.raiseAnError(IOError,'Filter not recognized in hdf5Database.retrieveHistory function. Filter = ' + str(filter))
    else: self.raiseAnError(IOError,'History named ' + name + ' not found in database')

    return(result,attrs)

  def closeDatabaseW(self):
    """
    Function to close the database
    @ In,  None
    @ Out, None
    """
    self.h5_file_w.close()
    self.fileOpen       = False
    return

  def openDatabaseW(self,filename,mode='w'):
    """
    Function to open the database
    @ In,  filename : name of the file (string)
    @ In,  mode     : open mode (default "w=write")
    @ Out, fh5      : hdf5 object
    """
    fh5 = h5.File(filename,mode)
    self.fileOpen       = True
    return fh5

  def __returnParentGroupPath(self,parent_name):
    """
    Function to return a parent group Path
    @ In, parent_name, parent ID
    """
    if parent_name != '/':
      parent_group_name = '-$' # control variable
      for index in xrange(len(self.allGroupPaths)):
        test_list = self.allGroupPaths[index].split('/')
        if test_list[len(test_list)-1] == parent_name:
          parent_group_name = self.allGroupPaths[index]
          break
    else: parent_group_name = None
    return parent_group_name

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False
