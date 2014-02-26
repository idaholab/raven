'''
Created on Mar 25, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
import numpy as np
import h5py  as h5
import os
import copy

from utils import *

'''
  *************************
  *  HDF5 DATABASE CLASS  *
  *************************
'''
class hdf5Database(object):
  '''
  class to create a h5py (hdf5) database
  '''
  def __init__(self,name, databaseDir, filename=None):
    # database name (i.e. arbitrary name).
    # It is the database name that has been found in the xml input
    self.name       = name
    # Database type :
    # -> The structure type is "inferred" by the first group is going to be added
    # * MC  = MonteCarlo => Storing by a Parallel structure 
    # * DET = Dynamic Event Tree => Storing by a Hierarchical structure
    self.type       = None
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
      # check if the file exists
      if not os.path.isfile(self.filenameAndPath): raise IOError('DATABASE HDF5 : ERROR -> when you specify a filename for HDF5, the file must exist \n File not found: '+self.filenameAndPath)
      # Open file
      self.h5_file_w = self.openDataBaseW(self.filenameAndPath,'r+')
      # Call the private method __createObjFromFile, that constructs the list of the paths "self.allGroupPaths"
      # and the dictionary "self.allGroupEnds" based on the database that already exists
      self.parent_group_name = b'/'
      self.__createObjFromFile()
      # "self.firstRootGroup", true if the root group is present (or added), false otherwise
      self.firstRootGroup = True
    else:
      # check if the file exists, in case warn the user and delete it
      if os.path.isfile(self.filenameAndPath):
        print('DATABASE HDF5 : Warning -> The HDF5 database already exist in directory "'+self.databaseDir+'". \nDATABASE HDF5 : Warning -> This action will delete the old database!')
        os.remove(self.filenameAndPath)
      # self.h5_file_w is the HDF5 object. Open the database in "write only" mode 
      self.h5_file_w = self.openDataBaseW(self.filenameAndPath,'w')
      # Add the root as first group
      self.allGroupPaths.append("/")
      # The root group is not an end group
      self.allGroupEnds["/"] = False
      # The first root group has not been added yet
      self.firstRootGroup = False
      # The root name is / . it can be changed if addGroupInit is called
      self.parent_group_name = b'/'
    
  def __createObjFromFile(self):
    '''
    Function to create the list "self.allGroupPaths" and the dictionary "self.allGroupEnds"
    from a database that already exists. It uses the h5py method "visititems" in conjunction
    with the private method "self.__isGroup"
    @ In, None  
    @ Out, None
    '''
    self.allGroupPaths = []
    self.allGroupEnds  = {}
    if not self.fileOpen:
      self.h5_file_w = self.openDataBaseW(self.filenameAndPath,'a')
    self.h5_file_w.visititems(self.__isGroup)

  def __isGroup(self,name,obj):
    '''
    Function to check if an object name is of type "group". If it is, the function stores 
    its name into the "self.allGroupPaths" list and update the dictionary "self.allGroupEnds"
    @ In, name : object name
    @ In, obj  : the object itself
    '''
    if isinstance(obj,h5.Group):
      self.allGroupPaths.append(name)
      if "EndGroup" in obj.attrs:
        self.allGroupEnds[name]  = obj.attrs["EndGroup"]
      else: 
        print('DATABASE HDF5 : not found attribute EndGroup in group ' + name + '.Set True.')
        self.allGroupEnds[name]  = True
      if "rootname" in obj.attrs: self.parent_group_name = name
    return
   
  def addGroup(self,gname,attributes,source,upGroup=False):
    '''
    Function to add a group into the database
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    '''
    if source['type'] == 'Datas':
      self.addGroupDatas(gname,attributes,source)
      return
    
    if 'parent_id' in attributes.keys():
      '''
        If Hierarchical structure, firstly add the root group 
      '''
      if not self.firstRootGroup:
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
    return

  def addGroupInit(self,gname,attributes=None,upGroup=False):
    '''
    Function to add an empty group to the database
    This function is generally used when the user provides a rootname in the input
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, gname      : group name
    @ Out, None
    '''
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        if gname in comparisonName: raise IOError("Root Group named " + gname + " already present in database " + self.name)
    self.parent_group_name = "/" + gname 
    # Create the group
    grp = self.h5_file_w.create_group(gname)
    # Add metadata
    if attributes:
      for key in attributes.keys():
        grp.attrs[key] = attributes[key]
    grp.attrs['rootname'] = True
    grp.attrs['EndGroup'] = False
    self.allGroupPaths.append("/" + gname)
    self.allGroupEnds["/" + gname] = False
    return

  def __addGroupRootLevel(self,gname,attributes,source,upGroup=False):
    '''
    Function to add a group into the database (root level)
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    '''
    # Check in the "self.allGroupPaths" list if a group is already present... 
    # If so, error (Deleting already present information is not desiderable) 
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        if gname in comparisonName: raise IOError("Group named " + gname + " already present in database " + self.name)
    if source['type'] == 'csv':
      # Source in CSV format
      f = open(source['name'],'rb')
      # Retrieve the headers of the CSV file
      firstRow = f.readline().strip(b"\r\n")
      #firstRow = f.readline().translate(None,"\r\n")
      headers = firstRow.split(b",")
      #print(repr(headers))
      # Load the csv into a numpy array(n time steps, n parameters) 
      data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
      # First parent group is the root name
      parent_name = self.parent_group_name.replace('/', '')
      # Create the group
      if parent_name != '/':
        parent_group_name = self.__returnParentGroupPath(parent_name)
        # Retrieve the parent group from the HDF5 database
        if parent_group_name in self.h5_file_w: rootgrp = self.h5_file_w.require_group(parent_group_name)
        else: raise ValueError("NOT FOUND group named " + parent_group_name)
        if upGroup: 
          grp = rootgrp.require_group(gname)
          del grp[gname+"_data"]
        else: grp = rootgrp.create_group(gname)
      else: 
        if upGroup: grp = self.h5_file_w.require_group(gname)
        else:       grp = self.h5_file_w.create_group(gname) 
      print('DATABASE HDF5 : Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
      # Create dataset in this newly added group
      dataset = grp.create_dataset(gname+"_data", dtype="float", data=data)
      # Add metadata
      grp.attrs["output_space_headers"   ] = headers
      grp.attrs["n_params"  ] = data[0,:].size
      grp.attrs["parent_id" ] = "root"
      grp.attrs["start_time"] = data[0,0]
      grp.attrs["end_time"  ] = data[data[:,0].size-1,0]
      grp.attrs["n_ts"      ] = data[:,0].size
      grp.attrs["EndGroup"  ] = True
      #FIXME should all the exceptions below be except KeyError to allow for other errors to break code?
      if "input_file" in attributes: grp.attrs[toString("input_file")] = toString(" ".join(attributes["input_file"])) if type(attributes["input_file"]) == type([]) else toString(attributes["input_file"])
      grp.attrs["source_type"] = source['type']
          
      if source['type'] == 'csv': grp.attrs["source_file"] = source['name']

      #look for keyword attributes from the sampler
      attempt_attr= {'branch_changed_param'      :'branch_changed_param',
                     'branch_changed_param_value':'branch_changed_param_value',
                     'conditional_prb'           :'conditional_prb',
                     'initiator_distribution'    :'initiator_distribution',
                     'Probability_threshold'     :'PbThreshold',
                     'quad_pts'                  :'quad_pts',
                     'partial_coeffs'            :'partial_coeffs',
                     'exp_order'                 :'exp_order',
                     }
      for attr in attempt_attr.keys():
        if attempt_attr[attr] in attributes:
          grp.attrs[toBytes(attr)]=[toBytes(x) for x in attributes[attempt_attr[attr]]]
    else:
      # do something else
      pass
    # Add the group name into the list "self.allGroupPaths" and 
    # set the relative bool flag into the dictionary "self.allGroupEnds"
    if parent_group_name != "/":
      self.allGroupPaths.append(parent_group_name + "/" + gname)
      self.allGroupEnds[parent_group_name + "/" + gname] = True
    else:
      self.allGroupPaths.append("/" + gname)
      self.allGroupEnds["/" + gname] = True


  def addGroupDatas(self,gname,attributes,source,upGroup=False):
    '''
    Function to add a data (class Datas) or Dictionary into the DataBase
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, a TimePointSet)
    @ In, upGroup    : update Group????
    @ Out, None
    '''
    if not upGroup:
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        if gname in comparisonName: raise IOError("Group named " + gname + " already present in database " + self.name)
    parent_name = self.parent_group_name.replace('/', '')
    # Create the group
    if parent_name != '/':
      parent_group_name = self.__returnParentGroupPath(parent_name)
      # Retrieve the parent group from the HDF5 database
      if parent_group_name in self.h5_file_w: parentgroup_obj = self.h5_file_w.require_group(parent_group_name)
      else: raise ValueError("NOT FOUND group named " + parentgroup_obj)
    else: parentgroup_obj = self.h5_file_w
    
    if type(source['name']) == dict:
      print('Please Andrea change bytes ')
      # create the group
      if upGroup: 
        groups = parentgroup_obj.require_group(gname)
        del groups[gname+"_data"]
      else: groups = parentgroup_obj.create_group(gname)
      groups.attrs[b'main_class' ] = b'PythonType'
      groups.attrs[b'source_type'] = b'Dictionary'
      if 'input_space_params' in source['name'].keys():
        testkey   = []
        testvalue = []
        input_space_params_keys = list(source['name']['input_space_params'].keys())
        for i in range(len(input_space_params_keys)): 
          testkey.append(bytes(input_space_params_keys[i]))
        input_space_params_values = list(source['name']['input_space_params'].values())
        for i in range(len(input_space_params_values)): 
          testvalue.append(toBytes(input_space_params_values[i]))  
        groups.attrs[b'input_space_headers' ] = copy.deepcopy(testkey) 
        groups.attrs[b'input_space_values' ] = copy.deepcopy(testvalue)
        out_headers = list(source['name'].keys())
      else: out_headers = list(source['name'].keys())
      groups.attrs[b'n_params'   ] = len(out_headers)  
      groups.attrs[b'output_space_headers'] = copy.deepcopy([toBytes(str(out_headers[i]))  for i in range(len(out_headers))]) 
      groups.attrs[b'EndGroup'   ] = True
      groups.attrs[b'parent_id'  ] = parent_name
      maxsize = 0
      for key in source['name'].keys():
        if key == 'input_space_params': continue
        if type(source['name'][key]) == np.ndarray:
          if maxsize < source['name'][key].size : actualone = source['name'][key].size
        elif type(source['name'][key]) in [int,float,bool]: actualone = 1
        else: raise IOError('DATABASE HDF5 : The type of the dictionary paramaters must be within float,bool,int,numpy.ndarray')
        if maxsize < actualone: maxsize = actualone
      groups.attrs[b'n_ts'  ] = maxsize
      dataout = np.zeros((maxsize,len(out_headers)))
      cnt = 0
      for index in range(len(source['name'].keys())):
        if list(source['name'].keys())[index]== 'input_space_params': cnt -= cnt  
        else: 
          cnt = index
          name_values = list(source['name'].values())
          if type(name_values[cnt]) == np.ndarray:  dataout[0:name_values[cnt].size,cnt] =  copy.deepcopy(name_values[cnt][:])
          else: dataout[0,cnt] = copy.deepcopy(name_values[cnt])
      # create the data set
      dataset_out = groups.create_dataset(gname + "_data", dtype="float", data=dataout)     
      if parent_group_name != "/":
        self.allGroupPaths.append(parent_group_name + "/" + gname)
        self.allGroupEnds[parent_group_name + "/" + gname] = True
      else:
        self.allGroupPaths.append("/" + gname)
        self.allGroupEnds["/" + gname] = True         
    else:
      # Retrieve the headers from the data (inputs and outputs)
      headers_in  = source['name'].getInpParametersValues().keys()
      headers_out = source['name'].getOutParametersValues().keys()
      # for a "histories" type we create a number of groups = number of histories (compatibility with loading structure)
      data_in  = source['name'].getInpParametersValues().values()
      data_out = source['name'].getOutParametersValues().values()    
      if source['name'].type in ['Histories','TimePointSet']:
        groups = []
        if 'Histories' in source['name'].type: nruns = len(data_in)
        else:                                  nruns = data_in[0].size
        for run in range(nruns): 
          if upGroup: 
            groups.append(parentgroup_obj.require_group(gname + b'|' +str(run)))
            if (gname + "_data") in groups[run] : del groups[run][gname+"_data"]
          else:groups.append(parentgroup_obj.create_group(gname + b'|' +str(run)))
          
          groups[run].attrs[b'source_type'] = bytes(source['name'].type)
          groups[run].attrs[b'main_class' ] = b'Datas'
          groups[run].attrs[b'EndGroup'   ] = True
          groups[run].attrs[b'parent_id'  ] = parent_name
          if source['name'].type == 'Histories': 
            groups[run].attrs[b'input_space_headers' ] = copy.deepcopy([bytes(data_in[run].keys()[i])  for i in range(len(data_in[run].keys()))]) 
            groups[run].attrs[b'output_space_headers'] = copy.deepcopy([bytes(data_out[run].keys()[i])  for i in range(len(data_out[run].keys()))]) 
            groups[run].attrs[b'input_space_values'  ] = copy.deepcopy(data_in[run].values())
            groups[run].attrs[b'n_params'            ] = len(data_out[run].keys())
            #collect the outputs
            dataout = np.zeros((data_out[run].values()[0].size,len(data_out[run].values())))
            for param in range(len(data_out[run].values())): dataout[:,param] = data_out[run].values()[param][:]
            groups[run].create_dataset(gname +"_data" , dtype="float", data=copy.deepcopy(dataout))
            groups[run].attrs[b'n_ts'                ] = len(data_out[run].values())
          else:
            groups[run].attrs[b'input_space_headers' ] = copy.deepcopy([bytes(headers_in[i])  for i in range(len(headers_in))]) 
            groups[run].attrs[b'output_space_headers'] = copy.deepcopy([bytes(headers_out[i])  for i in range(len(headers_out))]) 
            groups[run].attrs[b'input_space_values'  ] = copy.deepcopy([np.atleast_1d(np.array(data_in[x][run])) for x in range(len(data_in))])
            groups[run].attrs[b'n_params'            ] = len(headers_out)
            groups[run].attrs[b'n_ts'                ] = 1
            #collect the outputs
            dataout = np.zeros((1,len(data_out)))
            for param in range(len(data_out)): dataout[0,param] = copy.deepcopy(data_out[param][run])
            groups[run].create_dataset(gname +"_data", dtype="float", data=dataout)          
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
        groups.attrs[b'main_class' ] = b'Datas'
        groups.attrs[b'source_type'] = bytes(source['name'].type)
        groups.attrs[b'n_params'   ] = len(headers_out)
        groups.attrs[b'input_space_headers' ] = copy.deepcopy([bytes(headers_in[i])  for i in range(len(headers_in))]) 
        groups.attrs[b'output_space_headers'] = copy.deepcopy([bytes(headers_out[i])  for i in range(len(headers_out))]) 
        groups.attrs[b'input_space_values' ] = copy.deepcopy([np.array(data_in[i])  for i in range(len(data_in))])
        groups.attrs[b'source_type'] = bytes(source['name'].type)
        groups.attrs[b'EndGroup'   ] = True
        groups.attrs[b'parent_id'  ] = parent_name
        dataout = np.zeros((data_out[0].size,len(data_out)))
        groups.attrs[b'n_ts'  ] = data_out[0].size
        for run in range(len(data_out)): dataout[:,int(run)] = copy.deepcopy(data_out[run][:])
        dataset_out = groups.create_dataset(gname + "_data", dtype="float", data=dataout)     
        if parent_group_name != "/":
          self.allGroupPaths.append(parent_group_name + "/" + gname)
          self.allGroupEnds[parent_group_name + "/" + gname] = True
        else:
          self.allGroupPaths.append("/" + gname)
          self.allGroupEnds["/" + gname] = True   
      else: raise IOError('DATABASE HDF5 : The function addGroupDatas accepts Data(s) or dictionaries as inputs only!!!!!')
  
  def __addSubGroup(self,gname,attributes,source):
    '''
    Function to add a group into the database (Hierarchical)
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    '''
    if source['type'] == 'csv':
      # Source in CSV format
      f = open(source['name'],'rb')
      # Retrieve the headers of the CSV file
      # Retrieve the header of the CSV file
      headers = f.readline().split(b",")
      # Load the csv into a numpy array(n time steps, n parameters)
      data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
      # Check if the parent attribute is not null
      # In this case append a subgroup to the parent group
      # Otherwise => it's the main group
      if "parent_id" in attributes:
        parent_name = attributes["parent_id"]
      else:
        raise IOError ("NOT FOUND attribute <parent_id> into <attributes> dictionary")
      # Find parent group path
      if parent_name != '/':
        parent_group_name = self.__returnParentGroupPath(parent_name)
      else: parent_group_name = parent_name   
      # Retrieve the parent group from the HDF5 database
      if parent_group_name in self.h5_file_w: grp = self.h5_file_w.require_group(parent_group_name)
      else: raise ValueError("NOT FOUND group named " + parent_group_name)  
      # The parent group is not the endgroup for this branch
      self.allGroupEnds[parent_group_name] = False
      grp.attrs["EndGroup"]   = False
      print('DATABASE HDF5 : Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
      # Create the sub-group
      sgrp = grp.create_group(gname)
      # Create data set in this new group
      dataset = sgrp.create_dataset(gname+"_data", dtype="float", data=data)
      # Add the metadata
      sgrp.attrs["output_space_headers"   ] = headers
      sgrp.attrs["n_params"  ] = data[0,:].size
      sgrp.attrs["parent"    ] = "root"
      sgrp.attrs["start_time"] = data[0,0]
      sgrp.attrs["end_time"  ] = data[data[:,0].size-1,0]
      sgrp.attrs["n_ts"      ] = data[:,0].size
      sgrp.attrs["EndGroup"  ] = True

      if "input_file" in attributes:
        grp.attrs[toString("input_file")] = toString(" ".join(attributes["input_file"])) if type(attributes["input_file"]) == type([]) else toString(attributes["input_file"])
      grp.attrs["source_type"] = source['type']
          
      if source['type'] == 'csv': grp.attrs["source_file"] = source['name']
      #look for keyword attributes from the sampler
      attempt_attr= {'branch_changed_param'      :'branch_changed_param',
                     'branch_changed_param_value':'branch_changed_param_value',
                     'conditional_prb'           :'conditional_prb',
                     'initiator_distribution'    :'initiator_distribution',
                     'Probability_threshold'     :'PbThreshold',
                     'quad_pts'                  :'quad_pts',
                     'partial_coeffs'            :'partial_coeffs',
                     'exp_order'                 :'exp_order',
                     }
      for attr in attempt_attr.keys():
        if attempt_attr[attr] in attributes:
          sgrp.attrs[toBytes(attr)]=[toBytes(x) for x in attributes[attempt_attr[attr]]]
    else:
      # do something else
      pass
    # The sub-group is the new ending group
    if parent_group_name != "/":
      self.allGroupPaths.append(parent_group_name + "/" + gname)
      self.allGroupEnds[parent_group_name + "/" + gname] = True
    else:
      self.allGroupPaths.append("/" + gname) 
      self.allGroupEnds["/" + gname] = True
    return

  def computeBack(self,nameFrom,nameTo):
    '''
    Function to compute the number of step back from a group to another
    @ In,  nameFrom : group name (from)
    @ In,  nameTo   : group name (to)
    @ Out, back     : number of step back (integer)
    '''
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
    if not found: raise Exception("ERROR: Group named " + nameTo + " not found in the HDF5 database" + self.filenameAndPath)
    else: listGroups = path.split("/")  # Split the path in order to create a list of the groups in this history
    # Retrieve indeces of groups "nameFrom" and "nameTo" v
    fr = listGroups.index(nameFrom)
    to = listGroups.index(nameTo)
    # Compute steps back
    back = to - fr
    return back

  def retrieveAllHistoryPaths(self,rootName=None):
    '''
    Function to create a list of all the histories' paths present in an existing database
    @ In,  rootName (optional), It's the root name, if present, only the groups that have this root are going to be returned
    @ Out, List of the histories' paths
    '''
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
    '''
    Function to create a list of all the histories' names present in an existing database
    @ In,  rootName (optional), It's the root name, if present, only the history names that have this root are going to be returned
    @ Out, List of the histories' names
    '''
    if not self.fileOpen: self.__createObjFromFile() # Create the "self.allGroupPaths" list from the existing database
    workingList = []
    for index in xrange(len(self.allGroupPaths)):
      if self.allGroupEnds[self.allGroupPaths[index]]:
        if rootName and not (rootName in self.allGroupPaths[index].split('/')[1]): continue
        workingList.append(self.allGroupPaths[index].split('/')[len(self.allGroupPaths[index].split('/'))-1])
    return workingList

  def retrieveHistory(self,name,filter=None):
    '''
    Function to retrieve the history whose end group name is "name"
    @ In,  name   : history name => It must correspond to a group name (string)
    @ In,  filter : filter for history retrieving
                    ('whole' = whole history, 
                     integer value = groups back from the group "name", 
                     or None = retrieve only the group "name". Defaul is None)
    @ Out, history: tuple where position 0 = 2D numpy array (history), 1 = dictionary (metadata) 
    '''
    list_str_w = []
    list_path  = []
    path       = ''
    found      = False
    result     = None
    attrs = {}
    
    # Check if the h5 file is already open, if not, open it
    # and create the "self.allGroupPaths" list from the existing database
    if not self.fileOpen: self.__createObjFromFile()
    # Find the endgroup that coresponds to the given name
    for i in xrange(len(self.allGroupPaths)):
      list_str_w = self.allGroupPaths[i].split("/")
      if list_str_w[len(list_str_w)-1] == name:
        found = True
        path  = self.allGroupPaths[i]
        list_path = list_str_w
        break
    if found:
      # Check the filter type
      if not filter or filter == 0:
        # Grep only History from group "name"
        grp = self.h5_file_w.require_group(path)
        # Retrieve dataset
        dataset = grp.require_dataset(name +'_data', (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float').value          
        # Get numpy array
        result = dataset[:,:]
        # Get attributes (metadata)
        attrs = grp.attrs
      elif  filter == 'whole':
        # Retrieve the whole history from group "name" to the root 
        # Start constructing the merged numpy array
        where_list = []
        name_list  = []
        if self.parent_group_name != '/': back = len(list_path)-2
        else: back = len(list_path)-1
        if back <= 0: back = 1
          
        i=0
        #Question, should all the "" be removed, or just the first?
        try: list_path.remove("")
        except ValueError:  pass #Not found.
        # Find the paths for the completed history
        while (i < back):
          path_w = ''
          for j in xrange(len(list_path) - i):
            if list_path[j] != "": path_w = path_w + "/" + list_path[j] 
          if path_w != "":  where_list.append(path_w)
          list = where_list[i].split("/")
          name_list.append(list[len(list)-1])
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
          if n_params != int(grp.attrs['n_params']): raise Exception("Can not merge datasets with different number of parameters")
          # Get numpy array
          gb_res[i]   = copy.deepcopy(dataset[:,:])
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
        try:    attrs["output_space_headers"]         = gb_attrs[0]["output_space_headers"].tolist()
        except: attrs["output_space_headers"]         = gb_attrs[0]["output_space_headers"]
        try:    attrs["input_space_headers"]         = gb_attrs[0]["input_space_headers"].tolist()
        except: 
          try:    attrs["input_space_headers"]         = gb_attrs[0]["input_space_headers"]
          except: pass
        try:    attrs["input_space_values"]         = gb_attrs[0]["input_space_values"].tolist()
        except: 
          try:    attrs["input_space_values"]         = gb_attrs[0]["input_space_values"]
          except: pass
        attrs["n_params"]        = gb_attrs[0]["n_params"]       
        attrs["parent_id"]       = where_list[0]
        attrs["start_time"]      = result[0,0]
        attrs["end_time"]        = result[result[:,0].size-1,0]
        attrs["n_ts"]            = result[:,0].size
        attrs["source_type"]     = gb_attrs[0]["source_type"]
        attrs["input_file"]      = []
        attrs["source_file"]     = []
        for param_key in ["branch_changed_param","conditional_prb",
                          "branch_changed_param_value",
                          "initiator_distribution","Probability_threshold",
                          "end_timestep"]:
          if param_key in gb_attrs[0]:
            attrs[param_key] = []
        for key in gb_res.keys():
          for param_key in ["input_file","branch_changed_param",
                            "conditional_prb","branch_changed_param_value",
                            "initiator_distribution","Probability_threshold",
                            "end_timestep"]:
            if param_key in gb_attrs[key]:
              attrs[param_key].append(gb_attrs[key][param_key])
          if attrs["source_type"] == 'csv' and 'source_file' in gb_attrs[key].keys(): attrs["source_file"].append(gb_attrs[key]["source_file"])
  
      else:
        # A number of groups' back have been inputted
        # Follow the same strategy used above (filter = whole)
        # but stop at back(th) group starting from group "name"
        if is_number(filter):
          back = int(filter) + 1
          if len(list_path) < back: raise Exception("Error. Number of branches back > number of actual branches in dataset for History ending with " + name)
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
            list = where_list[i].split("/")
            name_list.append(list[len(list)-1])
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
              raise Exception("Can not merge datasets with different number of parameters")
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
          try:    attrs["output_space_headers"]         = gb_attrs[0]["output_space_headers"].tolist()
          except: attrs["output_space_headers"]         = gb_attrs[0]["output_space_headers"]
          try:    attrs["input_space_headers"]         = gb_attrs[0]["input_space_headers"].tolist()
          except: 
            try:    attrs["input_space_headers"]         = gb_attrs[0]["input_space_headers"]
            except: pass
          try:    attrs["input_space_values"]         = gb_attrs[0]["input_space_values"].tolist()
          except: 
            try:    attrs["input_space_values"]         = gb_attrs[0]["input_space_values"]
            except: pass
          attrs["n_params"]        = gb_attrs[0]["n_params"]       
          attrs["parent"]          = where_list[0]
          attrs["start_time"]      = result[0,0]
          attrs["end_time"]        = result[result[:,0].size-1,0]
          attrs["n_ts"]            = result[:,0].size
          attrs["source_type"]     = gb_attrs[0]["source_type"]
          attrs["input_file"]      = []
          attrs["source_file"]     = []
          if "branch_changed_param" in gb_attrs[0]:
            par = gb_attrs[0]["branch_changed_param"]
            attrs["branch_changed_param"]    = []
            attrs["conditional_prb"] = []
          for param_key in ["branch_changed_param_value","initiator_distribution","Probability_threshold","end_timestep"]:
            if param_key in gb_attrs[0]:
              attrs[param_key] = []
          for key in gb_res:
            for param_key in ["input_file","branch_changed_param",
                              "conditional_prb","branch_changed_param_value",
                              "initiator_distribution","Probability_threshold",
                              "end_timestep"]:
              if param_key in gb_attrs[key]:
                attrs[param_key].append(gb_attrs[key][param_key])
            if attrs["source_type"] == 'csv': attrs["source_file"].append(gb_attrs[key]["source_file"])
                  
        else: raise IOError("Error. Filter not recognized in hdf5Database.retrieveHistory function. Filter = " + str(filter)) 
    else: raise IOError("History named " + name + " not found in database")
  
    return(copy.deepcopy(result),copy.deepcopy(attrs))

  def closeDataBaseW(self):
    '''
    Function to close the database
    @ In,  None
    @ Out, None
    '''
    self.h5_file_w.close()
    self.fileOpen       = False
    return

  def openDataBaseW(self,filename,mode='w'):
    '''
    Function to open the database
    @ In,  filename : name of the file (string)
    @ In,  mode     : open mode (default "w=write")
    @ Out, fh5      : hdf5 object
    '''
    fh5 = h5.File(filename,mode)
    self.fileOpen       = True
    return fh5
  
  def __returnParentGroupPath(self,parent_name):
    '''
    Function to return a parent group Path
    @ In, parent_name, parent ID
    '''
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
      
