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
    ''' 
      database name (i.e. arbitrary name).
      It is the database name that has been found in the xml input
    '''
    self.name       = name
    '''  
      Database type :
      -> The structure type is "inferred" by the first group is going to be added
      * MC  = MonteCarlo => Storing by a Parallel structure 
      * DET = Dynamic Event Tree => Storing by a Hierarchical structure
    '''
    self.type       = None
    ''' 
      .H5 file name (to be created or read) 
    '''
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
      # Open file
      self.h5_file_w = self.openDataBaseW(self.filenameAndPath,'r+')
      # Call the private method __createObjFromFile, that constructs the list of the paths "self.allGroupPaths"
      # and the dictionary "self.allGroupEnds" based on the database that already exists
      self.__createObjFromFile()
      # "self.firstRootGroup", true if the root group is present (or added), false otherwise
      self.firstRootGroup = True
    else:
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
      try:
        self.allGroupEnds[name]  = obj.attrs["EndGroup"]
      except: 

        print('DATABASE HDF5 : not found attribute EndGroup in group ' + name + '.Set True.')
        self.allGroupEnds[name]  = True
    return
   
  def addGroup(self,gname,attributes,source):
    '''
    Function to add a group into the database
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    '''
    #if self.type == 'DET':
    if source['type'] == 'Datas':
      self.addGroupDatas(gname,attributes,source)
      return
    
    if 'parent_id' in attributes.keys():
      '''
        If Hierarchical structure, firstly add the root group 
      '''
      if not self.firstRootGroup:
        self.__addGroupRootLevel(gname,attributes,source)
        self.firstRootGroup = True
        self.type = 'DET'
      else:
        # Add sub group in the Hierarchical structure
        self.__addSubGroup(gname,attributes,source)
    else:
      # Parallel structure (always root level)
      self.__addGroupRootLevel(gname,attributes,source)
      self.firstRootGroup = True
      self.type = 'MC'
    return

  def addGroupInit(self,gname,attributes=None):
    '''
    Function to add an empty group to the database
    This function is generally used when the user provides a rootname in the input
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, gname      : group name
    @ Out, None
    '''
    for index in xrange(len(self.allGroupPaths)):
      comparisonName = self.allGroupPaths[index]
      if gname in comparisonName:
        raise IOError("Root Group named " + gname + " already present in database " + self.name)
    
    self.parent_group_name = "/" + gname 
    # Create the group
    grp = self.h5_file_w.create_group(gname)
    
    # Add metadata
    if attributes:
      for key in attributes.keys():
        grp.attrs[key] = attributes[key]

    self.allGroupPaths.append("/" + gname)
    self.allGroupEnds["/" + gname] = False
    
    return

  def __addGroupRootLevel(self,gname,attributes,source):
    '''
    Function to add a group into the database (root level)
    @ In, gname      : group name
    @ In, attributes : dictionary of attributes that must be added as metadata
    @ In, source     : data source (for example, csv file)
    @ Out, None
    '''
    # Check in the "self.allGroupPaths" list if a group is already present... 
    # If so, error (Deleting already present information is not desiderable) 
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
        grp = rootgrp.create_group(gname)
      else: grp = self.h5_file_w.create_group(gname)

      print('DATABASE HDF5 : Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
      # Create dataset in this newly added group
      dataset = grp.create_dataset(gname+"_data", dtype="float", data=data)
      # Add metadata
      grp.attrs["headers"   ] = headers
      grp.attrs["n_params"  ] = data[0,:].size
      grp.attrs["parent_id" ] = "root"
      grp.attrs["start_time"] = data[0,0]
      grp.attrs["end_time"  ] = data[data[:,0].size-1,0]
      grp.attrs["n_ts"      ] = data[:,0].size
      grp.attrs["EndGroup"  ] = True
      #FIXME should all the exceptions below be except KeyError to allow for other errors to break code?
      try: grp.attrs[toString("input_file")] = toString(" ".join(attributes["input_file"])) if type(attributes["input_file"]) == type([]) else toString(attributes["input_file"])
      except: pass        
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
        try: grp.attrs[toBytes(attr)]=[toBytes(x) for x in attributes[attempt_attr[attr]]]
        except KeyError: pass
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


  def addGroupDatas(self,gname,attributes,source):
    for index in xrange(len(self.allGroupPaths)):
      comparisonName = self.allGroupPaths[index]
      if gname in comparisonName: raise IOError("Group named " + gname + " already present in database " + self.name)

    # get input parameters
    inputSpace  = source['name'].getInpParametersValues()
    outputSpace = source['name'].getOutParametersValues()
    # Retrieve the headers from the data (inputs and outputs)
    headers_in  = inputSpace.keys()
    headers_out = outputSpace.keys()

    parent_name = self.parent_group_name.replace('/', '')
    
    # Create the group
    if parent_name != '/':
      parent_group_name = self.__returnParentGroupPath(parent_name)
      # Retrieve the parent group from the HDF5 database
      if parent_group_name in self.h5_file_w: parentgroup_obj = self.h5_file_w.require_group(parent_group_name)
      else: raise ValueError("NOT FOUND group named " + parentgroup_obj)
      #grp_in  = rootgrp.create_group(gname + b'|input_space' )
      #grp_out = rootgrp.create_group(gname + b'|output_space')
    else: 
      parentgroup = self.h5_file_w
      #grp_in  = self.h5_file_w.create_group(gname + b'|input_space' )
      #grp_out = self.h5_file_w.create_group(gname + b'|output_space')
    # for a "histories" type we create a number of groups = number of histories (compatibility with loading structure)
    data_in = inputSpace.values()
    if source['name'].type in ['Histories','TimePointSet']:
      groups = []
      for run in range(len(data_in)): 
        groups.append(parentgroup_obj.create_group(gname + '|' +str(run)))
        
        groups[run].attrs['input_space'] = [[headers_in[x].encode() for x in range(len(headers_in))],]
      
      
      
      
    elif toaddDictionaryCase:
      pass
    else:
      groups = parentgroup_obj.create_group(gname)
    
    
    
    if isinstance(data_in[0], dict):
      #it is an Histories type
      datain = np.zeros((len(data_in),len(data_in[0].values())))
      headers_in = data_in[0].keys()
      
      for run in range(len(data_in)):
        for param in range(len(data_in[run].values())): datain[int(run),param] = copy.deepcopy(data_in[run].values()[param])

        
       
      
      
      
    else:
      data_in = inputSpace.values()
      datain = np.zeros((data_in[0].size,len(data_in)))
      for run in range(len(data_in)):
        for param in range(data_in[run].size): datain[param,int(run)] = copy.deepcopy(data_in[run][param])
    dataset_in  = grp_in.create_dataset(gname + b'|input_space', dtype="float", data=datain)

    grp_in.attrs[b"headers"   ] = [headers_in[x].encode() for x in range(len(headers_in))] 
    grp_in.attrs["n_params"   ] = len(headers_in)
    grp_in.attrs["parent_id"  ] = parent_name
    grp_in.attrs["start_time" ] = 0.0
    grp_in.attrs["end_time"   ] = 0.0
    grp_in.attrs["EndGroup"   ] = True
    grp_in.attrs["main_class" ] = "Datas"
    grp_in.attrs["source_type"] = source['name'].type
    
    
    data_out = outputSpace.values()
    if isinstance(data_out[0], dict):
      #it means it is an Histories type => create a numpy array on the fly
      # since we have a dictionary of histories, we are going to create a number of datasets = number of history
      dataout = []
      dataset_out = []
      starttimes = []
      endtimes   = []
      headers_out = data_out[0].keys()
      for run in range(len(data_out)):
        dataout.append(np.zeros((data_out[run].values()[0].size,len(data_out[run].values()))))
        for param in range(len(data_out[run].values())):
          dataout[run][:,param] = data_out[run].values()[param][:]
        dataset_out.append(grp_out.create_dataset(gname + b'|output_space'+"_data$" + str(run), dtype="float", data=dataout))
        index_sttime = None
        try: index_sttime = data_out[run].keys().index('time')
        except: pass
        if  index_sttime: 
          starttimes.append(dataout[run][0,index_sttime])
          endtimes.append(dataout[run][dataout[run][:,0].size-1,index_sttime])
    else:
      dataout = np.zeros((data_out[0].size,len(data_out)))
      for run in range(len(data_out)):
        for param in range(data_out[run].size): dataout[param,int(run)] = copy.deepcopy(data_out[run][param])
      dataset_out = grp_out.create_dataset(gname + b'|output_space'+"_data", dtype="float", data=dataout)

    grp_out.attrs["headers"    ] = [headers_out[x].encode() for x in range(len(headers_out))] 
    grp_out.attrs["n_params"   ] = len(headers_out)
    grp_out.attrs["parent_id"  ] = parent_name
    grp_out.attrs["EndGroup"   ] = True
    grp_out.attrs["main_class" ] = "Datas"
    grp_out.attrs["source_type"] = source['name'].type

    # Add the group name into the list "self.allGroupPaths" and 
    # set the relative bool flag into the dictionary "self.allGroupEnds"
    if parent_group_name != "/":
      self.allGroupPaths.append(parent_group_name + "/" + gname)
      self.allGroupEnds[parent_group_name + "/" + gname] = True
    else:
      self.allGroupPaths.append("/" + gname)
      self.allGroupEnds["/" + gname] = True
      
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
      try: parent_name = attributes["parent_id"]
      except: raise IOError ("NOT FOUND attribute <parent_id> into <attributes> dictionary")
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
      sgrp.attrs["headers"   ] = headers
      sgrp.attrs["n_params"  ] = data[0,:].size
      sgrp.attrs["parent"    ] = "root"
      sgrp.attrs["start_time"] = data[0,0]
      sgrp.attrs["end_time"  ] = data[data[:,0].size-1,0]
      sgrp.attrs["n_ts"      ] = data[:,0].size
      sgrp.attrs["EndGroup"  ] = True

      try: sgrp.attrs["input_file"] = attributes["input_file"]
      except: pass
      sgrp.attrs["source_type"] = source['type']
      if source['type'] == 'csv': sgrp.attrs["source_file"] = source['name']
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
        try: sgrp.attrs[toBytes(attr)]=[toBytes(x) for x in attributes[attempt_attr[attr]]]
        except KeyError: pass
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
        try: list_path.remove("")
        except: pass
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
          gb_res[i]   = dataset[:,:]
          gb_attrs[i] =grp.attrs        
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
        
        attrs["headers"]         = gb_attrs[0]["headers"].tolist()
        attrs["n_params"]        = gb_attrs[0]["n_params"]       
        attrs["parent_id"]       = where_list[0]
        attrs["start_time"]      = result[0,0]
        attrs["end_time"]        = result[result[:,0].size-1,0]
        attrs["n_ts"]            = result[:,0].size
        attrs["source_type"]     = gb_attrs[0]["source_type"]
        attrs["input_file"]      = []
        attrs["source_file"]     = []
        try:
          par = gb_attrs[0]["branch_changed_param"]
          attrs["branch_changed_param"]    = []
        except: pass
        try:
          par = gb_attrs[0]["conditional_prb"]
          attrs["conditional_prb"] = []
        except: pass
        try:
          par2 = gb_attrs[0]["branch_changed_param_value"]
          attrs["branch_changed_param_value"]    = []
        except: pass
        try:
          par3 = gb_attrs[0]["initiator_distribution"]
          attrs["initiator_distribution"]    = []
        except: pass
        try:
          par4 = gb_attrs[0]["Probability_threshold"]
          attrs["Probability_threshold"]    = []
        except: pass
        try:
          par5 = gb_attrs[0]["end_timestep"]
          attrs["end_timestep"]    = []
        except: pass                    
        for key in gb_res:
          try: attrs["input_file"].append(gb_attrs[key]["input_file"])
          except: pass  
          if attrs["source_type"] == 'csv': attrs["source_file"].append(gb_attrs[key]["source_file"])
          try: attrs["branch_changed_param"].append(gb_attrs[key]["branch_changed_param"])
          except: pass
          try: attrs["conditional_prb"].append(gb_attrs[key]["conditional_prb"])
          except: pass            
          try: attrs["branch_changed_param_value"].append(gb_attrs[key]["branch_changed_param_value"])
          except: pass                        
          try: attrs["initiator_distribution"].append(gb_attrs[key]["initiator_distribution"])
          except: pass                                    
          try: attrs["Probability_threshold"].append(gb_attrs[key]["Probability_threshold"])
          except: pass                                                
          try: attrs["end_timestep"].append(gb_attrs[key]["end_timestep"])
          except: pass                                                            
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
          except: pass
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
          attrs["headers"]         = gb_attrs[0]["headers"]
          attrs["n_params"]        = gb_attrs[0]["n_params"]       
          attrs["parent"]          = where_list[0]
          attrs["start_time"]      = result[0,0]
          attrs["end_time"]        = result[result[:,0].size-1,0]
          attrs["n_ts"]            = result[:,0].size
          attrs["source_type"]     = gb_attrs[0]["source_type"]
          attrs["input_file"]      = []
          attrs["source_file"]     = []
          try:
            par = gb_attrs[0]["branch_changed_param"]
            attrs["branch_changed_param"]    = []
            attrs["conditional_prb"] = []
          except: pass
          try:
            par2 = gb_attrs[0]["branch_changed_param_value"]
            attrs["branch_changed_param_value"]    = []
          except: pass
          try:
            par3 = gb_attrs[0]["initiator_distribution"]
            attrs["initiator_distribution"]    = []
          except: pass
          try:
            par4 = gb_attrs[0]["Probability_threshold"]
            attrs["Probability_threshold"]    = []
          except: pass
          try:
            par5 = gb_attrs[0]["end_timestep"]
            attrs["end_timestep"]    = []
          except: pass
          
          for key in gb_res:
            try: attrs["input_file"].append(gb_attrs[key]["input_file"])
            except: pass
            if attrs["source_type"] == 'csv': attrs["source_file"].append(gb_attrs[key]["source_file"])
            try: attrs["branch_changed_param"].append(gb_attrs[key]["branch_changed_param"])
            except: pass
            try: attrs["conditional_prb"].append(gb_attrs[key]["conditional_prb"])
            except: pass            
            try: attrs["branch_changed_param_value"].append(gb_attrs[key]["branch_changed_param_value"])
            except: pass                        
            try: attrs["initiator_distribution"].append(gb_attrs[key]["initiator_distribution"])
            except: pass                                    
            try: attrs["Probability_threshold"].append(gb_attrs[key]["Probability_threshold"])
            except: pass                                                
            try: attrs["end_timestep"].append(gb_attrs[key]["end_timestep"])
            except: pass                                                                          
        else: raise RunTimeError("Error. Filter not recognized in hdf5Database.retrieveHistory function. Filter = " + str(filter)) 
    else: raise RunTimeError("History named " + name + "not found in database")
    
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
      
