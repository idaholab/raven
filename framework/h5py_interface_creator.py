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

from utils import *

'''
  *************************
  *  HDF5 DATABASE CLASS  *
  *************************
'''
class hdf5Database:
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
        ''' 
          File name on disk (file exists => fileExist flag is True) 
        '''
        self.onDiskFile = filename
        self.fileExist  = True
      else:
        ''' 
          File name on disk (file does not exist => it will create => fileExist flag is False) 
        '''
        self.onDiskFile = name + "_" + str(self.type) + ".h5" 
        self.fileExist  = False 
      '''
        Database directory
      '''
      self.databaseDir =  databaseDir
      ''' 
        Create file name and path
      '''
      self.filenameAndPath = os.path.join(self.databaseDir,self.onDiskFile)
      '''
        Is the file opened?
      ''' 
      self.fileOpen       = False
      '''
        List of the paths of all the groups that are stored in the database
      '''
      self.allGroupPaths = []
      '''
        Dictonary of boolean variables, true if the corresponding group in self.allGroupPaths
        is an ending group (no sub-groups appended), false otherwise
      '''
      self.allGroupEnds = {}      
      '''
        We can create a base empty database or we open an existing one
      '''
      if self.fileExist:
        '''  
          self.h5_file_w is the HDF5 object. Open the database in "update" mode 
        '''
        ''' Open file'''
        self.h5_file_w = self.openDataBaseW(self.filenameAndPath,'r+')
        '''
          Call the private method __createObjFromFile, that constructs the list of the paths "self.allGroupPaths"
          and the dictionary "self.allGroupEnds" based on the database that already exists
        '''
        self.__createObjFromFile()
        '''
          "self.firstRootGroup", true if the root group is present (or added), false otherwise
        '''
        self.firstRootGroup = True
      else:
        '''  
          self.h5_file_w is the HDF5 object. Open the database in "write only" mode 
        '''
        self.h5_file_w = self.openDataBaseW(self.filenameAndPath,'w')
        '''
          Add the root as first group
        '''
        self.allGroupPaths.append("/")
        '''
          The root group is not an end group
        '''
        self.allGroupEnds["/"] = False
        ''' 
          The first root group has not been added yet
        '''
        self.firstRootGroup = False
    '''
      Function to create the list "self.allGroupPaths" and the dictionary "self.allGroupEnds"
      from a database that already exists. It uses the h5py method "visititems" in conjunction
      with the private method "self.__isGroup"
      @ In, None  
      @ Out, None
    '''
    def __createObjFromFile(self):
      self.allGroupPaths = []
      self.allGroupEnds  = {}
      if not self.fileOpen:
        self.h5_file_w = self.openDataBaseW(self.filenameAndPath,'a')
      self.h5_file_w.visititems(self.__isGroup)
    '''
      Function to check if an object name is of type "group". If it is, the function stores 
      its name into the "self.allGroupPaths" list and update the dictionary "self.allGroupEnds"
      @ In, name : object name
      @ In, obj  : the object itself
    '''
    def __isGroup(self,name,obj):
      if isinstance(obj,h5.Group):
        self.allGroupPaths.append(name)
        try:
          self.allGroupEnds[name]  = obj.attrs["EndGroup"]
        except: 

          print('DATABASE HDF5 : not found attribute EndGroup in group ' + name + '.Set True.')
          self.allGroupEnds[name]  = True
    '''
      Function to add a group into the database
      @ In, gname      : group name
      @ In, attributes : dictionary of attributes that must be added as metadata
      @ In, source     : data source (for example, csv file)
      @ Out, None
    '''
    def addGroup(self,gname,attributes,source):
      #if self.type == 'DET':
      if 'parent_id' in attributes.keys():
        '''
          If Hierarchical structure, firstly add the root group 
        '''
        if not self.firstRootGroup:
          self.__addGroupRootLevel(gname,attributes,source)
          self.firstRootGroup = True
          self.type = 'DET'
        else:
          '''
            Add sub group in the Hierarchical structure
          '''
          self.__addSubGroup(gname,attributes,source)
      else:
        '''
          Parallel structure (always root level)
        '''
        self.__addGroupRootLevel(gname,attributes,source)
        self.firstRootGroup = True
        self.type = 'MC'
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
        if gname in comparisonName:
          raise IOError("Group named " + gname + " already present in database " + self.name)
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
        # First parent group is the root itself
        parent_group_name = "/"
        # Create the group
        grp = self.h5_file_w.create_group(gname)
                
        print('DATABASE HDF5 : Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
        # Create dataset in this newly added group
        dataset = grp.create_dataset(gname+"_data", dtype="float", data=data)
        # Add metadata
        grp.attrs["headers"]    = headers
        grp.attrs["n_params"]   = data[0,:].size
        grp.attrs["parent_id"]  = "root"
        grp.attrs["start_time"] = data[0,0]
        grp.attrs["end_time"]   = data[data[:,0].size-1,0]
        grp.attrs["n_ts"]       = data[:,0].size
        grp.attrs["EndGroup"]   = True
        #FIXME should all the exceptions below be except KeyError to allow for other errors to break code?
        try:
          grp.attrs["input_file"] = attributes["input_file"]
        except:
          pass        
        grp.attrs["source_type"] = source['type']
            
        if source['type'] == 'csv':
          grp.attrs["source_file"] = source['name']

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
          try:
            grp.attrs[toBytes(attr)]=[toBytes(x) for x in attributes[attempt_attr[attr]]]
            #grp.attrs[toBytes(attr)]=[toBytes(x) for x in attributes[attempt_attr[attr]]]
          except KeyError:
            pass
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

    '''
      Function to add a group into the database (Hierarchical)
      @ In, gname      : group name
      @ In, attributes : dictionary of attributes that must be added as metadata
      @ In, source     : data source (for example, csv file)
      @ Out, None
    '''
    def __addSubGroup(self,gname,attributes,source):
      if source['type'] == 'csv':
        ''' Source in CSV format'''
        f = open(source['name'],'rb')
        ''' 
          Retrieve the headers of the CSV file
        '''
        '''
          Retrieve the header of the CSV file
        '''
        headers = f.readline().split(b",")
        '''
          Load the csv into a numpy array(n time steps, n parameters)
        '''
        data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
        '''
          Check if the parent attribute is not null
          In this case append a subgroup to the parent group
          Otherwise => it's the main group
        '''
        try:
          parent_name = attributes["parent_id"]
        except:
          raise IOError ("NOT FOUND attribute <parent_id> into <attributes> dictionary")
        '''
          Find parent group path
        '''
        if parent_name != '/':
          for index in xrange(len(self.allGroupPaths)):
            test_list = self.allGroupPaths[index].split('/')
            if test_list[len(test_list)-1] == parent_name:
              parent_group_name = self.allGroupPaths[index]
              break
        else:
          parent_group_name = parent_name   
        ''' 
          Retrieve the parent group from the HDF5 database
        '''
        if parent_group_name in self.h5_file_w:
          grp = self.h5_file_w.require_group(parent_group_name)
        else:
          raise ValueError("NOT FOUND group named " + parent_group_name)  
        '''
          The parent group is not the endgroup for this branch
        '''
        self.allGroupEnds[parent_group_name] = False
        grp.attrs["EndGroup"]   = False
               
        print('DATABASE HDF5 : Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
        '''
          Create the sub-group
        '''
        sgrp = grp.create_group(gname)
        '''
          Create data set in this new group
        '''
        dataset = sgrp.create_dataset(gname+"_data", dtype="float", data=data)
        '''
          Add the metadata
        '''
        sgrp.attrs["headers"]    = headers
        sgrp.attrs["n_params"]   = data[0,:].size
        sgrp.attrs["parent"]     = "root"
        sgrp.attrs["start_time"] = data[0,0]
        sgrp.attrs["end_time"]   = data[data[:,0].size-1,0]
        sgrp.attrs["n_ts"]       = data[:,0].size
        sgrp.attrs["EndGroup"]   = True

        try:
          ''' Add input files'''
          sgrp.attrs["input_file"] = attributes["input_file"]
        except:
          pass
        sgrp.attrs["source_type"] = source['type']
        if source['type'] == 'csv':
          sgrp.attrs["source_file"] = source['name']
        try:
          ''' parameter that has been changed '''
          sgrp.attrs["branch_changed_param"] = attributes["branch_changed_param"]
          testget = sgrp.attrs["branch_changed_param"]
        except:
          ''' no branching information '''
          pass
        try:
          ''' parameter that caused the branching '''
          sgrp.attrs["branch_changed_param_value"] = attributes["branch_changed_param_value"]
        except:
          '''' no branching information '''
          pass        
        try:
          sgrp.attrs["conditional_prb"] = attributes["conditional_prb"]
        except:
          ''' no branching information => i.e. MonteCarlo data '''
          pass
        try:
          ''' initiator distribution '''
          sgrp.attrs["initiator_distribution"] = attributes["initiator_distribution"]
        except:
          ''' no branching information '''
          pass        
        try:
          ''' initiator distribution Prbability Threshold '''
          sgrp.attrs["Probability_threshold"] = attributes["PbThreshold"]
        except:
          ''' no branching information '''
          pass
        try:
          ''' End time step '''
          sgrp.attrs["end_timestep"] = attributes["end_ts"]
        except:
          ''' no branching information '''
          pass        
      else:
        ''' do something else '''
        pass
      '''
        The sub-group is the new ending group
      '''
      if parent_group_name != "/":
        self.allGroupPaths.append(parent_group_name + "/" + gname)
        self.allGroupEnds[parent_group_name + "/" + gname] = True
      else:
        self.allGroupPaths.append("/" + gname) 
        self.allGroupEnds["/" + gname] = True
      return
    '''
      Function to compute the number of step back from a group to another
      @ In,  nameFrom : group name (from)
      @ In,  nameTo   : group name (to)
      @ Out, back     : number of step back (integer)
    '''
    def computeBack(self,nameFrom,nameTo):
      '''
        "list_str_w", list in which the path for a particular group is stored (working variable)
        "path", string in which the path of the "to" group is stored
        "found", bolean variable ... I would say...self explanable :D
      '''
      list_str_w = []
      path       = ''
      found      = False
      ''' Find the path fo the "nameTo" group '''
      for i in xrange(len(self.allGroupPaths)):
        list_str_w = self.allGroupPaths[i].split("/")
        if list_str_w[len(list_str_w)-1] == nameTo:
          found = True
          path  = self.allGroupPaths[i]
          break      
      if not found:
        raise Exception("ERROR: Group named " + nameTo + " not found in the HDF5 database" + self.filenameAndPath)
      else:
        ''' Split the path in order to create a list of the groups in this history '''
        listGroups = path.split("/")
      ''' Retrieve indeces of groups "nameFrom" and "nameTo" v'''
      fr = listGroups.index(nameFrom)
      to = listGroups.index(nameTo)
      ''' Compute steps back'''
      back = to - fr
      return back

    '''
      Function to create a list of all the histories' paths present in an existing database
      @ In,  None
      @ Out, List of the histories' paths
    '''
    def retrieveAllHistoryPaths(self):
      if not self.fileOpen:
        '''
          Create the "self.allGroupPaths" list from the existing database
        '''
        self.__createObjFromFile()
      ''' Check database type '''
      if self.type == 'MC':
        ''' Parallel structure => "self.allGroupPaths" already contains the histories' paths '''
        return self.allGroupPaths
      else:
        ''' Tree structure => construct the histories' paths '''
        workingList = []
        for index in xrange(len(self.allGroupPaths)):
         if self.allGroupEnds[self.allGroupPaths[index]]:
           workingList.append(self.allGroupPaths[index])
        return workingList

    '''
      Function to create a list of all the histories' names present in an existing database
      @ In,  None
      @ Out, List of the histories' names
    '''
    def retrieveAllHistoryNames(self):
      if not self.fileOpen:
        '''
          Create the "self.allGroupPaths" list from the existing database
        '''
        self.__createObjFromFile()
      ''' Check database type '''
      if self.type == 'MC':
        ''' 
          Parallel structure => "self.allGroupPaths" already contains the histories' paths
          No need to check the end group boolean flag
        '''
        workingList = []
        for index in xrange(len(self.allGroupPaths)):
          test_list = self.allGroupPaths[index].split('/')
          workingList.append(test_list[len(test_list)-1])
          del test_list
        return workingList
      else:
        ''' 
          Tree structure => construct the histories' names 
          Check the end group boolean flag before storing the groups' names
        '''
        workingList = []
        for index in xrange(len(self.allGroupPaths)):
         if self.allGroupEnds[self.allGroupPaths[index]]:
           test_list = self.allGroupPaths[index].split('/')
           workingList.append(test_list[len(test_list)-1])
           del test_list
        return workingList

    '''
      Function to retrieve the history whose end group name is "name"
      @ In,  name   : history name => It must correspond to a group name (string)
      @ In,  filter : filter for history retrieving
                      ('whole' = whole history, 
                       integer value = groups back from the group "name", 
                       or None = retrieve only the group "name". Defaul is None)
      @ Out, history: tuple where position 0 = 2D numpy array (history), 1 = dictionary (metadata) 
    '''
    def retrieveHistory(self,name,filter=None):
      list_str_w = []
      list_path  = []
      path       = ''
      found      = False
      result     = None
      attrs = {}
      
      '''
        Check if the h5 file is already open, if not, open it
        and create the "self.allGroupPaths" list from the existing database
      '''
      if not self.fileOpen:
        self.__createObjFromFile()
      ''' 
        Find the endgroup that coresponds to the given name
      '''
      for i in xrange(len(self.allGroupPaths)):
        list_str_w = self.allGroupPaths[i].split("/")
        if list_str_w[len(list_str_w)-1] == name:
          found = True
          path  = self.allGroupPaths[i]
          list_path = list_str_w
          break
      
      if found:
        ''' Check the filter type '''
        if not filter or filter == 0:
          ''' Grep only History from group "name" '''
          grp = self.h5_file_w.require_group(path)
          ''' Retrieve dataset'''
          dataset = grp.require_dataset(name+"_data", (grp.attrs['n_ts'],grp.attrs['n_params']), dtype='float')          
#          dataset = grp.require_dataset(name=name +'_data', (grp.attrs('n_params'),grp.attrs('n_ts')), dtype='float', exact=True)
          ''' Get numpy array '''
          result = dataset[:,:]
          ''' Get attributes (metadata)'''
          attrs = grp.attrs
        elif  filter == 'whole':
          ''' 
            Retrieve the whole history from group "name" to the root 
          '''
          ''' Start constructing the merged numpy array '''
          where_list = []
          name_list  = []
          back = len(list_path)-1
          if back <= 0:
            back = 1
            
          i=0
          try:
            list_path.remove("")
          except:
            pass
          '''
            Find the paths for the completed history
          '''
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
          ''' get the relative groups' data '''
          gb_res ={}
          gb_attrs={}
          where_list.reverse()
          name_list.reverse()
          n_tot_ts = 0
          n_params = 0
          '''
            Retrieve every single partial history that will be merged to create the whole history
          '''
          for i in xrange(len(where_list)):
            grp = self.h5_file_w.require_group(where_list[i])
            namel = name_list[i] +'_data'
            dataset = grp.require_dataset(namel, (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float')
            if i == 0:
              n_params = int(grp.attrs['n_params'])
            
            if n_params != int(grp.attrs['n_params']):
              raise Exception("Can not merge datasets with different number of parameters")
            ''' Get numpy array '''
            gb_res[i]   = dataset[:,:]
            gb_attrs[i] =grp.attrs        
            n_tot_ts = n_tot_ts + int(grp.attrs['n_ts'])
          ''' Create the numpy array '''
          result = np.zeros((n_tot_ts,n_params))
          ts = 0
          ''' Retrieve the metadata'''
          for key in gb_res:
            arr = gb_res[key]
            result[ts:ts+arr[:,0].size,:] = arr[:,:]
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
          except:
            pass
          try:
            par = gb_attrs[0]["conditional_prb"]
            attrs["conditional_prb"] = []
          except:
            pass
          try:
            par2 = gb_attrs[0]["branch_changed_param_value"]
            attrs["branch_changed_param_value"]    = []
          except:
            pass
          try:
            par3 = gb_attrs[0]["initiator_distribution"]
            attrs["initiator_distribution"]    = []
          except:
            pass
          try:
            par4 = gb_attrs[0]["Probability_threshold"]
            attrs["Probability_threshold"]    = []
          except:
            pass
          try:
            par5 = gb_attrs[0]["end_timestep"]
            attrs["end_timestep"]    = []
          except:
            pass                    
          for key in gb_res:
            try:
              attrs["input_file"].append(gb_attrs[key]["input_file"])
            except:
              pass  
            if attrs["source_type"] == 'csv':
              attrs["source_file"].append(gb_attrs[key]["source_file"])
            try:
              # parameter that caused the branching
              attrs["branch_changed_param"].append(gb_attrs[key]["branch_changed_param"])
            except:
              # no branching information
              pass
            try:
              attrs["conditional_prb"].append(gb_attrs[key]["conditional_prb"])
            except:
              # no branching information => i.e. MonteCarlo data
              pass            
            try:
              attrs["branch_changed_param_value"].append(gb_attrs[key]["branch_changed_param_value"])
            except:
              # no branching information => i.e. MonteCarlo data
              pass                        
            try:
              attrs["initiator_distribution"].append(gb_attrs[key]["initiator_distribution"])
            except:
              # no branching information => i.e. MonteCarlo data
              pass                                    
            try:
              attrs["Probability_threshold"].append(gb_attrs[key]["Probability_threshold"])
            except:
              # no branching information => i.e. MonteCarlo data
              pass                                                
            try:
              attrs["end_timestep"].append(gb_attrs[key]["end_timestep"])
            except:
              # no branching information => i.e. MonteCarlo data
              pass                                                            
        else:
          ''' 
            A number of groups' back have been inputted
            Follow the same strategy used above (filter = whole)
            but stop at back(th) group starting from group "name"
          '''
          if is_number(filter):
            back = int(filter) + 1
            if len(list_path) < back:
              raise Exception("Error. Number of branches back > number of actual branches in dataset for History ending with " + name)
            # start constructing the merged numpy array
            where_list = []
            name_list  = []
            i=0
            try:
              list_path.remove("")
            except:
              pass
            '''
            Find the paths for the completed history
            '''
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
            ''' get the relative groups' data '''
            gb_res ={}
            gb_attrs={}
            where_list.reverse()
            name_list.reverse()
            n_tot_ts = 0
            n_params = 0
            '''
              Retrieve every single partial history that will be merged to create the whole history
            '''
            for i in xrange(len(where_list)):
              grp = self.h5_file_w.require_group(where_list[i])
              namel = name_list[i] +'_data'
              dataset = grp.require_dataset(namel, (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float')
              if i == 0:
                n_params = int(grp.attrs['n_params'])
              
              if n_params != int(grp.attrs['n_params']):
                raise Exception("Can not merge datasets with different number of parameters")
              # get numpy array
              gb_res[i]   = dataset[:,:]
              gb_attrs[i] =grp.attrs        
              n_tot_ts = n_tot_ts + int(grp.attrs['n_ts'])
            ''' Create numpy array '''
            result = np.zeros((n_tot_ts,n_params))
            ts = 0
            ''' Retrieve metadata '''
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
            except:
              pass
            try:
              par2 = gb_attrs[0]["branch_changed_param_value"]
              attrs["branch_changed_param_value"]    = []
            except:
              pass
            try:
              par3 = gb_attrs[0]["initiator_distribution"]
              attrs["initiator_distribution"]    = []
            except:
              pass
            try:
              par4 = gb_attrs[0]["Probability_threshold"]
              attrs["Probability_threshold"]    = []
            except:
              pass
            try:
              par5 = gb_attrs[0]["end_timestep"]
              attrs["end_timestep"]    = []
            except:
              pass
            
            for key in gb_res:
              try:
                attrs["input_file"].append(gb_attrs[key]["input_file"])
              except:
                pass
              if attrs["source_type"] == 'csv':
                attrs["source_file"].append(gb_attrs[key]["source_file"])
              try:
                ''' parameter that caused the branching '''
                attrs["branch_changed_param"].append(gb_attrs[key]["branch_changed_param"])
              except:
                ''' no branching information => i.e. MonteCarlo data '''
                pass
              try:
                attrs["conditional_prb"].append(gb_attrs[key]["conditional_prb"])
              except:
                ''' no branching information => i.e. MonteCarlo data '''
                pass            
              try:
                attrs["branch_changed_param_value"].append(gb_attrs[key]["branch_changed_param_value"])
              except:
                ''' no branching information => i.e. MonteCarlo data '''
                pass                        
              try:
                attrs["initiator_distribution"].append(gb_attrs[key]["initiator_distribution"])
              except:
                ''' no branching information => i.e. MonteCarlo data '''
                pass                                    
              try:
                attrs["Probability_threshold"].append(gb_attrs[key]["Probability_threshold"])
              except:
                ''' no branching information => i.e. MonteCarlo data '''
                pass                                                
              try:
                attrs["end_timestep"].append(gb_attrs[key]["end_timestep"])
              except:
                ''' no branching information => i.e. MonteCarlo data '''
                pass                                                                          
          else:
            ''' ERR '''
            raise Exception("Error. Filter not recognized in hdf5Database.retrieveHistory function. Filter = " + str(filter)) 
      else:
        raise Exception("History named " + name + "not found in database")
      return(result,attrs)

    '''
      Function to close the database
      @ In,  None
      @ Out, None
    '''
    def closeDataBaseW(self):
      self.h5_file_w.close()
      self.fileOpen       = False
      return

    '''
      Function to open the database
      @ In,  filename : name of the file (string)
      @ In,  mode     : open mode (default "w=write")
      @ Out, fh5      : hdf5 object
    '''
    def openDataBaseW(self,filename,mode='w'):
      fh5 = h5.File(filename,mode)
      self.fileOpen       = True
      return fh5
  
def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False      
      
