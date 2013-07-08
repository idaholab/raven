'''
Created on Mar 25, 2013

@author: alfoa
'''
import numpy as np
import h5py  as h5

class hdf5Database:
    '''
    class to create a h5py (hdf5) database
    '''
    def __init__(self,name, type, filename=None):
      '''
      Constructor
      '''
      # data base name (i.e. arbitrary name)
      self.name       = name                           
      # data base type (MC=MonteCarlo,DET=Dynamic Event Tree, etc.)
      self.type       = type
      # .h5 file name (to be created or read)
      if filename:
        self.onDiskFile = filename
        self.fileExist  = True
      else:
        self.onDiskFile = name + "_" + str(self.type) + ".h5" 
        self.fileExist  = False 
      # is the file opened? 
      self.fileOpen       = False
      # paths of all groups into the data base
      self.allGroupPaths = []
      # dict of boolean, true if the corresponding group in self.allGroupPaths
      # is an ending group, false if it is not
      self.allGroupEnds = {}      
      # we can create a base empty database or we open an existing one
      if self.fileExist:
        self.h5_file_w = self.openDataBaseW(self.onDiskFile,'r+')
        self.__createObjFromFile()
        self.firstRootGroup = True
      else:
        self.h5_file_w = self.openDataBaseW(self.onDiskFile,'w')
        # add the root as first group
        self.allGroupPaths.append("/") 
        self.allGroupEnds["/"] = False
        # the first root group has been added (DET)
        self.firstRootGroup = False
    
    def __createObjFromFile(self):
      self.allGroupPaths = []
      self.allGroupEnds  = {}
      if not self.fileOpen:
        self.h5_file_w = self.openDataBaseW(self.onDiskFile,'a')
      self.h5_file_w.visititems(self.__isGroup)
    
    def __isGroup(self,name,obj):
      if isinstance(obj,h5.Group):
        self.allGroupPaths.append(name)
        try:
          self.allGroupEnds[name]  = obj.attrs["EndGroup"]
        except:
          print('not found attribute EndGroup in group ' + name + '.Set True.')
          self.allGroupEnds[name]  = True
        
    def addGroup(self,gname,attributes,source):
      if self.type == 'DET':
        # TREE structure
        if not self.firstRootGroup:
          self.__addGroupRootLevel(gname,attributes,source)
          self.firstRootGroup = True
        else:
          self.__addSubGroup(gname,attributes,source)
      else:
        # ROOT structure
        self.__addGroupRootLevel(gname,attributes,source)
      return

    def __addGroupRootLevel(self,gname,attributes,source):
      for index in xrange(len(self.allGroupPaths)):
        comparisonName = self.allGroupPaths[index]
        if gname in comparisonName:
          raise IOError("Group named " + gname + " already present in database " + self.name)
      if source['type'] == 'csv':
        f = open(source['name'],'rb')
        # take the header of the CSV file
        firstRow = f.readline().translate(None,"\r\n")
        headers = firstRow.split(",")
        # load the csv into a numpy array(n time steps, n parameters)
        data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
        
        parent_group_name = "/"
        grp = self.h5_file_w.create_group(gname)
        
        print('Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
        
        # create data set in this new group
        dataset = grp.create_dataset(gname+"_data", dtype="float", data=data)
        
        grp.attrs["headers"]    = headers
        grp.attrs["n_params"]   = data[0,:].size
        grp.attrs["parent_id"]  = "root"
        grp.attrs["start_time"] = data[0,0]
        grp.attrs["end_time"]   = data[data[:,0].size-1,0]
        grp.attrs["n_ts"]       = data[:,0].size
        grp.attrs["EndGroup"]   = True
        try:
          grp.attrs["input_file"] = attributes["input_file"]
        except:
          pass
        grp.attrs["source_type"] = source['type']
            
        if source['type'] == 'csv':
          grp.attrs["source_file"] = source['name']

        try:
          # parameter that has been changed 
          grp.attrs["branch_changed_param"] = attributes["branch_changed_param"]
        except:
          # no branching information
          pass
        try:
          # parameter that caused the branching
          grp.attrs["branch_changed_param_value"] = attributes["branch_changed_param_value"]
        except:
          # no branching information
          pass        
        try:
          grp.attrs["conditional_prb"] = attributes["conditional_prb"]
        except:
          # no branching information => i.e. MonteCarlo data
          pass
        try:
          # initiator distribution
          grp.attrs["initiator_distribution"] = attributes["initiator_distribution"]
        except:
          # no branching information
          pass        
        try:
          # initiator distribution
          grp.attrs["Probability_threshold"] = attributes["PbThreshold"]
        except:
          # no branching information
          pass
        try:
          # initiator distribution
          grp.attrs["end_timestep"] = attributes["end_ts"]
        except:
          # no branching information
          pass        
      else:
        # do something else
        pass
      if parent_group_name != "/":
        self.allGroupPaths.append(parent_group_name + "/" + gname)
        self.allGroupEnds[parent_group_name + "/" + gname] = True
      else:
        self.allGroupPaths.append("/" + gname)
        self.allGroupEnds["/" + gname] = True

    def __addSubGroup(self,gname,attributes,source):
      if source['type'] == 'csv':
        f = open(source['name'],'rb')
        # take the header of the CSV file
        headers = f.readline().split(",")
        # load the csv into a numpy array(n time steps, n parameters)
        data = np.loadtxt(f,dtype='float',delimiter=',',ndmin=2)
        # check if the parent attribute is not null
        # in this case append a subgroup to the parent group
        # otherwise => it's the main group
        try:
          parent_name = attributes["parent_id"]
        except:
          raise IOError ("NOT FOUND attribute <parent_id> into <attributes> dictionary")
        # check if the parent exists... in that case... retrieve it and add the new sub group
#        test_list = []
#        test = self.h5_file_w.visititems(test_list.append(self.h5_file_w.get(self.h5_file_w)))
        # find parent group path
        if parent_name != '/':
          for index in xrange(len(self.allGroupPaths)):
            test_list = self.allGroupPaths[index].split('/')
            if test_list[len(test_list)-1] == parent_name:
              parent_group_name = self.allGroupPaths[index]
              break
        else:
          parent_group_name = parent_name   
        
        if parent_group_name in self.h5_file_w:
          grp = self.h5_file_w.require_group(parent_group_name)
        else:
          raise ValueError("NOT FOUND group named " + parent_group_name)  
        # create sub group
        self.allGroupEnds[parent_group_name] = False
        grp.attrs["EndGroup"]   = False
        
        print('Adding group named "' + gname + '" in DataBase "'+ self.name +'"')
        
        sgrp = grp.create_group(gname)
        # create data set in this new group
        dataset = sgrp.create_dataset(gname+"_data", dtype="float", data=data)
        
        sgrp.attrs["headers"]    = headers
        sgrp.attrs["n_params"]   = data[0,:].size
        sgrp.attrs["parent"]     = "root"
        sgrp.attrs["start_time"] = data[0,0]
        sgrp.attrs["end_time"]   = data[data[:,0].size-1,0]
        sgrp.attrs["n_ts"]       = data[:,0].size
        sgrp.attrs["EndGroup"]   = True
        
        try:
          sgrp.attrs["input_file"] = attributes["input_file"]
        except:
          pass
        sgrp.attrs["source_type"] = source['type']
        if source['type'] == 'csv':
          sgrp.attrs["source_file"] = source['name']

        try:
          # parameter that has been changed 
          sgrp.attrs["branch_changed_param"] = attributes["branch_changed_param"]
          testget = sgrp.attrs["branch_changed_param"]
        except:
          # no branching information
          pass
        try:
          # parameter that caused the branching
          sgrp.attrs["branch_changed_param_value"] = attributes["branch_changed_param_value"]
        except:
          # no branching information
          pass        
        try:
          sgrp.attrs["conditional_prb"] = attributes["conditional_prb"]
        except:
          # no branching information => i.e. MonteCarlo data
          pass
        try:
          # initiator distribution
          sgrp.attrs["initiator_distribution"] = attributes["initiator_distribution"]
        except:
          # no branching information
          pass        
        try:
          # initiator distribution
          sgrp.attrs["Probability_threshold"] = attributes["PbThreshold"]
        except:
          # no branching information
          pass
        try:
          # initiator distribution
          sgrp.attrs["end_timestep"] = attributes["end_ts"]
        except:
          # no branching information
          pass        
      else:
        # do something else
        pass
      if parent_group_name != "/":
        self.allGroupPaths.append(parent_group_name + "/" + gname)
        self.allGroupEnds[parent_group_name + "/" + gname] = True
      else:
        self.allGroupPaths.append("/" + gname) 
        self.allGroupEnds["/" + gname] = True
      
      return
    
    def computeBack(self,nameFrom,nameTo):
      list_str_w = []
      list_path  = []
      path       = ''
      found      = False
      
      for i in xrange(len(self.allGroupPaths)):
        list_str_w = self.allGroupPaths[i].split("/")
        if list_str_w[len(list_str_w)-1] == nameTo:
          found = True
          path  = self.allGroupPaths[i]
          list_path = list_str_w
          break      
      if not found:
        raise("ERROR: Group named " + nameTo + " not found in the HDF5 database" + self.onDiskFile)
      else:
        listGroups = path.split("/")
      
      fr = listGroups.index(nameFrom)
      to = listGroups.index(nameTo)
      
      back = to - fr
      
      return back
    
    def retrieveAllHistoryPaths(self):
      if not self.fileOpen:
        self.__createObjFromFile()

      if self.type == 'MC':
        return self.allGroupPaths
      else:
        workingList = []
        for index in xrange(len(self.allGroupPaths)):
         if self.allGroupEnds[self.allGroupPaths[index]] == True:
           workingList.append(self.allGroupPaths[index])
        return workingList
      
    def retrieveAllHistoryNames(self):
      if not self.fileOpen:
        self.__createObjFromFile()

      if self.type == 'MC':
        return self.allGroupPaths
      else:
        workingList = []
        for index in xrange(len(self.allGroupPaths)):
         if self.allGroupEnds[self.allGroupPaths[index]] == True:
           test_list = self.allGroupPaths[index].split('/')
           workingList.append(test_list[len(test_list)-1])
           del test_list
        return workingList
      
    def retrieveHistory(self,name,filter=None):
      # name => history name => It must correspond to a group name
      # filter => what must be retrieved: - 'whole' = whole history => all branches back from name to root
      #                                   - a number => from name to number branch back
      #                                   - 0 or None => only History "name"
      # return a numpy ndarray merging all the requested histories
      list_str_w = []
      list_path  = []
      path       = ''
      found      = False
      result     = None
      attrs = {}
      
      #check if the h5 file is already open, if not, open it
      if not self.fileOpen:
        self.__createObjFromFile()
      
      for i in xrange(len(self.allGroupPaths)):
        list_str_w = self.allGroupPaths[i].split("/")
        if list_str_w[len(list_str_w)-1] == name:
          found = True
          path  = self.allGroupPaths[i]
          list_path = list_str_w
          break
      
      if found:
        # check the filter
        if not filter or filter == 0:
          # grep only History from group "name"
          grp = self.h5_file_w.require_group(path)
          
          dataset = grp.require_dataset(name+"_data", (grp.attrs['n_ts'],grp.attrs['n_params']), dtype='float')          
#          dataset = grp.require_dataset(name=name +'_data', (grp.attrs('n_params'),grp.attrs('n_ts')), dtype='float', exact=True)
          # get numpy array
          result = dataset[:,:]
          
          attrs = grp.attrs
        elif  filter == 'whole':
          # start constructing the merged numpy array
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
          for i in xrange(len(where_list)):
            grp = self.h5_file_w.require_group(where_list[i])
            namel = name_list[i] +'_data'
            dataset = grp.require_dataset(namel, (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float')
            if i == 0:
              n_params = int(grp.attrs['n_params'])
            
            if n_params != int(grp.attrs['n_params']):
              raise("Can not merge datasets with different number of parameters")
            # get numpy array
            gb_res[i]   = dataset[:,:]
            gb_attrs[i] =grp.attrs        
            n_tot_ts = n_tot_ts + int(grp.attrs['n_ts'])
          
          result = np.zeros((n_tot_ts,n_params))
          ts = 0
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
            attrs["input_file"].append(gb_attrs[key]["input_file"])
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
          if is_number(filter):
            back = int(filter) + 1
            if len(list_path) < back:
              raise("Error. Number of branches back > number of actual branches in dataset for History ending with " + name)
            # start constructing the merged numpy array
            where_list = []
            name_list  = []
            i=0
            try:
              list_path.remove("")
            except:
              pass  
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
            for i in xrange(len(where_list)):
              grp = self.h5_file_w.require_group(where_list[i])
              namel = name_list[i] +'_data'
              dataset = grp.require_dataset(namel, (int(grp.attrs['n_ts']),int(grp.attrs['n_params'])), dtype='float')
              if i == 0:
                n_params = int(grp.attrs['n_params'])
              
              if n_params != int(grp.attrs['n_params']):
                raise("Can not merge datasets with different number of parameters")
              # get numpy array
              gb_res[i]   = dataset[:,:]
              gb_attrs[i] =grp.attrs        
              n_tot_ts = n_tot_ts + int(grp.attrs['n_ts'])
            
            result = np.zeros((n_tot_ts,n_params))
            ts = 0
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
              attrs["input_file"].append(gb_attrs[key]["input_file"])
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
            # ERR
            raise("Error. Filter not recognized in hdf5Database.retrieveHistory function. Filter = " + str(filter)) 
      else:
        raise("History named " + name + "not found in database")

      return(result,attrs)

    def printOutCsvHistory(self,what):
      pass


    def closeDataBaseW(self):
      self.h5_file_w.close()
      self.fileOpen       = False
      return
  
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
      