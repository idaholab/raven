'''
Created on April 4, 2013

@author: alfoa
'''
import numpy as np
from h5py_interface_creator import hdf5Database as h5Data

class hdf5Manager:
    '''
    class to manage a or multiple h5py (hdf5) database/s,
    to build them and to retrieve attributes and values in them
    '''
    def __init__(self):
      '''
      Constructor
      '''
      # container of data bases object
      self.h5DataBases  = {}
      # number of data based actually available
      self.n_databases  = 0
      # map<string,(bool)> ie. database name, built or not
      self.areTheyBuilt = {}
      # map<string,(bool)> ie. database name, database already exists???
      self.doTheyAlreadyExist  = {}

    def addDataBase(self,name,type,exist=False):
      self.h5DataBases[name] = h5Data(name,type,exist)
      self.n_databases = self.n_databases + 1
      self.doTheyAlreadyExist[name] = exist
      self.areTheyBuilt[name] = False

    def removeDataBase(self,name):
      try:
        del self.h5DataBases[name]
        del self.doTheyAlreadyExist[name]
        del self.areTheyBuilt[name]
        self.n_databases = self.n_databases - 1
      except:
        pass
    
    def addGroup(self,name,attributes,loadFrom):
      if name in self.h5DataBases.keys():
        self.h5DataBases[name].addGroup(attributes["group"],attributes,loadFrom)
        self.areTheyBuilt[name] = True
      else:
         raise("data set named " + name + "not found")  
      return
    
    def __returnHistory(self,attributes):
      if(attributes["d_name"] in self.h5DataBases.keys()):
        if (not self.doTheyAlreadyExist[attributes["d_name"]]) and (not self.areTheyBuilt[attributes["d_name"]]):
          raise("ERROR: Can not retrieve an History from" + attributes["d_name"] + ".It has not built yet.")
        if attributes['filter']:
          tupleVar = self.h5DataBases[attributes["d_name"]].retrieveHistory(attributes["history"],attributes['filter'])
        else:
          tupleVar = self.h5DataBases[attributes["d_name"]].retrieveHistory(attributes["history"])          
      else:
        raise("data set named " + attributes["d_name"] + "not found")
      return tupleVar
    
    def __retrieveDataTimePoint(self,attributes):
      histVar = self.__returnHistory(attributes)

      if attributes['outParam'] == 'all':
        all_out_param  = True
      else:
        all_out_param = False
    
      if attributes['time'] == 'end':
        time_end = True
        time_float = -1.0
      else:
        # convert the time in float
        time_end = False
        time_float = float(attributes['time'])
            
      inDict  = {}
      outDict = {} 
      
      field_names = []
      #all_field_names = []
      
      if(all_out_param):
        field_names = histVar[1]["headers"]
        #all_field_names = field_names
      else:
        field_names = attributes['outParam']
        field_names.insert(0, 'time') 
        #all_field_names = histVar[1]["headers"]
    
      #fill input param dictionary
      for key in attributes['inParam']:
          if key in histVar[1]["headers"]:
            ix = histVar[1]["headers"].index(key)
            inDict[key] = histVar[0][0,ix]
          else:
            raise("ERROR: the parameter " + key + " has not been found")
    
    # fill output param dictionary
    
    # time end case
      if time_end:
        last_row = histVar[0][:,0].size - 1
        if all_out_param:
          for key in histVar[1]["headers"]:
            outDict[key] = histVar[0][last_row,histVar[1]["headers"].index(key)]
        else:
          for key in attributes['outParam']:
            if key in histVar[1]["headers"]:
              outDict[key] = histVar[0][last_row,histVar[1]["headers"].index(key)]        
            else:
              raise("ERROR: the parameter " + key + " has not been found")
      else:
      
        for i in histVar[0]:
          if histVar[0][i,0] >= time_float and time_float >= 0.0:
            try:
              previous_time = histVar[0][i-1,0]
              actual_time   = histVar[0][i,0]
            except:
              previous_time = histVar[0][i,0]
              actual_time   = histVar[0][i,0]          
            if all_out_param:
              for key in histVar[1]["headers"]:
                if(actual_time == previous_time):
                  outDict[key] = (histVar[0][i,histVar[1]["headers"].index(key)]  - time_float) / actual_time 
                else:
                  actual_value   = histVar[0][i,histVar[1]["headers"].index(key)]
                  previous_value = histVar[0][i-1,histVar[1]["headers"].index(key)] 
                  outDict[key] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
            else:
              for key in attributes['outParam']:
                if key in histVar[1]["headers"]:
                  if(actual_time == previous_time):
                    outDict[key] = (histVar[0][i,histVar[1]["headers"].index(key)]  - time_float) / actual_time 
                  else:
                    actual_value   = histVar[0][i,histVar[1]["headers"].index(key)]
                    previous_value = histVar[0][i-1,histVar[1]["headers"].index(key)] 
                    outDict[key] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
                           
                else:
                  raise("ERROR: the parameter " + key + " has not been found")      
      return (inDict,outDict)

    def __retrieveDataTimePointSet(self,attributes):

      if attributes['outParam'] == 'all':
        all_out_param  = True
      else:
        all_out_param = False
    
      if attributes['time'] == 'end':
        time_end = True
        time_float = -1.0
      else:
        # convert the time in float
        time_end = False
        time_float = float(attributes['time'])
        
      inDict  = {}
      outDict = {}    
      hist_list = []
      hist_list = attributes['histories']
      
      for i in range(len(hist_list)): 
        #load the data into the numpy array
        attributes['history'] = hist_list[i]
        histVar = self.__returnHistory(attributes)

        if i == 0:
          if(all_out_param):
            field_names = histVar[1]["headers"]
          else:
            field_names = attributes['outParam']
            field_names.insert(0, 'time') 
        
        #fill input param dictionary
        for key in attributes['inParam']:
          if key in histVar[1]["headers"]:
            ix = histVar[1]["headers"].index(key)
            if i == 0:
              #create numpy array
              inDict[key] = np.zeros(len(hist_list))
              
            inDict[key][i] = histVar[0][0,ix]
            #inDict[key][i] = 1
          else:
            raise("ERROR: the parameter " + str(key) + " has not been found")
        # time end case
        if time_end:
          last_row = histVar[1][:,0].size - 1
          if all_out_param:
            for key in histVar[1]["headers"]:
              if i == 0:
                #create numpy array
                outDict[key] = np.zeros(len(hist_list))  
              
              outDict[key][i] = histVar[0][last_row,histVar[1]["headers"].index(key)]
          else:
            for key in attributes['outParam']:
              if key in histVar[1]["headers"]:
                if i == 0:
                  #create numpy array
                  outDict[key] = np.zeros(len(hist_list))
                outDict[key][i] = histVar[0][last_row,histVar[1]["headers"].index(key)]
              else:
                raise("ERROR: the parameter " + str(key) + " has not been found")
        else:
          
          for i in histVar[0]:
            if histVar[0][i,0] >= time_float and time_float >= 0.0:
              try:
                previous_time = histVar[0][i-1,0]
                actual_time   = histVar[0][i,0]
              except:
                previous_time = histVar[0][i,0]
                actual_time   = histVar[0][i,0]          
              if all_out_param:
                for key in histVar[1]["headers"]:
                  if(actual_time == previous_time):
                    if i == 0:
                      #create numpy array
                      outDict[key] = np.zeros(np.shape(len(hist_list)))           
                                
                    outDict[key][i] = (histVar[0][i,histVar[1]["headers"].index(key)]  - time_float) / actual_time 
                  else:
                    if i == 0:
                      #create numpy array
                      outDict[key] = np.zeros(np.shape(len(hist_list))) 
                                      
                    actual_value   = histVar[0][i,histVar[1]["headers"].index(key)]
                    previous_value = histVar[0][i-1,histVar[1]["headers"].index(key)] 
                    outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
              else:
                for key in attributes['outParam']:
                  if key in histVar[1]["headers"]:
                    if(actual_time == previous_time):
                      if i == 0:
                        #create numpy array
                        outDict[key] = np.zeros(np.shape(len(hist_list))) 
                                              
                      outDict[key][i] = (histVar[0][i,histVar[1]["headers"].index(key)]  - time_float) / actual_time 
                    else:
                      if i == 0:
                        #create numpy array
                        outDict[key] = np.zeros(np.shape(len(hist_list)))
                      
                      actual_value   = histVar[0][i,histVar[1]["headers"].index(key)]
                      previous_value = histVar[0][i-1,histVar[1]["headers"].index(key)] 
                      outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
                  else:
                    raise("ERROR: the parameter " + key + " has not been found")      
        del histVar       
      return (inDict,outDict)
    
    def __retrieveDataHistory(self,attributes):
      
      time_float = []
      if attributes['outParam'] == 'all':
        all_out_param  = True
      else:
        all_out_param = False
      if attributes['time']:
        if attributes['time'] == 'all':
          time_all = True
        else:
          # convert the time in float
          time_all = False
          time_float = [float(x) for x in attributes['time']]
      else:
        time_all = True
                     
      inDict  = {}
      outDict = {}  
   
      #load the data into the tuple
      histVar = self.__returnHistory(attributes)
      
      if(all_out_param):
        field_names = histVar[1]["headers"]
      else:
        field_names = attributes["outParam"]
        field_names.insert(0, 'time') 
      
      #fill input param dictionary
      for key in attributes["inParam"]:
          if key in histVar[1]["headers"]:
            ix = histVar[1]["headers"].index(key)
            inDict[key] = histVar[0][0,ix]
          else:
            raise("ERROR: the parameter " + key + " has not been found")
      
      # time all case
      if time_all:
        if all_out_param:
          for key in histVar[1]["headers"]:
            outDict[key] = histVar[0][:,histVar[1]["headers"].index(key)]
        else:
          for key in attributes["outParam"]:
            if key in histVar[1]["headers"]:
              outDict[key] = histVar[0][:,histVar[1]["headers"].index(key)]        
            else:
              raise("ERROR: the parameter " + key + " has not been found")
      else:
        # it will be implemented when we decide a strategy about time filtering 
        ## for now it is a copy paste of the time_all case
        if all_out_param:
          for key in histVar[1]["headers"]:
            outDict[key] = histVar[0][:,histVar[1]["headers"].index(key)]
        else:
          for key in attributes["outParam"]:
            if key in histVar[1]["headers"]:
              outDict[key] = histVar[0][:,histVar[1]["headers"].index(key)]        
            else:
              raise("ERROR: the parameter " + key + " has not been found")
      return (inDict,outDict)

    def retrieveData(self,attributes):
      #time,inParam,outParam)
      if attributes["type"] == "TimePoint":
        data = self.__retrieveDataTimePoint(attributes)
      elif attributes["type"] == "TimePointSet":
        data = self.__retrieveDataTimePointSet(attributes)
      elif attributes["type"] == "History":
        data = self.__retrieveDataHistory(attributes)
#      elif attributes["type"] == "Histories":
#        data = self.__retrieveDataHistories(attributes)
      else:
        raise("Type" + attributes["type"] +" unknown.Caller: hdf5Manager.retrieveData") 
      return data
      
      
       