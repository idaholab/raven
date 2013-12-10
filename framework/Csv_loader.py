'''
Created on Feb 7, 2013
@author: alfoa
This python module performs the loading of 
data from csv files
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import numpy as np
import csv

class CsvLoader:
  def __init__(self):
    '''
    Constructor
    '''
    self.all_out_param      = False # all output parameters?
    self.field_names        = []    # 
    self.all_field_names    = []
    
  def loadCsvFile(self,filein):
    # open file
    myFile = open (filein,'rb')
    # read the field names
    head = myFile.readline().decode()
    self.all_field_names = head.split(',')
    for index in range(len(self.all_field_names)): self.all_field_names[index] = self.all_field_names[index].replace("\n", "")
    # load the table data (from the csv file) into a numpy nd array 
    data = np.loadtxt(myFile,dtype='float',delimiter=',',ndmin=2)
    # close file
    myFile.close()  
    return data

  # function to get actual field names (desired output parameter keywords)
  def getFieldNames(self):
    return self.field_names

  # function to get all field names found in the csv file
  def getAllFieldNames(self):
    return self.all_field_names

  # function to grep max dimensions in multiple csv files
  def parseFilesToGrepDimensions(self,filesin):      
    '''
    filesin = file names
    NtimeSteps   = maxNumberOfTs
    maxNumOfParams = max number of parameters
    NSamples = number of Samples   
    '''    
    NSamples       = len(filesin)
    maxNumOfParams = 0
    NtimeSteps     = 0  
    for i in range(filesin):    
      with open(filesin[i],'rb') as f:
        reader = csv.DictReader(f)
        #reader.next #XXX This line does nothing
        if(len(reader.fieldnames) > maxNumOfParams):
          maxNumOfParams = len(reader.fieldnames)
        
        countTimeSteps = 1  
        row = next(reader)
        for row in reader:
          countTimeSteps = countTimeSteps + 1   
        
        if(countTimeSteps>NtimeSteps):
          NtimeSteps = countTimeSteps
    return (NtimeSteps,maxNumOfParams,NSamples)  
 
  def csvLoadData(self,filein,options):
    if   options['type'] == 'TimePoint':
      return self.__csvLoaderForTimePoint(filein[0],options['time'],options['inParam'],options['outParam'],options['input_ts'])
    elif options['type'] == 'TimePointSet':
      return self.__csvLoaderForTimePointSet(filein,options['time'],options['inParam'],options['outParam'],options['input_ts'])
    elif options['type'] == 'History':
      return self.__csvLoaderForHistory(filein[0],options['time'],options['inParam'],options['outParam'],options['input_ts'])
    elif options['type'] == 'Histories':
      listhist_in  = {}
      listhist_out = {}
      for index in xrange(len(filein)):
        tupleVar = self.__csvLoaderForHistory(filein[index],options['time'],options['inParam'],options['outParam'],options['input_ts'])
        # dictionary of dictionary key = i => ith history ParameterValues dictionary
        listhist_in[index]  = tupleVar[0]
        listhist_out[index] = tupleVar[1]
        del tupleVar
      return(listhist_in,listhist_out)
    else:
      raise IOError ('CSV LOADER : ******ERROR Type ' + options['type'] + 'unknown')
    
  # loader for time point data type
  def __csvLoaderForTimePoint(self,filein,time,inParam,outParam,input_ts):
    '''
    filein = file name
    time   = time
    paramList = parameters to be picked up (optional)
    '''
    #load the data into the numpy array
    data = self.loadCsvFile(filein)
    
    if 'all' in outParam:
      self.all_out_param  = True
    else:
      self.all_out_param = False
    
    if (time == 'end') or (not time):
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = float(time)
    if input_ts: ints = int(input_ts)
    else: ints = 0
    if ints > data[:,0].size : raise IOError('CSV LOADER : ******ERROR input_ts is greater than number of actual ts in file '+ str(filein) + '!')
       
    #inDict  = inParamDict
    #outDict = outParamDict       
    inDict  = {}
    outDict = {} 
    
    if(self.all_out_param):
      self.field_names = self.all_field_names
    else:
      self.field_names = outParam
      self.field_names.insert(0, 'time') 
    
    #fill input param dictionary
    for key in inParam:
        if key in self.all_field_names:
          ix = self.all_field_names.index(key)
          inDict[key] = np.atleast_1d(np.array(data[input_ts,ix]))
        else:
          raise Exception("ERROR: the parameter " + key + " has not been found")
    
    # fill output param dictionary
    
    # time end case
    if time_end:
      last_row = data[:,0].size - 1
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = np.atleast_1d(np.array(data[last_row,self.all_field_names.index(key)]))
      else:
        for key in outParam:
          if key in self.all_field_names:
            outDict[key] = np.atleast_1d(np.array(data[last_row,self.all_field_names.index(key)]))       
          else:
            raise Exception("ERROR: the parameter " + key + " has not been found")
    else:
      
      for i in data:
        if data[i,0] >= time_float and time_float >= 0.0:
          if i-1 >= 0:
            previous_time = data[i-1,0]
          else:
            previous_time = data[i,0]
          actual_time   = data[i,0]          
          if self.all_out_param:
            for key in self.all_field_names:
              if(actual_time == previous_time):
                outDict[key] = np.atleast_1d(np.array((data[i,self.all_field_names.index(key)]  - time_float) / actual_time)) 
              else:
                actual_value   = data[i,self.all_field_names.index(key)]
                previous_value = data[i-1,self.all_field_names.index(key)] 
                outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))    
          else:
            for key in outParam:
              if key in self.all_field_names:
                if(actual_time == previous_time):
                  outDict[key] = np.atleast_1d(np.array((data[i,self.all_field_names.index(key)]  - time_float) / actual_time)) 
                else:
                  actual_value   = data[i,self.all_field_names.index(key)]
                  previous_value = data[i-1,self.all_field_names.index(key)] 
                  outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))    
                         
              else:
                raise Exception("ERROR: the parameter " + key + " has not been found")
    return (inDict,outDict)

  def __csvLoaderForTimePointSet(self,filesin,time,inParam,outParam,input_ts):
    '''
    loader for time point set data type
    filesin = file names
    time   = time
    inParam = parameters to be picked up 
    outParam = parameters to be picked up    
    '''
    if 'all' in outParam:
      self.all_out_param  = True
    else:
      self.all_out_param = False
    
    if (time == 'end') or (not time):
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = float(time)
    if input_ts: ints = int(input_ts)
    else: ints = 0
    
          
    inDict  = {}
    outDict = {}    
    
    for i in range(len(filesin)): 
      #load the data into the numpy array
      data = self.loadCsvFile(filesin[i])
      if ints > data[:,0].size : raise IOError('CSV LOADER : ******ERROR input_ts is greater than number of actual ts in file '+ str(filesin[i]) + '!') 
      if i == 0:
        if(self.all_out_param):
          self.field_names = self.all_field_names
        else:
          self.field_names = outParam
          self.field_names.insert(0, 'time')       
      #fill input param dictionary
      for key in inParam:
        print(self.all_field_names)
        if key in self.all_field_names:
          
          ix = self.all_field_names.index(key)
          if i == 0:
            #create numpy array
            inDict[key] = np.zeros(len(filesin))
            
          inDict[key][i] = data[ints,ix]
          #inDict[key][i] = 1
        else:
          raise Exception("ERROR: the parameter " + str(key) + " has not been found")
      # time end case
      if time_end:
        last_row = data[:,0].size - 1
        if self.all_out_param:
          for key in self.all_field_names:
            if i == 0:
              #create numpy array
              outDict[key] = np.zeros(len(filesin))  
            
            outDict[key][i] = data[last_row,self.all_field_names.index(key)]
        else:
          for key in outParam:
            if key in self.all_field_names:
              if i == 0:
                #create numpy array
                outDict[key] = np.zeros(len(filesin))
              outDict[key][i] = data[last_row,self.all_field_names.index(key)]
            else:
              raise Exception("ERROR: the parameter " + str(key) + " has not been found")
      else:
        
        for i in data:
          if data[i,0] >= time_float and time_float >= 0.0:
            if i-1 >= 0:
              previous_time = data[i-1,0]
            else:
              previous_time = data[i,0]
            actual_time   = data[i,0]          
            if self.all_out_param:
              for key in self.all_field_names:
                if(actual_time == previous_time):
                  if i == 0:
                    #create numpy array
                    outDict[key] = np.zeros(np.shape(len(filesin)))           
                              
                  outDict[key][i] = (data[i,self.all_field_names.index(key)]  - time_float) / actual_time 
                else:
                  if i == 0:
                    #create numpy array
                    outDict[key] = np.zeros(np.shape(len(filesin))) 
                                    
                  actual_value   = data[i,self.all_field_names.index(key)]
                  previous_value = data[i-1,self.all_field_names.index(key)] 
                  outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
            else:
              for key in outParam:
                if key in self.all_field_names:
                  if(actual_time == previous_time):
                    if i == 0:
                      #create numpy array
                      outDict[key] = np.zeros(np.shape(len(filesin))) 
                                            
                    outDict[key][i] = (data[i,self.all_field_names.index(key)]  - time_float) / actual_time 
                  else:
                    if i == 0:
                      #create numpy array
                      outDict[key] = np.zeros(np.shape(len(filesin)))
                    
                    actual_value   = data[i,self.all_field_names.index(key)]
                    previous_value = data[i-1,self.all_field_names.index(key)] 
                    outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
                else:
                  raise Exception("ERROR: the parameter " + key + " has not been found")      
      del data 
    return (inDict,outDict)

  def __csvLoaderForHistory(self,filein,time,inParam,outParam,input_ts):
    '''
    loader for history data type
    filein = file name
    time_filter   = time_filter
    inParamDict = parameters to be picked up (dictionary of values)
    outParamDict = parameters to be picked up (dictionary of lists)   
    '''
    #load the data into the numpy array
    data = self.loadCsvFile(filein)
        
    time_float = []
    
    if 'all' in outParam:
      self.all_out_param  = True
    else:
      self.all_out_param = False
    
    if time:
      if 'all' in time:
        time_all = True
      else:
        time_all = False
        time_float = [float(x) for x in time]
    else:
      # WE HAVE TO TAKE A DECISION REGARDING THE FILTERING
      
      time_all = True
      #time_float[0] = -1.0
    if input_ts: ints = int(input_ts)
    else: ints = 0
    if ints > data[:,0].size : raise IOError('CSV LOADER : ******ERROR input_ts is greater than number of actual ts in file '+ str(filein) + '!')    
    inDict  = {}
    outDict = {}  
    
    if(self.all_out_param):
      self.field_names = self.all_field_names
    else:
      self.field_names = outParam
      self.field_names.insert(0, 'time') 
    
    #fill input param dictionary
    for key in inParam:
        if key in self.all_field_names:
          ix = self.all_field_names.index(key)
          inDict[key] = np.atleast_1d(np.array(data[ints,ix]))
        else:
          raise Exception("ERROR: the parameter " + key + " has not been found")
    
    # time all case
    if time_all:
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = data[:,self.all_field_names.index(key)]
      else:
        for key in outParam:
          if key in self.all_field_names:
            outDict[key] = data[:,self.all_field_names.index(key)]        
          else:
            raise Exception("ERROR: the parameter " + key + " has not been found")
    else:
      # it will be implemented when we decide a strategy about time filtering 
      ## for now it is a copy paste of the time_all case
      if self.all_out_param:
        for key in self.all_field_names:
          outDict[key] = data[:,self.all_field_names.index(key)]
      else:
        for key in outParam:
          if key in self.all_field_names:
            outDict[key] = data[:,self.all_field_names.index(key)]        
          else:
            raise Exception("ERROR: the parameter " + key + " has not been found")      
    return (inDict,outDict)         
#  def csvLoaderForHistories(self,filesin,time_filters=None,inParamDict,outParamDict):
#    '''
#    filein = file name
#    time_filter   = time_filter
#    inParamDict = parameters to be picked up (dictionary of dictionaries of lists))
#    outParamDict = parameters to be picked up (dictionary of dictionaries of lists)   
#    '''
#    inDict  = {}
#    outDict = {}
#    # firstly we read the first file to grep information regarding the number of time-step
#    self.parseFilesToGrepDimensions(filesin, NtimeSteps, maxNumOfParams, NSamples)
#
#    inDict  = inParamDict
#    outDict = outParamDict
#    if(inDict['all']):
#      inputParamN = maxNumOfParams
#    else:
#      inputParamN = len(inDict.keys())
#    if(outDict['all']):
#      outputParamN = maxNumOfParams
#    else:
#      outputParamN = len(outDict.keys())  
#    # now we "allocate" the numpy matrix
#    data_work_in = np.zeros(shape=(inputParamN,NtimeSteps,NSamples))
#    data_work_out = np.zeros(shape=(outputParamN,NtimeSteps,NSamples))
#    
#    for i in range(NSamples):
#      self.csvLoaderForHistory(filein[i],time_filter,inDict,outDict)
#      index = 0
#      for key in inDict:
#        data_work_in[index,:,i] = float(inDict[key])
#        index = index + 1
#      index = 0  
#      for key in outDict:
#        data_work_out[index,:,i] = float(outDict[key])
#        index = index + 1    
#    #construct the output dictionaries
#    key_index = 0
#    for key in inDict:
#      if(str(key) != 'all'):
#        inParamDict[key] = data_work_in[key_index,:,:]
#        key_index = key_index + 1
#    key_index = 0    
#    for key in outDict:
#      if(str(key) != 'all'):
#        outParamDict[key] = data_work_out[key_index,:,:]
#        key_index = key_index + 1      
#    return

