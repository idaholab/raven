'''
Created on Feb 7, 2013
@author: alfoa
This python module performs the loading of 
data from csv files
'''
import os
import linecache
import csv 
import numpay as np

class CsvLoader:
  def __init__(self):
    '''
    Constructor
    '''
    self.all_out_param      = False # all output parameters?
    self.previous_list      = {}    # working dict
    self.actual_list        = {}    # working dict
    self.field_names        = []
    
  def csvLoaderForTimePoint(self,filein,time,inParamDict,outParamDict):
    '''
    filein = file name
    time   = time
    paramList = parameters to be picked up (optional)
    '''
  
    if outParamDict['all']:
      self.all_out_param  = True
      outParamDict['all'] = None
    else:
      self.all_out_param = False
    
    if time == 'end':
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = [float(x) for x in time]
      
    inDict  = inParamDict
    outDict = outParamDict       
    
    #we now load the csv file
    with open(filein,'rb') as f:
      reader = csv.DictReader(f)
      reader.next
      if(self.all_out_param):
          self.field_names = reader.fieldnames
      else:
          self.field_names = outDict.keys()
          self.field_names.insert(0, 'time')
      
      row = reader.next()
      # Put the input params values into the inParamDict
      for key, value in inDict.items():
        try:
          value = float(row[key])
        except:
          print("ERROR: the parameter " + key + " has not been found")
      #loop over rows    
      for row in reader:
          self.previous_dict = self.actual_dict
          self.actual_dict   = row  
          if(float(row['time']) >= time_float and time_float >= 0.0):
            # we have to interpolate from values in previous_list
            # and actual_list
            previous_time = float(previous_dict['time'])
            actual_time   = float(actual_dict['time'])
            # This loop add interpolated values into the outParamDict
            if(not self.all_out_param):
              for key, value in outDict.items():
                try:
                  actual_value = float(row[key])
                  previous_value =  float(self.previous_dict[key])
                  outDict[key] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)
                except:
                  raise("ERROR: the parameter "+ key +" has not been found")
              break
            else:
              outDict = self.actual_dict        
              break
      #if time == end => we have to get the last row => it's in the actual_dict
      if time_end:
        if(not self.all_out_param):
          for key, value in outDict.items():
            try:
              outDict[key] = float(self.actual_dict[key])
            except:
              raise("ERROR: the parameter "+ key +" has not been found")
        else:
          outDict = self.actual_dict
          
    for key in outDict:
      outParamDict[key] = np.array(outDict[key], dtype=float)
    for key in inDict:
      inParamDict[key] = np.array(inDict[key], dtype=float)           
    return
  def csvLoaderForTimePointSet(self,filesin,time,inParamDict,outParamDict):
    '''
    filesin = file names
    time   = time
    inParamDict = parameters to be picked up 
    outParamDict = parameters to be picked up    
    '''
    if outParamDict['all']:
      self.all_out_param  = True
      outParamDict['all'] = None
    else:
      self.all_out_param = False
    
    if time == 'end':
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = [float(x) for x in time]
      
    inDict  = inParamDict
    outDict = outParamDict    
    
    for i in range(len(filesin)): 
      #we now load the csv file
      with open(filesin[i],'rb') as f:
        reader = csv.DictReader(f)
        reader.next
        if(self.all_out_param):
            self.field_names = reader.fieldnames
        else:
            self.field_names = outDict.keys()
            self.field_names.insert(0, 'time')
      
        row = reader.next()
        # Put the input params values into the inParamDict
        for key, value in inDict.items():
          try:
            inDict[key].append(float(row[key]))
          except:
            print("ERROR: the parameter " + key + " has not been found")
        #loop over rows    
        for row in reader:
            self.previous_dict = self.actual_dict
            self.actual_dict   = row  
            if(float(row['time']) >= time_float and time_float >= 0.0):
              # we have to interpolate from values in previous_list
              # and actual_list
              previous_time = float(previous_dict['time'])
              actual_time   = float(actual_dict['time'])
              # This loop add interpolated values into the outParamDict
              if(not self.all_out_param):
                for key, value in outParamDict.items():
                  try:
                    actual_value = float(row[key])
                    previous_value =  float(self.previous_dict[key])
                    outDict[key].append(float((actual_value-previous_value)/(actual_time-previous_time)*
                                                   (time_float-previous_time)))
                  except:
                    raise("ERROR: the parameter "+ key +" has not been found")
                break
              else:
                for key in self.actual_dict.keys():
                  outDict[key].append(self.actual_dict[key])        
                break
        #if time == end => we have to get the last row => it's in the actual_dict
        if time_end:
          if(not self.all_out_param):
            for key, value in outDict.items():
              try:
                outDict[key].append(float(self.actual_dict[key]))
              except:
                raise("ERROR: the parameter "+ key +" has not been found")
          else:
            for key in self.actual_dict.keys():
              outDict[key].append(float(self.actual_dict[key]))   
    #convert the lists in working dicts into numpy arrays
    for key in outDict:
      outParamDict[key] = np.array(outDict[key], dtype=float)
    for key in inDict:
      inParamDict[key] = np.array(inDict[key], dtype=float)    
    return
  def csvLoaderForHistory(self,filein,time_filter=None,inParamDict,outParamDict):
    '''
    filein = file name
    time_filter   = time_filter
    inParamDict = parameters to be picked up (dictionary of values)
    outParamDict = parameters to be picked up (dictionary of lists)   
    '''
    inDict  = inParamDict
    outDict = outParamDict
    
    if(inDict['all']):
      inDict['all'] = None
    else:
      inputParamN = len(inDict.keys())
    if(outDict['all']):
      self.all_out_param  = True
      outParamDict['all'] = None            
    else:
      self.all_out_param  = False

    if time_filter:
      time_all = False
      time_float = [float(x) for x in time_filter]
    else:
      # WE HAVE TO TAKE A DECISION REGARDING THE FILTERING
      time_all = True
      time_float = -1.0
   
    #we now load the csv file
    with open(filein,'rb') as f:
      reader = csv.DictReader(f)
      reader.next
      if(self.all_out_param):
        self.field_names = reader.fieldnames
      else:
        self.field_names = outParamDict.keys()
        self.field_names.insert(0, 'time')
      
      row = reader.next()
      # Put the input params values into the inParamDict
      for key, value in inDict.items():
        try:
          inDict[key]=(float(row[key]))
        except:
          print("ERROR: the parameter " + key + " has not been found")
      #loop over rows    
      for row in reader:
          self.previous_dict = self.actual_dict
          self.actual_dict   = row  
          if(not self.all_out_param):
            for key, value in outDict.items():
              try:
                outDict[key].append(float(row[key]))
              except:
                raise("ERROR: the parameter "+ key +" has not been found")
            break
          else:
            for key in row.keys():
              outDict[key].append(float(row[key]))        
            break
    #convert the lists in working dicts into numpy arrays
    for key in outDict:
      outParamDict[key] = np.array(outDict[key], dtype=float)
    for key in inDict:
      inParamDict[key] = np.array(inDict[key], dtype=float)        
    return          
  def csvLoaderForHistories(self,filesin,time_filters=None,inParamDict,outParamDict):
    '''
    filein = file name
    time_filter   = time_filter
    inParamDict = parameters to be picked up (dictionary of dictionaries of lists))
    outParamDict = parameters to be picked up (dictionary of dictionaries of lists)   
    '''
    inDict  = {}
    outDict = {}
    # firstly we read the first file to grep information regarding the number of time-step
    self.parseFilesToGrepDimensions(filesin, NtimeSteps, maxNumOfParams, NSamples)

    inDict  = inParamDict
    outDict = outParamDict
    if(inDict['all']):
      inputParamN = maxNumOfParams
    else:
      inputParamN = len(inDict.keys())
    if(outDict['all']):
      outputParamN = maxNumOfParams
    else:
      outputParamN = len(outDict.keys())  
    # now we "allocate" the numpy matrix
    data_work_in = np.zeros(shape=(inputParamN,NtimeSteps,NSamples))
    data_work_out = np.zeros(shape=(outputParamN,NtimeSteps,NSamples))
    
    for i in range(NSamples):
      self.csvLoaderForHistory(filein[i],time_filter,inDict,outDict)
      index = 0
      for key in inDict:
        data_work_in[index,:,i] = float(inDict[key])
        index = index + 1
      index = 0  
      for key in outDict:
        data_work_out[index,:,i] = float(outDict[key])
        index = index + 1    
    #construct the output dictionaries
    key_index = 0
    for key in inDict:
      if(str(key) != 'all'):
        inParamDict[key] = data_work_in[key_index,:,:]
        key_index = key_index + 1
    key_index = 0    
    for key in outDict:
      if(str(key) != 'all'):
        outParamDict[key] = data_work_out[key_index,:,:]
        key_index = key_index + 1      
    return

  def parseFilesToGrepDimensions(self,filesin,NtimeSteps,maxNumOfParams,NSamples):      
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
        reader.next
        if(len(reader.fieldnames) > maxNumOfParams):
          maxNumOfParams = len(reader.fieldnames)
        
        countTimeSteps = 1  
        row = reader.next()
        for row in reader:
          countTimeSteps = countTimeSteps + 1   
        
        if(countTimeSteps>NtimeSteps):
          NtimeSteps = countTimeSteps
    return      

      
      
      
    