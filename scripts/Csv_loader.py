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
      for key, value in inParamDict.items():
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
              for key, value in outParamDict.items():
                try:
                  actual_value = float(row[key])
                  previous_value =  float(self.previous_dict[key])
                  outParamDict[key] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)
                except:
                  raise("ERROR: the parameter "+ key +" has not been found")
              break
            else:
              outParamDict = self.actual_dict        
              break
      #if time == end => we have to get the last row => it's in the actual_dict
      if time_end:
        if(not self.all_out_param):
          for key, value in outParamDict.items():
            try:
              outParamDict[key] = float(self.actual_dict[key])
            except:
              raise("ERROR: the parameter "+ key +" has not been found")
        else:
          outParamDict = self.actual_dict
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
    
    for i in range(len(filesin)): 
      #we now load the csv file
      with open(filesin[i],'rb') as f:
        reader = csv.DictReader(f)
        reader.next
        if(self.all_out_param):
            self.field_names = reader.fieldnames
        else:
            self.field_names = outParamDict.keys()
            self.field_names.insert(0, 'time')
      
        row = reader.next()
        # Put the input params values into the inParamDict
        for key, value in inParamDict.items():
          try:
            inParamDict[key].append(float(row[key]))
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
                    outParamDict[key].append(float((actual_value-previous_value)/(actual_time-previous_time)*
                                                   (time_float-previous_time)))
                  except:
                    raise("ERROR: the parameter "+ key +" has not been found")
                break
              else:
                for key in self.actual_dict.keys():
                  outParamDict[key].append(self.actual_dict[key])        
                break
        #if time == end => we have to get the last row => it's in the actual_dict
        if time_end:
          if(not self.all_out_param):
            for key, value in outParamDict.items():
              try:
                outParamDict[key].append(float(self.actual_dict[key]))
              except:
                raise("ERROR: the parameter "+ key +" has not been found")
          else:
            for key in self.actual_dict.keys():
              outParamDict[key].append(float(self.actual_dict[key]))   
    return


