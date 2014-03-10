'''
Created on April 9, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
import xml.etree.ElementTree as ET
from BaseType import BaseType
from h5py_interface_creator import hdf5Database as h5Data
import copy
import os
import gc
from utils import toBytes

class DateBase(BaseType):
  '''
  class to handle database,
  to add and to retrieve attributes and values from it
  '''

  def __init__(self):
    '''
    Constructor
    '''
    # Base Class
    BaseType.__init__(self)
    # Database object
    self.database = None
    # Database directory. Default = working directory.
    self.databaseDir = ''

  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    # Check if a directory has been provided
    try:    self.databaseDir = xmlNode.attrib['directory']
    except KeyError: self.databaseDir = os.path.join(os.getcwd(),'DataBaseStorage')
    return

  def addInitParams(self,tempDict):
    '''
    Function that adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    ''' 
    return tempDict

  def addGroup(self,attributes,loadFrom):
    '''
    Function used to add group to the database
    @ In, attributes : options
    @ In, loadFrom   : source of the data
    '''
    pass

  def retrieveData(self,attributes):
    '''
    Function used to retrieve data from the database
    @ In, attributes : options
    @ Out, data      : the requested data
    '''
    pass
#    '''
#      Function used to finalize the the database
#      @ In, None 
#      @ Out, None
#    '''
#    def finalize(self):
#      self.database.closeDataBaseW()
#      pass
'''
  *************************s
  *  HDF5 DATABASE CLASS  *
  *************************
'''
class HDF5(DateBase):
  '''
  class to handle h5py (hdf5) database,
  to add and to retrieve attributes and values from it
  '''

  def __init__(self):
    '''
    Constructor
    '''
    DateBase.__init__(self)
    self.subtype  = None
    self.exist = False
    self.built = False
    self.type = "HDF5"

  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    DateBase.readMoreXML(self, xmlNode)
    '''
      Check if database directory exist, otherwise create it
    '''
    if not os.path.exists(self.databaseDir): os.makedirs(self.databaseDir)

    # Check if a filename has been provided
    # if yes, we assume the user wants to load the data from there
    # or update it
    try:
      file_name = xmlNode.attrib['filename']
      self.database = h5Data(self.name,self.databaseDir,file_name)
      self.exist   = True
    except KeyError:
      self.database = h5Data(self.name,self.databaseDir) 
      self.exist   = False
      
  def addInitParams(self,tempDict):
    '''
    Function that adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    ''' 
    tempDict = DateBase.addInitParams(self,tempDict)
    tempDict['exist'] = self.exist
    return tempDict

  def getEndingGroupPaths(self):
    '''
    Function to retrieve all the groups' paths of the ending groups
    @ In, None
    @ Out, List of the ending groups' paths 
    ''' 
    return self.database.retrieveAllHistoryPaths()

  def getEndingGroupNames(self):
    '''
    Function to retrieve all the groups' names of the ending groups
    @ In, None
    @ Out, List of the ending groups' names 
    ''' 
    return self.database.retrieveAllHistoryNames()

  def addGroup(self,attributes,loadFrom,upGroup=False):
    '''
    Function to add a group in the HDF5 database
    @ In, attributes : dictionary of attributes (metadata and options)
    @ In, loadFrom   : source of the data (for example, a csv file)
    @ Out, None 
    ''' 
    attributes["group"] = attributes['prefix']
    self.database.addGroup(attributes["group"],attributes,loadFrom,upGroup)
    self.built = True
    
  def addGroupDatas(self,attributes,loadFrom,upGroup=False):
    #### TODO: this function and the function above can be merged together (Andrea)
    '''
    Function to add a group in the HDF5 database
    @ In, attributes : dictionary of attributes (metadata and options)
    @ In, loadFrom   : source of the data (must be a data(s) or a dictionary)
    @ Out, None 
    ''' 
    if type(loadFrom) != dict:
      if not loadFrom.type in ['TimePoint','TimePointSet','History','Histories']: raise IOError('DATABASE      : ERROR addGroupDatas function needs to have a Datas as imput source')
      attributes['type'] = 'Datas'
    attributes['name'] = loadFrom
    self.database.addGroupDatas(attributes["group"],attributes,attributes,upGroup)
    self.built = True
  
  def initialize(self,gname,attributes=None,upGroup=False):
    '''
    Function to add an initial root group into the data base...
    This group will not contain a dataset but, eventually, only
    metadata
    @ In, gname      : name of the root group
    @ Out, attributes: metadata muste be appended to the root group
    '''
    self.database.addGroupInit(gname,attributes,upGroup)
  
  def returnHistory(self,attributes):
    '''
    Function to retrieve a history from the HDF5 database
    @ In, attributes : dictionary of attributes (metadata, history name and options)
    @ Out, tupleVar  : tuple in which the first position is a numpy aray and the second is a dictionary of the metadata
    Note:
    # DET => a Branch from the tail (group name in attributes) to the head (dependent on the filter)
    # MC  => The History named ["group"] (one run)
    '''
    if (not self.exist) and (not self.built): raise IOError("ERROR: Can not retrieve an History from data set" + self.name + ".It has not built yet.")
    if 'filter' in attributes.keys():#attributes['filter']:
      tupleVar = self.database.retrieveHistory(attributes["history"],attributes['filter'])
    else:
      tupleVar = self.database.retrieveHistory(attributes["history"])
    return copy.deepcopy(tupleVar)

  def __retrieveDataTimePoint(self,attributes):
    '''
    Function to retrieve a TimePoint from the HDF5 database
    @ In, attributes : dictionary of attributes (variables must be retrieved)
    @ Out, tupleVar  : tuple in which the first position is a dictionary of numpy arays (input variable) 
    and the second is a dictionary of the numpy arrays (output variables).
    Note: This function retrieve a TimePoint from an HDF5 database
    '''
    # Firstly, retrieve the history from which the TimePoint must be extracted
    histVar = self.returnHistory(attributes)
    # Check the outParam variables and the time filters
    if attributes['outParam'] == 'all': all_out_param  = True
    else: all_out_param = False
  
    if attributes['time'] == 'end' or (not attributes['time']):
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
    # Retrieve the field_names (aka headers if from CSV)
    if(all_out_param): field_names = histVar[1]["output_space_headers"]
      #all_field_names = field_names
    else:
      field_names = attributes['outParam']
      field_names.insert(0, 'time') 
      #all_field_names = histVar[1]["headers"]
    ints = 0
    if 'input_ts' in attributes.keys(): 
      if attributes['input_ts']: ints = int(attributes['input_ts'])
    else:                               ints = 0   
    
    # fill input param dictionary
    for key in attributes["inParam"]:
        if 'input_space_headers' in histVar[1]:
          if key in histVar[1]['input_space_headers']:
            ix = histVar[1]['input_space_headers'].index(key)
            inDict[key] = np.atleast_1d(np.array(histVar[1]['input_space_values'][ix]))
          elif key in histVar[1]["output_space_headers"]:
            ix = histVar[1]["output_space_headers"].index(key)
            if ints > histVar[0][:,0].size : raise IOError('DATABASE      : ******ERROR input_ts is greater than number of actual ts in history ' +attributes['history']+ '!') 
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: raise Exception("ERROR: the parameter " + key + " has not been found")            
        else:
          if key in histVar[1]["output_space_headers"] or \
             toBytes(key) in histVar[1]["output_space_headers"]:
            if key in histVar[1]["output_space_headers"]: 
              ix = histVar[1]["output_space_headers"].index(key)
            else:
              ix = histVar[1]["output_space_headers"].index(toBytes(key))
            if ints > histVar[0][:,0].size : raise IOError('DATABASE      : ******ERROR input_ts is greater than number of actual ts in history ' +attributes['history']+ '!')
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: raise Exception("ERROR: the parameter " + key + " has not been found")
  
    # Fill output param dictionary
    if time_end:
      # time end case => TimePoint is the final status 
      last_row = histVar[0][:,0].size - 1
      if all_out_param:
        # Retrieve all the parameters 
        for key in histVar[1]["output_space_headers"]: outDict[key] = np.atleast_1d(np.array(histVar[0][last_row,histVar[1]["output_space_headers"].index(key)]))
      else:
        # Retrieve only some parameters 
        for key in attributes['outParam']:
          if key in histVar[1]["output_space_headers"] or \
             toBytes(key) in histVar[1]["output_space_headers"]: 
            if key in histVar[1]["output_space_headers"]:
              outDict[key] = np.atleast_1d(np.array(histVar[0][last_row,histVar[1]["output_space_headers"].index(key)]))
            else:
              outDict[key] = np.atleast_1d(np.array(histVar[0][last_row,histVar[1]["output_space_headers"].index(toBytes(key))]))
          else: raise RuntimeError("ERROR: the parameter " + key + " has not been found")
    else:
      # Arbitrary point in time case... If the requested time point does not match any of the stored ones and 
      # start_time <= requested_time_point <= end_time, compute an interpolated value
      for i in histVar[0]:
        if histVar[0][i,0] >= time_float and time_float >= 0.0:
          if i-1 >= 0:
            previous_time = histVar[0][i-1,0]
          else:
            previous_time = histVar[0][i,0]
          actual_time   = histVar[0][i,0]
          if all_out_param:
            # Retrieve all the parameters 
            for key in histVar[1]["output_space_headers"]:
              if(actual_time == previous_time): outDict[key] = np.atleast_1d(np.array((histVar[0][i,histVar[1]["output_space_headers"].index(key)]  - time_float) / actual_time)) 
              else:
                actual_value   = histVar[0][i,histVar[1]["output_space_headers"].index(key)]
                previous_value = histVar[0][i-1,histVar[1]["output_space_headers"].index(key)] 
                outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))    
          else:
            # Retrieve only some parameters
            for key in attributes['outParam']:
              if key in histVar[1]["output_space_headers"]:
                if(actual_time == previous_time): outDict[key] = np.atleast_1d(np.array((histVar[0][i,histVar[1]["output_space_headers"].index(key)]  - time_float) / actual_time))
                else:
                  actual_value   = histVar[0][i,histVar[1]["output_space_headers"].index(key)]
                  previous_value = histVar[0][i-1,histVar[1]["output_space_headers"].index(key)] 
                  outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))    
                         
              else: raise Exception("ERROR: the parameter " + key + " has not been found")
    # return tuple of dictionaries
    return (copy.deepcopy(inDict),copy.deepcopy(outDict))

  def __retrieveDataTimePointSet(self,attributes):
    '''
    Function to retrieve a TimePointSet from the HDF5 database
    @ In, attributes : dictionary of attributes (variables must be retrieved)
    @ Out, tupleVar  : tuple in which the first position is a dictionary of numpy arays (input variable) 
    and the second is a dictionary of the numpy arrays (output variables).
    Note: This function retrieve a TimePointSet from an HDF5 database
    '''
    # Check the outParam variables and the time filters
    if attributes['outParam'] == 'all': all_out_param  = True
    else: all_out_param = False
  
    if attributes['time'] == 'end' or (not attributes['time']):
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = float(attributes['time'])

    ints = 0
    if 'input_ts' in attributes.keys(): 
      if attributes['input_ts']: ints = int(attributes['input_ts'])
    else:                               ints = 0   
          
    inDict  = {}
    outDict = {}    
    hist_list = []
    hist_list = attributes['histories']
    # Retrieve all the associated histories and process them
    for i in range(len(hist_list)): 
      # Load the data into the numpy array
      attributes['history'] = hist_list[i]
      histVar = self.returnHistory(attributes)

      if i == 0:
        if(all_out_param): field_names = histVar[1]["output_space_headers"]
        else:
          field_names = attributes['outParam']
          field_names.insert(0, 'time') 

      for key in attributes["inParam"]:
        if 'input_space_headers' in histVar[1]:
          if key in histVar[1]['input_space_headers']:
            ix = histVar[1]['input_space_headers'].index(key)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            inDict[key][i] = histVar[1]['input_space_values'][ix][0]
          elif key in histVar[1]["output_space_headers"]:
            ix = histVar[1]["output_space_headers"].index(key)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            if ints > histVar[0][:,0].size : raise IOError('DATABASE      : ******ERROR input_ts is greater than number of actual ts in history ' +hist_list[i]+ '!')
            inDict[key][i] = np.array(histVar[0][ints,ix])
          else: raise Exception("ERROR: the parameter " + key + " has not been found")            
        else:
          if key in histVar[1]["output_space_headers"] or\
             toBytes(key) in histVar[1]["output_space_headers"]:
            if key in histVar[1]["output_space_headers"]:
              ix = histVar[1]["output_space_headers"].index(key)
            else:
              ix = histVar[1]["output_space_headers"].index(toBytes(key))
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            if ints > histVar[0][:,0].size : raise IOError('DATABASE      : ******ERROR input_ts is greater than number of actual ts in history ' +hist_list[i]+ '!')
            inDict[key][i] = histVar[0][ints,ix]
          else: raise Exception("ERROR: the parameter " + key + " has not been found in "+str(histVar[1]))   
      
      # time end case => TimePointSet is at the final status 
      if time_end:
        last_row = histVar[0][:,0].size - 1
        if all_out_param:
          # Retrieve all the parameters 
          for key in histVar[1]["output_space_headers"]:
            if i == 0: outDict[key] = np.zeros(len(hist_list))
            outDict[key][i] = histVar[0][last_row,histVar[1]["output_space_headers"].index(key)]
        else:
          # Retrieve only some parameters
          for key in attributes['outParam']:
            if key in histVar[1]["output_space_headers"] or \
               toBytes(key) in histVar[1]["output_space_headers"]:
              if i == 0: outDict[key] = np.zeros(len(hist_list))
              if key in histVar[1]["output_space_headers"]:
                outDict[key][i] = histVar[0][last_row,histVar[1]["output_space_headers"].index(key)]
              else:
                outDict[key][i] = histVar[0][last_row,histVar[1]["output_space_headers"].index(toBytes(key))]
            else: raise RuntimeError("ERROR: the parameter " + str(key) + " has not been found")
      else:
        # Arbitrary point in time case... If the requested time point Set does not match any of the stored ones and 
        # start_time <= requested_time_point <= end_time, compute an interpolated value
        for i in histVar[0]:
          if histVar[0][i,0] >= time_float and time_float >= 0.0:
            if i-1 >= 0:
              previous_time = histVar[0][i-1,0]
            else:
              previous_time = histVar[0][i,0]
            actual_time   = histVar[0][i,0]          
            if all_out_param:
              # Retrieve all the parameters 
              for key in histVar[1]["output_space_headers"]:
                if(actual_time == previous_time):
                  if i == 0: outDict[key] = np.zeros(np.shape(len(hist_list)))
                  outDict[key][i] = (histVar[0][i,histVar[1]["output_space_headers"].index(key)]  - time_float) / actual_time 
                else:
                  if i == 0: outDict[key] = np.zeros(np.shape(len(hist_list)))
                  actual_value   = histVar[0][i,histVar[1]["output_space_headers"].index(key)]
                  previous_value = histVar[0][i-1,histVar[1]["output_space_headers"].index(key)] 
                  outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
            else:
              # Retrieve only some parameters
              for key in attributes['outParam']:
                if key in histVar[1]["output_space_headers"]:
                  if(actual_time == previous_time):
                    if i == 0:outDict[key] = np.zeros(np.shape(len(hist_list))) 
                    outDict[key][i] = (histVar[0][i,histVar[1]["output_space_headers"].index(key)]  - time_float) / actual_time 
                  else:
                    if i == 0: outDict[key] = np.zeros(np.shape(len(hist_list)))
                    actual_value   = histVar[0][i,histVar[1]["output_space_headers"].index(key)]
                    previous_value = histVar[0][i-1,histVar[1]["output_space_headers"].index(key)] 
                    outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)    
                else: raise RuntimeError("ERROR: the parameter " + key + " has not been found")      
      del histVar
    # return tuple of timepointSet
    return (copy.deepcopy(inDict),copy.deepcopy(outDict))

  def __retrieveDataHistory(self,attributes):
    '''
    Function to retrieve a History from the HDF5 database
    @ In, attributes : dictionary of attributes (variables and history name must be retrieved)
    @ Out, tupleVar  : tuple in which the first position is a dictionary of numpy arays (input variable) 
    and the second is a dictionary of the numpy arrays (output variables).
    Note: This function retrieve a History from an HDF5 database
    '''
    time_float = []
    # Check the outParam variables and the time filters
    if attributes['outParam'] == 'all': all_out_param  = True
    else:  all_out_param = False
    if 'time' in attributes and attributes['time']:
      if attributes['time'] == 'all': time_all = True
      else:
        # convert the time in float
        time_all = False
        time_float = [float(x) for x in attributes['time']]
    else: time_all = True
    
    ints = 0
    if 'input_ts' in attributes.keys(): 
      if attributes['input_ts']: ints = int(attributes['input_ts'])
    else:                               ints = 0   
                    
    inDict  = {}
    outDict = {}  
    # Call the function to retrieve a single history and 
    # load the data into the tuple 
    histVar = self.returnHistory(attributes)
    
    if(all_out_param): field_names = histVar[1]["output_space_headers"]
    else:
      # Retrieve only some parameters 
      field_names = attributes["outParam"]
      field_names.insert(0, 'time') 
    
    # fill input param dictionary
    for key in attributes["inParam"]:
        if 'input_space_headers' in histVar[1]:
          if key in histVar[1]['input_space_headers']:
            ix = histVar[1]['input_space_headers'].index(key)
            inDict[key] = np.atleast_1d(np.array(histVar[1]['input_space_values'][ix]))
          elif key in histVar[1]["output_space_headers"]:
            ix = histVar[1]["output_space_headers"].index(key)
            if ints > histVar[0][:,0].size : raise IOError('DATABASE      : ******ERROR input_ts is greater than number of actual ts in history ' +attributes['history']+ '!')
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: raise Exception("ERROR: the parameter " + key + " has not been found")            
        else:
          if key in histVar[1]["output_space_headers"]:
            ix = histVar[1]["output_space_headers"].index(key)
            if ints > histVar[0][:,0].size : raise IOError('DATABASE      : ******ERROR input_ts is greater than number of actual ts in history ' +attributes['history']+ '!')
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: raise Exception("ERROR: the parameter " + key + " has not been found")   

    # Time all case => The history is completed (from start_time to end_time)
    if time_all:
      if all_out_param:
        for key in histVar[1]["output_space_headers"]:
          outDict[key] = histVar[0][:,histVar[1]["output_space_headers"].index(key)]
      else:
        for key in attributes["outParam"]:
          if key in histVar[1]["output_space_headers"]:
            outDict[key] = histVar[0][:,histVar[1]["output_space_headers"].index(key)]        
          else:
            raise Exception("ERROR: the parameter " + key + " has not been found")
    else:
      # **************************************************************************
      # * it will be implemented when we decide a strategy about time filtering  *
      # * for now it is a copy paste of the time_all case                        *
      # **************************************************************************
      if all_out_param:
        for key in histVar[1]["output_space_headers"]: outDict[key] = histVar[0][:,histVar[1]["output_space_headers"].index(key)]
      else:
        for key in attributes["outParam"]:
          if key in histVar[1]["output_space_headers"]:
            outDict[key] = histVar[0][:,histVar[1]["output_space_headers"].index(key)]        
          else: raise Exception("ERROR: the parameter " + key + " has not been found")
    # Return tuple of dictionaries containing the histories
    return (copy.deepcopy(inDict),copy.deepcopy(outDict))

  def retrieveData(self,attributes):
    '''
    Function interface for retrieving a TimePoint or TimePointSet or History from the HDF5 database
    @ In, attributes : dictionary of attributes (variables, history name,metadata must be retrieved)
    @ Out, data     : tuple in which the first position is a dictionary of numpy arays (input variable) 
    and the second is a dictionary of the numpy arrays (output variables).
    Note: Interface function
    '''
    if attributes["type"] == "TimePoint":      data = self.__retrieveDataTimePoint(attributes)
    elif attributes["type"] == "TimePointSet": data = self.__retrieveDataTimePointSet(attributes)
    elif attributes["type"] == "History":      data = self.__retrieveDataHistory(attributes)
    elif attributes["type"] == "Histories":
      listhist_in  = {}
      listhist_out = {}
      endGroupNames = self.getEndingGroupNames()
      for index in xrange(len(endGroupNames)):
        attributes['history'] = endGroupNames[index]
        tupleVar = self.__retrieveDataHistory(attributes)
        # dictionary of dictionary key = i => ith history ParameterValues dictionary
        listhist_in[index]  = tupleVar[0]
        listhist_out[index] = tupleVar[1]
        del tupleVar
      data = (listhist_in,listhist_out)
    else: raise RuntimeError("Type" + attributes["type"] +" unknown.Caller: hdf5Manager.retrieveData")
    # return data
    gc.collect()
    return copy.deepcopy(data) 

__base                  = 'DataBase'
__interFaceDict         = {}
__interFaceDict['HDF5'] = HDF5
__knownTypes            = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  '''
  Function interface for creating an instance to a database specialized class (for example, HDF5)
  @ In, type                : class type (string)
  @ Out, class Instance     : instance to that class
  Note: Interface function
  '''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
