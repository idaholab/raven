'''
Created on April 9, 2013

@author: alfoa
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import os
import abc
import gc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from h5py_interface_creator import hdf5Database as h5Data
import utils
#Internal Modules End--------------------------------------------------------------------------------

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
    self.workingDir = ''
    self.printTag = utils.returnPrintTag('DATABASE')

  def _readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    # Check if a directory has been provided
    if 'directory' in xmlNode.attrib.keys(): self.databaseDir = copy.copy(xmlNode.attrib['directory'])
    else:                                    self.databaseDir = os.path.join(self.workingDir,'DatabaseStorage')

  def addInitParams(self,tempDict):
    '''
    Function that adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict
    '''
    return tempDict

  @abc.abstractmethod
  def addGroup(self,attributes,loadFrom):
    '''
    Function used to add group to the database
    @ In, attributes : options
    @ In, loadFrom   : source of the data
    '''
    pass
  @abc.abstractmethod
  def retrieveData(self,attributes):
    '''
    Function used to retrieve data from the database
    @ In, attributes : options
    @ Out, data      : the requested data
    '''
    pass

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

  def __init__(self,runInfoDict):
    '''
    Constructor
    '''
    DateBase.__init__(self)
    self.subtype  = None
    self.exist    = False
    self.built    = False
    self.type     = 'HDF5'
    self.file_name = ""
    self.printTag = utils.returnPrintTag('DATABASE HDF5')
    self.workingDir = runInfoDict['WorkingDir']
    self.databaseDir = self.workingDir

  def __getstate__(self):
    """
    Overwrite state (for pickle-ing)
    we do not pickle the HDF5 (C++) instance
    but only the info to re-load it
    """
    # capture what is normally pickled
    state = self.__dict__.copy()
    # we pop the database instance and close it
    state.pop("database")
    self.database.closeDatabaseW()
    # what we return here will be stored in the pickle
    return state

  def __setstate__(self, newstate):
    self.__dict__.update(newstate)
    self.database = h5Data(self.name,self.databaseDir,self.file_name)
    self.exist    = True

  def _readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    DateBase._readMoreXML(self, xmlNode)
    # Check if database directory exist, otherwise create it
    if '~' in self.databaseDir: self.databaseDir = copy.copy(os.path.expanduser(self.databaseDir))
    if not os.path.exists(self.databaseDir): os.makedirs(self.databaseDir)
    utils.raiseAMessage(self,'Database Directory is '+self.databaseDir+'!')
    # Check if a filename has been provided
    # if yes, we assume the user wants to load the data from there
    # or update it
    try:
      self.file_name = xmlNode.attrib['filename']
      self.database = h5Data(self.name,self.databaseDir,self.file_name)
      self.exist    = True
    except KeyError:
      self.file_name = self.name+".h5"
      self.database  = h5Data(self.name,self.databaseDir)
      self.exist     = False

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
    if 'metadata' in attributes.keys(): attributes['group'] = attributes['metadata']['prefix']
    elif 'prefix' in attributes.keys(): attributes['group'] = attributes['prefix']
    else                              : utils.raiseAnError(IOError,self,'addGroup function needs a prefix (ID) for adding a new group to a database!')
    self.database.addGroup(attributes['group'],attributes,loadFrom,upGroup)
    self.built = True

  def addGroupDataObjects(self,attributes,loadFrom,upGroup=False):
    #### TODO: this function and the function above can be merged together (Andrea)
    '''
    Function to add a group in the HDF5 database
    @ In, attributes : dictionary of attributes (metadata and options)
    @ In, loadFrom   : source of the data (must be a data(s) or a dictionary)
    @ Out, None
    '''
    source = {}
    if type(loadFrom) != dict:
      if not loadFrom.type in ['TimePoint','TimePointSet','History','Histories']: utils.raiseAnError(IOError,self,'addGroupDataObjects function needs to have a Data(s) as input source')
      source['type'] = 'DataObjects'
    source['name'] = loadFrom
    self.database.addGroupDataObjects(attributes['group'],attributes,source,upGroup)
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
    # MC  => The History named ['group'] (one run)
    '''
    if (not self.exist) and (not self.built): utils.raiseAnError(IOError,self,'Can not retrieve an History from data set' + self.name + '.It has not built yet.')
    if 'filter' in attributes.keys(): tupleVar = self.database.retrieveHistory(attributes['history'],attributes['filter'])
    else:                             tupleVar = self.database.retrieveHistory(attributes['history'])
    return tupleVar

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
    else:                               all_out_param = False

    if attributes['time'] == 'end' or (not attributes['time']):
      time_end = True
      time_float = -1.0
    else:
      # convert the time in float
      time_end = False
      time_float = float(attributes['time'])

    if 'operator' in attributes.keys():
      operator = True
      time_end = True
      time_float = -1.0
    else: operator = False

    inDict  = {}
    outDict = {}
    metaDict= {}
    if 'metadata' in histVar[1].keys(): metaDict[0] = histVar[1]['metadata']
    else                              : metaDict[0] = None

    ints = 0
    if 'inputTs' in attributes.keys():
      if attributes['inputTs']: ints = int(attributes['inputTs'])
    else:                               ints = 0

    # fill input param dictionary
    for key in attributes['inParam']:
        if 'input_space_headers' in histVar[1]:
          inInKey = utils.keyIn(histVar[1]['input_space_headers'],key)
          inOutKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inInKey is not None:
            ix = histVar[1]['input_space_headers'].index(inInKey)
            inDict[key] = np.atleast_1d(np.array(histVar[1]['input_space_values'][ix]))
          elif inOutKey is not None:
            ix = histVar[1]['output_space_headers'].index(inOutKey)
            if ints > histVar[0][:,0].size  and ints != -1: utils.raiseAnError(IOError,self,'inputTs is greater than number of actual ts in history ' +attributes['history']+ '!')
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: utils.raiseAnError(IOError,self,'the parameter ' + key + ' has not been found')
        else:
          if key in histVar[1]['output_space_headers'] or \
             utils.toBytes(key) in histVar[1]['output_space_headers']:
            if key in histVar[1]['output_space_headers']:
              ix = histVar[1]['output_space_headers'].index(key)
            else:
              ix = histVar[1]['output_space_headers'].index(utils.toBytes(key))
            if ints > histVar[0][:,0].size  and ints != -1: utils.raiseAnError(IOError,self,'inputTs is greater than number of actual ts in history ' +attributes['history']+ '!')
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: utils.raiseAnError(IOError,self,'the parameter ' + key + ' has not been found')

    # Fill output param dictionary
    if time_end:
      # time end case => TimePoint is the final status
      if all_out_param:
        # Retrieve all the parameters
        if operator:
          if   attributes['operator'].lower() == 'max'    : outDict[key] = np.atleast_1d(np.array(max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
          elif attributes['operator'].lower() == 'min'    : outDict[key] = np.atleast_1d(np.array(min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
          elif attributes['operator'].lower() == 'average': outDict[key] = np.atleast_1d(np.array(np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
          else: utils.raiseAnError(IOError,self,'Operator '+ attributes['operator'] + ' unknown for TimePoint construction. Available are min,max,average!!')
        else:
          for key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(histVar[0][-1,histVar[1]['output_space_headers'].index(key)]))
      else:
        # Retrieve only some parameters
        for key in attributes['outParam']:
          if key in histVar[1]['output_space_headers'] or \
             utils.toBytes(key) in histVar[1]['output_space_headers']:
            if operator:
              if   attributes['operator'].lower() == 'max'    :
                if key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
                else: outDict[key] = np.atleast_1d(np.array(max(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])))
              elif attributes['operator'].lower() == 'min'    :
                if key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
                else: outDict[key] = np.atleast_1d(np.array(min(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])))
              elif attributes['operator'].lower() == 'average':
                if key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
                else: outDict[key] = np.atleast_1d(np.array(np.average(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])))
              else: utils.raiseAnError(IOError,self,'Operator '+ attributes['operator'] + ' unknown for TimePoint construction. Available are min,max,average!!')
            else:
              if key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(histVar[0][-1,histVar[1]['output_space_headers'].index(key)]))
              else: outDict[key] = np.atleast_1d(np.array(histVar[0][-1,histVar[1]['output_space_headers'].index(utils.toBytes(key))]))
          else: utils.raiseAnError(RuntimeError,self,'the parameter ' + key + ' has not been found')
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
            for key in histVar[1]['output_space_headers']:
              if(actual_time == previous_time): outDict[key] = np.atleast_1d(np.array((histVar[0][i,histVar[1]['output_space_headers'].index(key)]  - time_float) / actual_time))
              else:
                actual_value   = histVar[0][i,histVar[1]['output_space_headers'].index(key)]
                previous_value = histVar[0][i-1,histVar[1]['output_space_headers'].index(key)]
                outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))
          else:
            # Retrieve only some parameters
            for key in attributes['outParam']:
              if key in histVar[1]['output_space_headers']:
                if(actual_time == previous_time): outDict[key] = np.atleast_1d(np.array((histVar[0][i,histVar[1]['output_space_headers'].index(key)]  - time_float) / actual_time))
                else:
                  actual_value   = histVar[0][i,histVar[1]['output_space_headers'].index(key)]
                  previous_value = histVar[0][i-1,histVar[1]['output_space_headers'].index(key)]
                  outDict[key] = np.atleast_1d(np.array((actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)))
              else: utils.raiseAnError(IOError,self,'the parameter ' + key + ' has not been found')
    # return tuple of dictionaries
    return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

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

    if 'operator' in attributes.keys():
      operator = True
      time_end = True
      time_float = -1.0
    else: operator = False


    ints = 0
    if 'inputTs' in attributes.keys():
      if attributes['inputTs']: ints = int(attributes['inputTs'])
    else:                               ints = 0

    inDict   = {}
    outDict  = {}
    metaDict = {}
    hist_list = attributes['histories']
    # Retrieve all the associated histories and process them
    for i in range(len(hist_list)):
      # Load the data into the numpy array
      attributes['history'] = hist_list[i]
      histVar = self.returnHistory(attributes)
      if 'metadata' in histVar[1].keys(): metaDict[i] = histVar[1]['metadata']
      else                              : metaDict[i] = None
      for key in attributes['inParam']:
        if 'input_space_headers' in histVar[1]:
          inInKey = utils.keyIn(histVar[1]['input_space_headers'],key)
          inOutKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inInKey is not None:
            ix = histVar[1]['input_space_headers'].index(inInKey)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            inDict[key][i] = np.atleast_1d(histVar[1]['input_space_values'][ix])[0]
          elif inOutKey is not None:
            ix = histVar[1]['output_space_headers'].index(inOutKey)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            if ints > histVar[0][:,0].size and ints != -1: utils.raiseAnError(IOError,self,'inputTs is greater than number of actual ts in history ' +hist_list[i]+ '!')
            inDict[key][i] = np.array(histVar[0][ints,ix])
          else: utils.raiseAnError(IOError,self,'the parameter ' + key + ' has not been found')
        else:
          inKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inKey is not None:
            ix = histVar[1]['output_space_headers'].index(inKey)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            if ints > histVar[0][:,0].size  and ints != -1: utils.raiseAnError(IOError,self,'inputTs is greater than number of actual ts in history ' +hist_list[i]+ '!')
            inDict[key][i] = histVar[0][ints,ix]
          else: utils.raiseAnError(IOError,self,'the parameter ' + key + ' has not been found in '+str(histVar[1]))

      # time end case => TimePointSet is at the final status
      if time_end:
        if all_out_param:
          # Retrieve all the parameters
          for key in histVar[1]['output_space_headers']:
            if i == 0: outDict[key] = np.zeros(len(hist_list))
            if operator:
              if   attributes['operator'].lower() == 'max'    : outDict[key][i] = max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
              elif attributes['operator'].lower() == 'min'    : outDict[key][i] = min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
              elif attributes['operator'].lower() == 'average': outDict[key][i] = np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
              else: utils.raiseAnError(IOError,self,'Operator '+ attributes['operator'] + ' unknown for TimePointSet construction. Available are min,max,average!!')
            else:
              outDict[key][i] = histVar[0][-1,histVar[1]['output_space_headers'].index(key)]
        else:
          # Retrieve only some parameters
          for key in attributes['outParam']:
            if key in histVar[1]['output_space_headers'] or \
               utils.toBytes(key) in histVar[1]['output_space_headers']:
              if i == 0: outDict[key] = np.zeros(len(hist_list))
              if operator:
                if   attributes['operator'].lower() == 'max'    :
                  if key in histVar[1]['output_space_headers']: outDict[key][i] = max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
                  else: outDict[key][i] = max(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])
                elif attributes['operator'].lower() == 'min'    :
                  if key in histVar[1]['output_space_headers']: outDict[key][i] = min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
                  else: outDict[key][i] = min(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])
                elif attributes['operator'].lower() == 'average':
                  if key in histVar[1]['output_space_headers']: outDict[key][i] = np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
                  else: outDict[key][i] = np.average(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])
                else: utils.raiseAnError(IOError,self,'Operator '+ attributes['operator'] + ' unknown for TimePointSet construction. Available are min,max,average!!')
              else:
                if key in histVar[1]['output_space_headers']: outDict[key][i] = histVar[0][-1,histVar[1]['output_space_headers'].index(key)]
                else: outDict[key][i] = histVar[0][-1,histVar[1]['output_space_headers'].index(utils.toBytes(key))]
            else: utils.raiseAnError(RuntimeError,self,'the parameter ' + str(key) + ' has not been found')
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
              for key in histVar[1]['output_space_headers']:
                if(actual_time == previous_time):
                  if i == 0: outDict[key] = np.zeros(np.shape(len(hist_list)))
                  outDict[key][i] = (histVar[0][i,histVar[1]['output_space_headers'].index(key)]  - time_float) / actual_time
                else:
                  if i == 0: outDict[key] = np.zeros(np.shape(len(hist_list)))
                  actual_value   = histVar[0][i,histVar[1]['output_space_headers'].index(key)]
                  previous_value = histVar[0][i-1,histVar[1]['output_space_headers'].index(key)]
                  outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)
            else:
              # Retrieve only some parameters
              for key in attributes['outParam']:
                if key in histVar[1]['output_space_headers']:
                  if(actual_time == previous_time):
                    if i == 0:outDict[key] = np.zeros(np.shape(len(hist_list)))
                    outDict[key][i] = (histVar[0][i,histVar[1]['output_space_headers'].index(key)]  - time_float) / actual_time
                  else:
                    if i == 0: outDict[key] = np.zeros(np.shape(len(hist_list)))
                    actual_value   = histVar[0][i,histVar[1]['output_space_headers'].index(key)]
                    previous_value = histVar[0][i-1,histVar[1]['output_space_headers'].index(key)]
                    outDict[key][i] = (actual_value-previous_value)/(actual_time-previous_time)*(time_float-previous_time)
                else: utils.raiseAnError(RuntimeError,self,'the parameter ' + key + ' has not been found')
      del histVar
    # return tuple of timepointSet
    return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

  def __retrieveDataHistory(self,attributes):
    '''
    Function to retrieve a History from the HDF5 database
    @ In, attributes : dictionary of attributes (variables and history name must be retrieved)
    @ Out, tupleVar  : tuple in which the first position is a dictionary of numpy arays (input variable)
    and the second is a dictionary of the numpy arrays (output variables).
    Note: This function retrieve a History from an HDF5 database
    '''
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
    if 'inputTs' in attributes.keys():
      if attributes['inputTs']: ints = int(attributes['inputTs'])
    else:                               ints = 0

    inDict  = {}
    outDict = {}
    metaDict= {}
    # Call the function to retrieve a single history and
    # load the data into the tuple
    histVar = self.returnHistory(attributes)
    if 'metadata' in histVar[1].keys(): metaDict[0] = histVar[1]['metadata']
    else                              : metaDict[0] = None
    # fill input param dictionary
    for key in attributes['inParam']:
        if 'input_space_headers' in histVar[1]:
          inInKey = utils.keyIn(histVar[1]['input_space_headers'],key)
          inOutKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inInKey is not None:
            ix = histVar[1]['input_space_headers'].index(inInKey)
            inDict[key] = np.atleast_1d(np.array(histVar[1]['input_space_values'][ix]))
          elif inOutKey is not None:
            ix = histVar[1]['output_space_headers'].index(inOutKey)
            if ints > histVar[0][:,0].size  and ints != -1: utils.raiseAnError(IOError,self,'inputTs is greater than number of actual ts in history ' +attributes['history']+ '!')
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: utils.raiseAnError(RuntimeError,self,'the parameter ' + key + ' has not been found in '+str(histVar[1]['input_space_headers'])+' or '+str(histVar[1]['output_space_headers']))
        else:
          inKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inKey is not None:
            ix = histVar[1]['output_space_headers'].index(inKey)
            if ints > histVar[0][:,0].size  and ints != -1: utils.raiseAnError(IOError,self,'inputTs is greater than number of actual ts in history ' +attributes['history']+ '!')
            inDict[key] = np.atleast_1d(np.array(histVar[0][ints,ix]))
          else: utils.raiseAnError(RuntimeError,self,'the parameter ' + key + ' has not been found in '+str(histVar[1]['output_space_headers']))

    # Time all case => The history is completed (from start_time to end_time)
    if time_all:
      if all_out_param:
        for key in histVar[1]['output_space_headers']:
          outDict[key] = histVar[0][:,histVar[1]['output_space_headers'].index(key)]
      else:
        for key in attributes['outParam']:
          inKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inKey:
            outDict[key] = histVar[0][:,histVar[1]['output_space_headers'].index(inKey)]
          else:
            utils.raiseAnError(RuntimeError,self,'the parameter ' + key + ' has not been found in '+str(histVar[1]['output_space_headers']))
    else:
      # **************************************************************************
      # * it will be implemented when we decide a strategy about time filtering  *
      # * for now it is a copy paste of the time_all case                        *
      # **************************************************************************
      if all_out_param:
        for key in histVar[1]['output_space_headers']: outDict[key] = histVar[0][:,histVar[1]['output_space_headers'].index(key)]
      else:
        for key in attributes['outParam']:
          if key in histVar[1]['output_space_headers']:
            outDict[key] = histVar[0][:,histVar[1]['output_space_headers'].index(key)]
          else: utils.raiseAnError(RuntimeError,self,'the parameter ' + key + ' has not been found')
    # Return tuple of dictionaries containing the histories
    return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

  def retrieveData(self,attributes):
    '''
    Function interface for retrieving a TimePoint or TimePointSet or History from the HDF5 database
    @ In, attributes : dictionary of attributes (variables, history name,metadata must be retrieved)
    @ Out, data     : tuple in which the first position is a dictionary of numpy arays (input variable)
    and the second is a dictionary of the numpy arrays (output variables).
    Note: Interface function
    '''
    if attributes['type'] == 'TimePoint':      data = self.__retrieveDataTimePoint(attributes)
    elif attributes['type'] == 'TimePointSet': data = self.__retrieveDataTimePointSet(attributes)
    elif attributes['type'] == 'History':      data = self.__retrieveDataHistory(attributes)
    elif attributes['type'] == 'Histories':
      listhist_in  = {}
      listhist_out = {}
      listhist_meta= {}
      endGroupNames = self.getEndingGroupNames()
      for index in range(len(endGroupNames)):
        attributes['history'] = endGroupNames[index]
        tupleVar = self.__retrieveDataHistory(attributes)
        # dictionary of dictionary key = i => ith history ParameterValues dictionary
        listhist_in[index]  = tupleVar[0]
        listhist_out[index] = tupleVar[1]
        listhist_meta[index]= tupleVar[2]
        del tupleVar
      data = (listhist_in,listhist_out,listhist_meta)
    else: utils.raiseAnError(RuntimeError,self,'Type' + attributes['type'] +' unknown.Caller: hdf5Manager.retrieveData')
    # return data
    gc.collect()
    return copy.copy(data)

__base                  = 'Database'
__interFaceDict         = {}
__interFaceDict['HDF5'] = HDF5
__knownTypes            = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

needsRunInfo = True

def returnInstance(Type,runInfoDict):
  '''
  Function interface for creating an instance to a database specialized class (for example, HDF5)
  @ In, type                : class type (string)
  @ Out, class Instance     : instance to that class
  Note: Interface function
  '''
  try: return __interFaceDict[Type](runInfoDict)
  except KeyError: utils.raiseAnError(NameError,'DATABASES','not known '+__base+' type '+Type)
