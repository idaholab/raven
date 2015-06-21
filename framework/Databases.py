"""
Created on April 9, 2013

@author: alfoa
"""
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
from scipy.interpolate import interp1d
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from h5py_interface_creator import hdf5Database as h5Data
import utils
#Internal Modules End--------------------------------------------------------------------------------

class DateBase(BaseType):
  """
  class to handle database,
  to add and to retrieve attributes and values from it
  """

  def __init__(self):
    """
    Constructor
    """
    # Base Class
    BaseType.__init__(self)
    # Database object
    self.database = None
    # Database directory. Default = working directory.
    self.databaseDir = ''
    self.workingDir = ''
    self.printTag = 'DATABASE'

  def _readMoreXML(self,xmlNode):
    """
    Function to read the portion of the xml input that belongs to this class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    """
    # Check if a directory has been provided
    if 'directory' in xmlNode.attrib.keys(): self.databaseDir = copy.copy(xmlNode.attrib['directory'])
    else:                                    self.databaseDir = os.path.join(self.workingDir,'DatabaseStorage')

  def addInitParams(self,tempDict):
    """
    Function that adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict
    """
    return tempDict

  @abc.abstractmethod
  def addGroup(self,attributes,loadFrom):
    """
    Function used to add group to the database
    @ In, attributes : options
    @ In, loadFrom   : source of the data
    """
    pass
  @abc.abstractmethod
  def retrieveData(self,attributes):
    """
    Function used to retrieve data from the database
    @ In, attributes : options
    @ Out, data      : the requested data
    """
    pass
"""
  *************************s
  *  HDF5 DATABASE CLASS  *
  *************************
"""
class HDF5(DateBase):
  """
  class to handle h5py (hdf5) database,
  to add and to retrieve attributes and values from it
  """
  def __init__(self,runInfoDict):
    """
    Constructor
    """
    DateBase.__init__(self)
    self.subtype  = None
    self.exist    = False
    self.built    = False
    self.type     = 'HDF5'
    self.file_name = ""
    self.printTag = 'DATABASE HDF5'
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
    """
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    """
    DateBase._readMoreXML(self, xmlNode)
    # Check if database directory exist, otherwise create it
    if '~' in self.databaseDir: self.databaseDir = copy.copy(os.path.expanduser(self.databaseDir))
    if not os.path.exists(self.databaseDir): os.makedirs(self.databaseDir)
    self.raiseAMessage('Database Directory is '+self.databaseDir+'!')
    # Check if a filename has been provided
    # if yes, we assume the user wants to load the data from there
    # or update it
    #try:
    if 'filename' in xmlNode.attrib.keys():
      self.file_name = xmlNode.attrib['filename']
      self.database = h5Data(self.name,self.databaseDir,self.messageHandler,self.file_name)
      self.exist    = True
    #except KeyError:
    else:
      self.file_name = self.name+".h5"
      self.database  = h5Data(self.name,self.databaseDir,self.messageHandler)
      self.exist     = False

  def addInitParams(self,tempDict):
    """
    Function that adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict
    """
    tempDict = DateBase.addInitParams(self,tempDict)
    tempDict['exist'] = self.exist
    return tempDict

  def getEndingGroupPaths(self):
    """
    Function to retrieve all the groups' paths of the ending groups
    @ In, None
    @ Out, List of the ending groups' paths
    """
    return self.database.retrieveAllHistoryPaths()

  def getEndingGroupNames(self):
    """
    Function to retrieve all the groups' names of the ending groups
    @ In, None
    @ Out, List of the ending groups' names
    """
    return self.database.retrieveAllHistoryNames()

  def addGroup(self,attributes,loadFrom,upGroup=False):
    """
    Function to add a group in the HDF5 database
    @ In, attributes : dictionary of attributes (metadata and options)
    @ In, loadFrom   : source of the data (for example, a csv file)
    @ Out, None
    """
    if 'metadata' in attributes.keys(): attributes['group'] = attributes['metadata']['prefix']
    elif 'prefix' in attributes.keys(): attributes['group'] = attributes['prefix']
    else                              : self.raiseAnError(IOError,'addGroup function needs a prefix (ID) for adding a new group to a database!')
    self.database.addGroup(attributes['group'],attributes,loadFrom,upGroup)
    self.built = True

  def addGroupDataObjects(self,attributes,loadFrom,upGroup=False):
    #### TODO: this function and the function above can be merged together (Andrea)
    """
    Function to add a group in the HDF5 database
    @ In, attributes : dictionary of attributes (metadata and options)
    @ In, loadFrom   : source of the data (must be a data(s) or a dictionary)
    @ Out, None
    """
    source = {}
    if type(loadFrom) != dict:
      if not loadFrom.type in ['Point','PointSet','History','HistorySet']: self.raiseAnError(IOError,'addGroupDataObjects function needs to have a Data(s) as input source')
      source['type'] = 'DataObjects'
    source['name'] = loadFrom
    self.database.addGroupDataObjects(attributes['group'],attributes,source,upGroup)
    self.built = True

  def initialize(self,gname,attributes=None,upGroup=False):
    """
    Function to add an initial root group into the data base...
    This group will not contain a dataset but, eventually, only
    metadata
    @ In, gname      : name of the root group
    @ Out, attributes: metadata muste be appended to the root group
    """
    self.database.addGroupInit(gname,attributes,upGroup)

  def returnHistory(self,attributes):
    """
    Function to retrieve a history from the HDF5 database
    @ In, attributes : dictionary of attributes (metadata, history name and options)
    @ Out, tupleVar  : tuple in which the first position is a numpy aray and the second is a dictionary of the metadata
    Note:
    # DET => a Branch from the tail (group name in attributes) to the head (dependent on the filter)
    # MC  => The History named ['group'] (one run)
    """
    if (not self.exist) and (not self.built): self.raiseAnError(IOError,'Can not retrieve an History from data set' + self.name + '.It has not built yet.')
    if 'filter' in attributes.keys(): tupleVar = self.database.retrieveHistory(attributes['history'],attributes['filter'])
    else:                             tupleVar = self.database.retrieveHistory(attributes['history'])
    return tupleVar

  def __retrieveDataPoint(self,attributes):
    """
    Function to retrieve a Point from the HDF5 database
    @ In, attributes : dictionary of attributes (variables must be retrieved)
    @ Out, tupleVar  : tuple in which the first position is a dictionary of numpy arays (input variable)
    and the second is a dictionary of the numpy arrays (output variables).
    Note: This function retrieve a Point from an HDF5 database
    """
    # Firstly, retrieve the history from which the Point must be extracted
    histVar = self.returnHistory(attributes)
    # Check the outParam variables and the outputPivotVal filters
    inParam, outParam, inputRow, outputRow                 = attributes['inParam'], attributes['outParam'], attributes.get('inputRow',None), attributes.get('outputRow',None)
    inputPivotVal, outputPivotVal, operator   = attributes.get('inputPivotValue',None), attributes.get('outputPivotValue',None), attributes.get('operator',None)
    pivotParameter                                         = attributes.get('pivotParameter',None)
    if 'all' in outParam: all_out_param = True
    else                : all_out_param = False
    if outputPivotVal != None:
      if 'end' in outputPivotVal: outputPivotVal_end = True
      else:
        outputPivotVal_end, outputPivotVal = False,  float(outputPivotVal)
    else: outputPivotVal_end = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None : inputRow  = int(inputRow) - 1
    if outputRow != None: outputRow = int(outputRow) - 1

    if pivotParameter != None:
      pivotIndex = histVar[1]['output_space_headers'].index(pivotParameter) if pivotParameter in histVar[1]['output_space_headers'] else None
      if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in Database '+ str(self.name) + '!')
    else:
      pivotIndex = histVar[1]['output_space_headers'].index("time") if "time" in histVar[1]['output_space_headers'] else None
      # if None...default is 0
      if pivotIndex == None: pivotIndex = 0
    if inputRow > histVar[0][:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in Database '+ str(self.name) + '!')

    inDict  = {}
    outDict = {}
    metaDict= {}

    if 'metadata' in histVar[1].keys(): metaDict[0] = histVar[1]['metadata']
    else                              : metaDict[0] = None

    for key in inParam:
      if 'input_space_headers' in histVar[1]:
        inInKey = utils.keyIn(histVar[1]['input_space_headers'],key)
        inOutKey = utils.keyIn(histVar[1]['output_space_headers'],key)
        if inInKey != None:
          ix = histVar[1]['input_space_headers'].index(inInKey)
          inDict[key] = np.atleast_1d(np.array(histVar[1]['input_space_values'][ix]))
        elif inOutKey != None and inInKey == None:
          ix = histVar[1]['output_space_headers'].index(inOutKey)
          if inputPivotVal != None:
            if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
            inDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))))
          else: inDict[key] = np.atleast_1d(np.array(histVar[0][inputRow,ix]))
        else: self.raiseAnError(IOError,'the parameter ' + key + ' has not been found')
      else:
        inKey = utils.keyIn(histVar[1]['output_space_headers'],key)
        if inKey is not None:
          ix = histVar[1]['output_space_headers'].index(inKey)
          if inputPivotVal != None:
            if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
            inDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))))
          else: inDict[key] = np.atleast_1d(np.array(histVar[0][inputRow,ix]))
        else: self.raiseAnError(IOError,'the parameter ' + key + ' has not been found in '+str(histVar[1]))
    # outputPivotVal end case => PointSet is at the final status
    if outputPivotVal_end:
      if all_out_param:
        for key in histVar[1]['output_space_headers']:
          outDict[key] = np.atleast_1d(np.array(histVar[0][-1,histVar[1]['output_space_headers'].index(key)]))
      else:
        for key in outParam:
          if key in histVar[1]['output_space_headers'] or \
             utils.toBytes(key) in histVar[1]['output_space_headers']:
            if key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(histVar[0][-1,histVar[1]['output_space_headers'].index(key)]))
            else: outDict[key] = np.atleast_1d(np.array(histVar[0][-1,histVar[1]['output_space_headers'].index(utils.toBytes(key))]))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    elif outputRow != None:
      if outputRow > histVar[0][:,0].size-1  and outputRow != -1: self.raiseAnError(IOError,'outputRow is greater than number of actual rows in Database '+ str(self.name) + '!')
      if all_out_param:
        for key in histVar[1]['output_space_headers']:
          outDict[key] = np.atleast_1d(np.array(histVar[0][outputRow,histVar[1]['output_space_headers'].index(key)]))
      else:
        for key in outParam:
          if key in histVar[1]['output_space_headers'] or \
             utils.toBytes(key) in histVar[1]['output_space_headers']:
            if key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(histVar[0][outputRow,histVar[1]['output_space_headers'].index(key)]))
            else: outDict[key] = np.atleast_1d(np.array(histVar[0][outputRow,histVar[1]['output_space_headers'].index(utils.toBytes(key))]))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    elif operator != None:
      if operator not in ['max','min','average']: self.raiseAnError(IOError,'operator unknown. Available are min,max,average')
      if histVar[1]['output_space_headers']:
        for key in histVar[1]['output_space_headers']:
          if operator == 'max'    : outDict[key] = np.atleast_1d(np.array(np.max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
          if operator == 'min'    : outDict[key] = np.atleast_1d(np.array(np.min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
          if operator == 'average': outDict[key] = np.atleast_1d(np.array(np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
      else:
        for key in outParam:
          if key in histVar[1]['output_space_headers'] or \
             utils.toBytes(key) in histVar[1]['output_space_headers']:
            if key in histVar[1]['output_space_headers']:
              if operator == 'max'    : outDict[key] = np.atleast_1d(np.array(np.max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
              if operator == 'min'    : outDict[key] = np.atleast_1d(np.array(np.min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
              if operator == 'average': outDict[key] = np.atleast_1d(np.array(np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])))
            else:
              if operator == 'max'    : outDict[key] = np.atleast_1d(np.array(np.max(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])))
              if operator == 'min'    : outDict[key] = np.atleast_1d(np.array(np.min(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])))
              if operator == 'average': outDict[key] = np.atleast_1d(np.array(np.average(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    else:
      # Arbitrary point in outputPivotVal case... If the requested outputPivotVal point Set does not match any of the stored ones and
      # start_outputPivotVal <= requested_outputPivotVal_point <= end_outputPivotVal, compute an interpolated value
      if all_out_param:
        for key in histVar[1]['output_space_headers']:
          outDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(key)], kind='linear')(outputPivotVal)))
      else:
        for key in outParam:
          if key in histVar[1]['output_space_headers'] or \
             utils.toBytes(key) in histVar[1]['output_space_headers']:
            if key in histVar[1]['output_space_headers']: outDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(key)], kind='linear')(outputPivotVal)))
            else                                        : outDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))], kind='linear')(outputPivotVal)))
          else                                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    # return tuple of dictionaries
    return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

  def __retrieveDataPointSet(self,attributes):
    """
    Function to retrieve a PointSet from the HDF5 database
    @ In, attributes : dictionary of attributes (variables must be retrieved)
    @ Out, tupleVar  : tuple in which the first position is a dictionary of numpy arays (input variable)
    and the second is a dictionary of the numpy arrays (output variables).
    Note: This function retrieve a PointSet from an HDF5 database
    """
    # Check the outParam variables and the outputPivotVal filters
    inParam, outParam, inputRow, outputRow                 = attributes['inParam'], attributes['outParam'], attributes.get('inputRow',None), attributes.get('outputRow',None)
    inputPivotVal, outputPivotVal, operator                = attributes.get('inputPivotValue',None), attributes.get('outputPivotValue',None), attributes.get('operator',None)
    pivotParameter                                         = attributes.get('pivotParameter',None)

    if outParam == 'all': all_out_param  = True
    else: all_out_param = False

    if outputPivotVal != None:
      if 'end' in outputPivotVal: outputPivotVal_end = True
      else:
        outputPivotVal_end, outputPivotVal = False,  float(outputPivotVal)
    else: outputPivotVal_end = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None : inputRow  = int(inputRow) - 1
    if outputRow != None: outputRow = int(outputRow) - 1

    inDict   = {}
    outDict  = {}
    metaDict = {}
    hist_list = attributes['HistorySet']
    # Retrieve all the associated HistorySet and process them
    for i in range(len(hist_list)):
      # Load the data into the numpy array
      attributes['history'] = hist_list[i]
      histVar = self.returnHistory(attributes)
      #look for pivotParameter
      if pivotParameter != None:
        pivotIndex = histVar[1]['output_space_headers'].index(pivotParameter) if pivotParameter in histVar[1]['output_space_headers'] else None
        if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in history '+ str(attributes['history']) + '!')
      else:
        pivotIndex = histVar[1]['output_space_headers'].index("time") if "time" in histVar[1]['output_space_headers'] else None
        # if None...default is 0
        if pivotIndex == None: pivotIndex = 0
      if inputRow > histVar[0][:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in history '+ str(attributes['history']) + '!')
      # check metadata
      if 'metadata' in histVar[1].keys(): metaDict[i] = histVar[1]['metadata']
      else                              : metaDict[i] = None
      for key in inParam:
        if 'input_space_headers' in histVar[1]:
          inInKey = utils.keyIn(histVar[1]['input_space_headers'],key)
          inOutKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inInKey != None:
            ix = histVar[1]['input_space_headers'].index(inInKey)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            inDict[key][i] = np.atleast_1d(histVar[1]['input_space_values'][ix])[0]
          elif inOutKey != None and inInKey == None:
            ix = histVar[1]['output_space_headers'].index(inOutKey)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            if inputPivotVal != None:
              if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
              inDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))
            else: inDict[key][i] = histVar[0][inputRow,ix]
          else: self.raiseAnError(IOError,'the parameter ' + key + ' has not been found')
        else:
          inKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inKey is not None:
            ix = histVar[1]['output_space_headers'].index(inKey)
            if i == 0: inDict[key] = np.zeros(len(hist_list))
            if inputPivotVal != None:
              if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
              inDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))
            else: inDict[key][i] = histVar[0][inputRow,ix]
          else: self.raiseAnError(IOError,'the parameter ' + key + ' has not been found in '+str(histVar[1]))
      # outputPivotVal end case => PointSet is at the final status
      if outputPivotVal_end:
        if all_out_param:
          for key in histVar[1]['output_space_headers']:
            if i == 0: outDict[key] = np.zeros(len(hist_list))
            outDict[key][i] = histVar[0][-1,histVar[1]['output_space_headers'].index(key)]
        else:
          for key in outParam:
            if key in histVar[1]['output_space_headers'] or \
               utils.toBytes(key) in histVar[1]['output_space_headers']:
              if i == 0: outDict[key] = np.zeros(len(hist_list))
              if key in histVar[1]['output_space_headers']: outDict[key][i] = histVar[0][-1,histVar[1]['output_space_headers'].index(key)]
              else: outDict[key][i] = histVar[0][-1,histVar[1]['output_space_headers'].index(utils.toBytes(key))]
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      elif outputRow != None:
        if outputRow > histVar[0][:,0].size-1  and outputRow != -1: self.raiseAnError(IOError,'outputRow is greater than number of actual rows in Database '+ str(self.name) + '!')
        if all_out_param:
          for key in histVar[1]['output_space_headers']:
            if i == 0: outDict[key] = np.zeros(len(hist_list))
            outDict[key][i] = histVar[0][outputRow,histVar[1]['output_space_headers'].index(key)]
        else:
          for key in outParam:
            if key in histVar[1]['output_space_headers'] or \
               utils.toBytes(key) in histVar[1]['output_space_headers']:
              if i == 0: outDict[key] = np.zeros(len(hist_list))
              if key in histVar[1]['output_space_headers']: outDict[key][i] = histVar[0][outputRow,histVar[1]['output_space_headers'].index(key)]
              else: outDict[key][i] = histVar[0][outputRow,histVar[1]['output_space_headers'].index(utils.toBytes(key))]
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      elif operator != None:
        if operator not in ['max','min','average']: self.raiseAnError(IOError,'operator unknown. Available are min,max,average')
        if histVar[1]['output_space_headers']:
          for key in histVar[1]['output_space_headers']:
            if i == 0: outDict[key] = np.zeros(len(hist_list))
            if operator == 'max'    : outDict[key][i] = np.max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
            if operator == 'min'    : outDict[key][i] = np.min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
            if operator == 'average': outDict[key][i] = np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
        else:
          for key in outParam:
            if key in histVar[1]['output_space_headers'] or \
               utils.toBytes(key) in histVar[1]['output_space_headers']:
              if i == 0: outDict[key] = np.zeros(len(hist_list))
              if key in histVar[1]['output_space_headers']:
                if operator == 'max'    : outDict[key][i] = np.max(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
                if operator == 'min'    : outDict[key][i] = np.min(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
                if operator == 'average': outDict[key][i] = np.average(histVar[0][:,histVar[1]['output_space_headers'].index(key)])
              else:
                if operator == 'max'    : outDict[key][i] = np.max(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])
                if operator == 'min'    : outDict[key][i] = np.min(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])
                if operator == 'average': outDict[key][i] = np.average(histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))])
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      else:
        # Arbitrary point in outputPivotVal case... If the requested outputPivotVal point Set does not match any of the stored ones and
        # start_outputPivotVal <= requested_outputPivotVal_point <= end_outputPivotVal, compute an interpolated value
        if all_out_param:
          for key in histVar[1]['output_space_headers']:
            if i == 0: outDict[key] = np.zeros(len(hist_list))
            outDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(key)], kind='linear')(outputPivotVal)
        else:
          for key in outParam:
            if i == 0: outDict[key] = np.zeros(len(hist_list))
            if key in histVar[1]['output_space_headers'] or \
               utils.toBytes(key) in histVar[1]['output_space_headers']:
              if key in histVar[1]['output_space_headers']: outDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(key)], kind='linear')(outputPivotVal)
              else                                        : outDict[key][i] = interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(utils.toBytes(key))], kind='linear')(outputPivotVal)
            else                                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      del histVar
    # return tuple of PointSet
    return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

  def __retrieveDataHistory(self,attributes):
    """
    Function to retrieve a History from the HDF5 database
    @ In, attributes : dictionary of attributes (variables and history name must be retrieved)
    @ Out, tupleVar  : tuple in which the first position is a dictionary of numpy arays (input variable)
    and the second is a dictionary of the numpy arrays (output variables).
    Note: This function retrieve a History from an HDF5 database
    """
    # Check the outParam variables and the outputPivotVal filters

    inParam, outParam, inputRow                 = attributes['inParam'], attributes['outParam'], attributes.get('inputRow',None)
    inputPivotVal, outputPivotVal               = attributes.get('inputPivotValue',None), attributes.get('outputPivotValue',None)
    pivotParameter                              = attributes.get('pivotParameter',None)
    if 'all' in outParam: all_out_param = True
    else                : all_out_param = False
    if outputPivotVal != None:
      if 'all' in outputPivotVal: outputPivotVal_all = True
      else:
        outputPivotVal_all, outputPivotVal = False,  [float(x) for x in outputPivotVal.split()]
    else: outputPivotVal_all = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None : inputRow  = int(inputRow) - 1
    inDict  = {}
    outDict = {}
    metaDict= {}
    # Call the function to retrieve a single history and
    # load the data into the tuple
    histVar = self.returnHistory(attributes)
    if pivotParameter != None:
      pivotIndex = histVar[1]['output_space_headers'].index(pivotParameter) if pivotParameter in histVar[1]['output_space_headers'] else None
      if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in Database '+ str(self.name) + '!')
    else:
      pivotIndex = histVar[1]['output_space_headers'].index("time") if "time" in histVar[1]['output_space_headers'] else None
      # if None...default is 0
      if pivotIndex == None: pivotIndex = 0

    if 'metadata' in histVar[1].keys(): metaDict[0] = histVar[1]['metadata']
    else                              : metaDict[0] = None
    # fill input param dictionary
    for key in inParam:
      if 'input_space_headers' in histVar[1]:
        inInKey = utils.keyIn(histVar[1]['input_space_headers'],key)
        inOutKey = utils.keyIn(histVar[1]['output_space_headers'],key)
        if inInKey != None:
          ix = histVar[1]['input_space_headers'].index(inInKey)
          inDict[key] = np.atleast_1d(np.array(histVar[1]['input_space_values'][ix]))
        elif inOutKey != None and inInKey == None:
          ix = histVar[1]['output_space_headers'].index(inOutKey)
          if inputPivotVal != None:
            if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
            inDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))))
          else: inDict[key] = np.atleast_1d(np.array(histVar[0][inputRow,ix]))
        else: self.raiseAnError(RuntimeError,'the parameter ' + key + ' has not been found in '+str(histVar[1]['input_space_headers'])+' or '+str(histVar[1]['output_space_headers']))
      else:
        inOutKey = utils.keyIn(histVar[1]['output_space_headers'],key)
        if inOutKey is not None:
          ix = histVar[1]['output_space_headers'].index(inOutKey)
          if inputPivotVal != None:
            if float(inputPivotVal) > np.max(histVar[0][:,pivotIndex]) or float(inputPivotVal) < np.min(histVar[0][:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in Database '+ str(self.name) + '!')
            inDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,ix], kind='linear')(float(inputPivotVal))))
          else: inDict[key] = np.atleast_1d(np.array(histVar[0][inputRow,ix]))
        else: self.raiseAnError(RuntimeError,'the parameter ' + key + ' has not been found in '+str(histVar[1]['output_space_headers']))

    #  all case => The history is completed (from start_time to end_time)
    if outputPivotVal_all:
      if all_out_param:
        for key in histVar[1]['output_space_headers']:
          outDict[key] = histVar[0][:,histVar[1]['output_space_headers'].index(key)]
      else:
        for key in outParam:
          inKey = utils.keyIn(histVar[1]['output_space_headers'],key)
          if inKey:
            outDict[key] = histVar[0][:,histVar[1]['output_space_headers'].index(inKey)]
          else:
            self.raiseAnError(RuntimeError,'the parameter ' + key + ' has not been found in '+str(histVar[1]['output_space_headers']))
    else:
      if all_out_param:
        for key in histVar[1]['output_space_headers']:
          outDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(key)], kind='linear')(outputPivotVal)))
      else:
        for key in outParam:
          if key in histVar[1]['output_space_headers']:
            outDict[key] = np.atleast_1d(np.array(interp1d(histVar[0][:,pivotIndex], histVar[0][:,histVar[1]['output_space_headers'].index(key)], kind='linear')(outputPivotVal)))
          else: self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    # Return tuple of dictionaries containing the HistorySet
    return (copy.copy(inDict),copy.copy(outDict),copy.copy(metaDict))

  def retrieveData(self,attributes):
    """
    Function interface for retrieving a Point or PointSet or History from the HDF5 database
    @ In, attributes : dictionary of attributes (variables, history name,metadata must be retrieved)
    @ Out, data     : tuple in which the first position is a dictionary of numpy arays (input variable)
    and the second is a dictionary of the numpy arrays (output variables).
    Note: Interface function
    """
    if attributes['type'] == 'Point':      data = self.__retrieveDataPoint(attributes)
    elif attributes['type'] == 'PointSet': data = self.__retrieveDataPointSet(attributes)
    elif attributes['type'] == 'History':      data = self.__retrieveDataHistory(attributes)
    elif attributes['type'] == 'HistorySet':
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
    else: self.raiseAnError(RuntimeError,'Type' + attributes['type'] +' unknown.Caller: hdf5Manager.retrieveData')
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

def returnInstance(Type,runInfoDict,caller):
  """
  Function interface for creating an instance to a database specialized class (for example, HDF5)
  @ In, type                : class type (string)
  @ Out, class Instance     : instance to that class
  Note: Interface function
  """
  try: return __interFaceDict[Type](runInfoDict)
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
