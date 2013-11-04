'''
Created on Feb 16, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
from BaseType import BaseType
from Csv_loader import CsvLoader as ld
import DataBases
import copy
import numpy as np
import utils

# Custom exceptions
class NotConsistentData(Exception):
    pass
class ConstructError(Exception):
    pass
#from hdf5_manager import hdf5Manager as AAFManager
#import h5py as h5

class Data(BaseType):
  def __init__(self,inParamValues = None, outParamValues = None):
    BaseType.__init__(self)
    self.dataParameters = {}                # in here we store all the data parameters (inputs params, output params,etc) 
    if inParamValues: 
      if type(inParamValues) != 'dict':
        raise ConstructError('DATAS     : ERROR ->  in __init__  in Datas of type ' + self.type + ' . inParamValues is not a dictionary')
      self.inpParametersValues = inParamValues
    else:
      self.inpParametersValues   = {}         # input parameters as keys, corresponding values 
    if outParamValues: 
      if type(outParamValues) != 'dict':
        raise ConstructError('DATAS     : ERROR ->  in __init__  in Datas of type ' + self.type + ' . outParamValues is not a dictionary')
      self.outParametersValues = outParamValues
    self.outParametersValues   = {}         # output variables as keys, corresponding values
    self.toLoadFromList = []                # loading source
    
  def readMoreXML(self,xmlNode):
    print('here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # retrieve input parameters' keywords
    self.dataParameters['inParam']  = xmlNode.find('Input' ).text.strip().split(',')
    # retrieve output parameters' keywords
    self.dataParameters['outParam'] = xmlNode.find('Output').text.strip().split(',')
    # retrieve history name if present
    try:   self.dataParameters['history'] = xmlNode.find('Input' ).attrib['name']
    except KeyError:self.dataParameters['history'] = None
    try:
      # check if time information are present... in case, store it
      time = xmlNode.attrib['time']
      if time == 'end' or time == 'all': self.dataParameters['time'] = time 
      else:
        try:   self.dataParameters['time'] = float(time)
        except:self.dataParameters['time'] = float(time.split(','))
    except KeyError:self.dataParameters['time'] = None
    try:
      self.print_CSV = bool(xmlNode.attrib['printCSV'])
    except KeyError:self.print_CSV = False
    
    try:
      self.CSVfilename = xmlNode.attrib['CSVfilename']    
    except KeyError:self.CSVfilename = None

  def addInitParams(self,tempDict):
    for i in range(len(self.dataParameters['inParam' ])):  tempDict['Input_'+str(i)]  = self.dataParameters['inParam' ][i]
    for i in range(len(self.dataParameters['outParam'])): tempDict['Output_'+str(i)] = self.dataParameters['outParam'][i]
    tempDict['Time'] = self.dataParameters['time']
    return tempDict
  
  def removeInputValue(self,name,value):
    if name in self.inpParametersValues.keys(): self.inpParametersValues.pop(name)
   
  def removeOutputValue(self,name,value):
    if name in self.outParametersValues.keys(): self.outParametersValues.pop(name)
  
  def updateInputValue(self,name,value):
    self.updateSpecializedInputValue(name,value)

  def updateOutputValue(self,name,value):
    self.updateSpecializedOutputValue(name,value)

  def addSpecializedReadingSettings(self):
    '''
      This function is used to add specialized attributes to the data in order to retrieve the data properly.
      Every specialized data needs to overwrite it!!!!!!!!
    '''
    raise NotImplementedError('DATAS     : ERROR -> The data of type '+self.type+' seems not to have a addSpecializedReadingSettings method overloaded!!!!')

  def checkConsistency(self):
    '''
      This function checks the consistency of the data structure... every specialized data needs to overwrite it!!!!!
    '''
    raise NotImplementedError('DATAS     : ERROR -> The data of type '+self.type+' seems not to have a checkConsistency method overloaded!!!!')

  def printCSV(self):
    # print content of data in a .csv format
    if self.debug:
      print('=======================')
      print('DATAS: print on file(s)')
      print('=======================')

    if (self.CSVfilename):
      filenameLocal = self.CSVfilename
    else:
      filenameLocal = self.name + '_dump'
    self.specializedPrintCSV(filenameLocal)

  def addOutput(self,toLoadFrom):
    ''' 
        this function adds the file name/names/object to the
        filename list + it calls the specialized functions to retrieve the different data
    '''
    print('DATAS       : toLoadFrom -> ')
    print(toLoadFrom)
    self.toLoadFromList.append(toLoadFrom)
    self.addSpecializedReadingSettings()
    sourceType = None
    try:    sourceType =  self.toLoadFromList[0].type
    except AttributeError: pass
    
    if(sourceType == 'HDF5'): tupleVar = self.toLoadFromList[0].retrieveData(self.dataParameters)
    else:                     tupleVar = ld().csvLoadData(self.toLoadFromList,self.dataParameters) 
    self.inpParametersValues = copy.deepcopy(tupleVar[0])
    self.outParametersValues = copy.deepcopy(tupleVar[1])
    self.checkConsistency()
    return

  def getInpParametersValues(self):
    return self.inpParametersValues  

  def getOutParametersValues(self):
    return self.outParametersValues 
  
  def getParam(self,typeVar,keyword):
    if typeVar == "input":
      if keyword in self.inpParametersValues.keys(): return self.inpParametersValues[keyword]
      else: raise Exception("DATAS     : ERROR -> parameter " + keyword + " not found in inpParametersValues dictionary. Function: Data.getParam")    
    elif typeVar == "output":
      if keyword in self.outParametersValues.keys(): return self.outParametersValues[keyword]    
      else: raise Exception("DATAS     : ERROR -> parameter " + keyword + " not found in outParametersValues dictionary. Function: Data.getParam")
    else: raise Exception("DATAS     : ERROR -> type " + typeVar + " is not a valid type. Function: Data.getParam")

class TimePoint(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except: sourceType = None
    if('HDF5' == sourceType):
      if(not self.dataParameters['history']): raise IOError('DATAS     : ERROR -> DATAS     : ERROR: In order to create a TimePoint data, history name must be provided')
      self.dataParameters['filter'] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data TimePoint
    '''
    for key in self.inpParametersValues.keys():
      if (self.inpParametersValues[key].size) != 1:
        raise NotConsistentData('DATAS     : ERROR -> The input parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self.inpParametersValues[key])))
    for key in self.outParametersValues.keys():
      if (self.outParametersValues[key].size) != 1:
        raise NotConsistentData('DATAS     : ERROR -> The output parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self.outParametersValues[key])))

  def updateSpecializedInputValue(self,name,value):
    if name in self.inpParametersValues.keys():
      self.inpParametersValues.pop(name)
    self.inpParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def updateSpecializedOutputValue(self,name,value):
    if name in self.inpParametersValues.keys():
      self.outParametersValues.pop(name)
    self.outParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal):
    file = open(filenameLocal + '.csv', 'wb')
    
    #Print input values
    inpKeys   = self.inpParametersValues.keys()
    inpValues = self.inpParametersValues.values()
    for i in range(len(inpKeys)):
      file.write(',' + inpKeys[i])
    file.write('\n')
    
    for i in range(len(inpKeys)):
      file.write(',' + str(inpValues[i][0]))
    file.write('\n')
    
    #Print time + output values
    outKeys   = self.outParametersValues.keys()
    outValues = self.outParametersValues.values()
    for i in range(len(outKeys)):
      file.write(',' + outKeys[i])
    file.write('\n')
    
    for i in range(len(outKeys)):
      file.write(',' + str(outValues[i][0]))
    file.write('\n')
    
    file.close()
    
class TimePointSet(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except: sourceType = None
    if('HDF5' == sourceType):
      self.dataParameters['histories'] = self.toLoadFromList[0].getEndingGroupNames()
      self.dataParameters['filter'   ] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data TimePointSet
    '''
    
    try:   sourceType = self.toLoadFromList[0].type
    except:sourceType = None
    if('HDF5' == sourceType):
      eg = self.toLoadFromList[0].getEndingGroupNames()
      for key in self.inpParametersValues.keys():
        if (self.inpParametersValues[key].size) != len(eg):
          raise NotConsistentData('DATAS     : ERROR -> The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self.inParametersValues[key].size))
      for key in self.outParametersValues.keys():
        if (self.outParametersValues[key].size) != len(eg):
          raise NotConsistentData('DATAS     : ERROR -> The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self.outParametersValues[key].size))
    else: 
      for key in self.inpParametersValues.keys():
        if (self.inpParametersValues[key].size) != len(self.toLoadFromList):
          raise NotConsistentData('DATAS     : ERROR -> The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(self.toLoadFromList)) + '.Actual size is ' + str(self.inParametersValues[key].size))
      for key in self.outParametersValues.keys():
        if (self.outParametersValues[key].size) != len(self.toLoadFromList):
          raise NotConsistentData('DATAS     : ERROR -> The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(self.toLoadFromList)) + '.Actual size is ' + str(self.outParametersValues[key].size))

  def updateSpecializedInputValue(self,name,value):
    if name in self.inpParametersValues.keys():
      popped = self.inpParametersValues.pop(name)
      self.inpParametersValues[name] = copy.deepcopy(np.concatenate((np.atleast_1d(np.array(popped)), np.atleast_1d(np.array(value)))))
    else:
      self.inpParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def updateSpecializedOutputValue(self,name,value):
    if name in self.outParametersValues.keys():
      popped = self.outParametersValues.pop(name)
      self.outParametersValues[name] = copy.deepcopy(np.concatenate((np.array(popped), np.atleast_1d(np.array(value)))))
    else:
      self.outParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal): 
    
    inpKeys   = list(self.inpParametersValues.keys())
    inpValues = list(self.inpParametersValues.values())
    
    outKeys   = list(self.outParametersValues.keys())
    outValues = list(self.outParametersValues.values())
    myFile = open(filenameLocal + '.csv', 'wb')
    myFile.write(b'counter')
    for i in range(len(inpKeys)):
        myFile.write(b',' + utils.toBytes(inpKeys[i]))
    for i in range(len(outKeys)):
        myFile.write(b',' + utils.toBytes(outKeys[i]))
    myFile.write(b'\n')
    
    for j in range(outValues[0].size):
      myFile.write(utils.toBytes(str(j+1)))
      for i in range(len(inpKeys)):
        myFile.write(b',' + utils.toBytes(str(inpValues[i][j])))
      for i in range(len(outKeys)):
        myFile.write(b',' + utils.toBytes(str(outValues[i][j])))
      myFile.write(b'\n')
      
    myFile.close()

class History(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except: sourceType = None
    if('HDF5' == sourceType):
      if(not self.dataParameters['history']): raise IOError('DATAS     : ERROR -> In order to create a History data, history name must be provided')
      self.dataParameters['filter'] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data History
    '''
    for key in self.inpParametersValues.keys():
      if (self.inpParametersValues[key].size) != 1:
        raise NotConsistentData('DATAS     : ERROR -> The input parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self.inpParametersValues[key])))
    for key in self.outParametersValues.keys():
      if (self.outParametersValues[key].ndim) != 1:
        raise NotConsistentData('DATAS     : ERROR -> The output parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be an 1D array.' + '.Actual dimension is ' + str(self.outParametersValues[key].ndim))

  def updateSpecializedInputValue(self,name,value):
    if name in self.inpParametersValues.keys():
      self.inpParametersValues.pop(name)
    self.inpParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def updateSpecializedOutputValue(self,name,value):
    if name in self.outParametersValues.keys():
      self.outParametersValues.pop(name)
    self.outParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal):
    file = open(filenameLocal + '.csv', 'wb')
    
    #Print input values
    inpKeys   = self.inpParametersValues.keys()
    inpValues = self.inpParametersValues.values()
    for i in range(len(inpKeys)):
      file.write(',' + inpKeys[i])
    file.write('\n')
    
    for i in range(len(inpKeys)):
      file.write(',' + str(inpValues[i][0]))
    file.write('\n')
    
    #Print time + output values
    outKeys   = self.outParametersValues.keys()
    outValues = self.outParametersValues.values()
    for i in range(len(outKeys)):
      file.write(',' + outKeys[i])
    file.write('\n')

    for j in range(outValues[0].size):
      for i in range(len(outKeys)):
        file.write(',' + str(outValues[i][j]))
      file.write('\n')
    
    file.close()

class Histories(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      self.dataParameters['filter'   ] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data Histories
    '''
    try: sourceType = self.toLoadFromList[0].type
    except AttributeError: sourceType = None
    
    if('HDF5' == sourceType):
      eg = self.toLoadFromList[0].getEndingGroupNames()
      if(len(eg) != len(self.inpParametersValues.keys())):
        raise NotConsistentData('DATAS     : ERROR -> Number of Histories contained in Histories data ' + self.name + ' != number of loading sources!!! ' + str(len(eg)) + ' !=' + str(len(self.inpParametersValues.keys())))
    else:
      if(len(self.toLoadFromList) != len(self.inpParametersValues.keys())):
        raise NotConsistentData('DATAS     : ERROR -> Number of Histories contained in Histories data ' + self.name + ' != number of loading sources!!! ' + str(len(self.toLoadFromList)) + ' !=' + str(len(self.inpParametersValues.keys())))
    for key in self.inpParametersValues.keys():
      for key2 in self.inpParametersValues[key].keys():
        if (self.inpParametersValues[key][key2].size) != 1:
          raise NotConsistentData('DATAS     : ERROR -> The input parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in Histories ' +self.name+ '!! It should be a single value.' + '.Actual size is ' + str(len(self.inpParametersValues[key][key2])))
    for key in self.outParametersValues.keys():
      for key2 in self.outParametersValues[key].keys():
        if (self.outParametersValues[key][key2].ndim) != 1:
          raise NotConsistentData('DATAS     : ERROR -> The output parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in Histories ' +self.name+ '!! It should be an 1D array.' + '.Actual dimension is ' + str(self.outParametersValues[key][key2].ndim))

  def updateSpecializedInputValue(self,name,value):
    if (not isinstance(value,(float,int,bool,np.ndarray))):
      raise NotConsistentData('DATAS     : ERROR -> Histories Data accepts only a numpy array (dim 1) or a single value for method "updateSpecializedInputValue". Got type ' + str(type(value)))
    if isinstance(value,np.ndarray): 
      if value.size != 1: raise NotConsistentData('DATAS     : ERROR -> Histories Data accepts only a numpy array of dim 1 or a single value for method "updateSpecializedInputValue". Size is ' + str(value.size))
    
    if type(name) == 'list':
      # there are info regarding the history number
      if name[0] in self.inpParametersValues.keys():
        gethistory = self.inpParametersValues.pop(name[0])
        popped = gethistory[name[1]]
        if name[1] in popped.keys():
          gethistory[name[1]] = copy.deepcopy(np.atleast_1d(np.array(value)))
          self.inpParametersValues[name[0]] = copy.deepcopy(gethistory)
      else:
        self.inpParametersValues[name[0]] = copy.deepcopy({name[1]:np.atleast_1d(np.array(value))})
    else:
      # no info regarding the history number => use internal counter
      if len(self.inpParametersValues.keys()) == 0: self.inpParametersValues[1] = copy.deepcopy({name:np.atleast_1d(np.array(value))})
      else:
        hisn = max(self.inpParametersValues.keys())
        if name in list(self.inpParametersValues.values())[-1]: 
          hisn += 1
          self.inpParametersValues[hisn] = {}
        self.inpParametersValues[hisn][name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def updateSpecializedOutputValue(self,name,value):
    if not isinstance(value,np.ndarray): raise NotConsistentData('DATAS     : ERROR -> Histories Data accepts only numpy array as type for method "updateSpecializedOutputValue". Got ' + str(type(value)))
    if type(name) == 'list':
      # there are info regarding the history number
      if name[0] in self.outParametersValues.keys():
        gethistory = self.outParametersValues.pop(name[0])
        popped = gethistory[name[1]]
        if name[1] in popped.keys():
          gethistory[name[1]] = copy.deepcopy(np.atleast_1d(np.array(value)))
          self.outParametersValues[name[0]] = copy.deepcopy(gethistory)
      else:
        self.outParametersValues[name[0]] = copy.deepcopy({name[1]:np.atleast_1d(np.array(value))})
    else:
      # no info regarding the history number => use internal counter
      if len(self.outParametersValues.keys()) == 0: self.outParametersValues[1] = copy.deepcopy({name:np.atleast_1d(np.array(value))})
      else:
        hisn = max(self.outParametersValues.keys())
        if name in list(self.outParametersValues.values())[-1]: 
          hisn += 1
          self.outParametersValues[hisn] = {}
        self.outParametersValues[hisn][name] = copy.deepcopy(np.atleast_1d(np.array(value)))
      
  def specializedPrintCSV(self,filenameLocal):
    
    inpKeys   = self.inpParametersValues.keys()
    inpValues = list(self.inpParametersValues.values())
    outKeys   = self.outParametersValues.keys()
    outValues = list(self.outParametersValues.values())
    
    for n in range(len(outKeys)):
      file = open(filenameLocal + '_'+ str(n) + '.csv', 'wb')
  
      inpKeys_h   = list(inpValues[n].keys())
      inpValues_h = list(inpValues[n].values())
      outKeys_h   = list(outValues[n].keys())
      outValues_h = list(outValues[n].values())


      for i in range(len(inpKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        file.write(prefix + utils.toBytes(inpKeys_h[i]))
      file.write(b'\n')
      
      for i in range(len(inpKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        file.write(prefix + utils.toBytes(str(inpValues_h[i][0])))
      file.write(b'\n')
      
      #Print time + output values
      for i in range(len(outKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        file.write(utils.toBytes(outKeys_h[i]) + b',')
      file.write(b'\n')
  
      for j in range(outValues_h[0].size):
        for i in range(len(outKeys_h)):
          if i == 0 : prefix = b''
          else:       prefix = b','
          file.write(prefix+ utils.toBytes(str(outValues_h[i][j])))
        file.write(b'\n')    
      
      file.close()
   
'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'Data'
__interFaceDict                 = {}
__interFaceDict['TimePoint'   ] = TimePoint
__interFaceDict['TimePointSet'] = TimePointSet
__interFaceDict['History'     ] = History
__interFaceDict['Histories'   ] = Histories
__knownTypes                    = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  







  
