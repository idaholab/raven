'''
Created on Feb 16, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
from BaseType import BaseType
from Csv_loader import CsvLoader as ld
import copy
import abc
import numpy as np
import utils

# Custom exceptions
class NotConsistentData(Exception):
    pass
class ConstructError(Exception):
    pass
#from hdf5_manager import hdf5Manager as AAFManager
#import h5py as h5

class Data(utils.metaclass_insert(abc.ABCMeta,BaseType)):
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
        except ValueError: self.dataParameters['time'] = float(time.split(','))
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

  @abc.abstractmethod
  def addSpecializedReadingSettings(self):
    '''
      This function is used to add specialized attributes to the data in order to retrieve the data properly.
      Every specialized data needs to overwrite it!!!!!!!!
    '''
    pass

  @abc.abstractmethod
  def checkConsistency(self):
    '''
      This function checks the consistency of the data structure... every specialized data needs to overwrite it!!!!!
    '''
    pass

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
    self.toLoadFromList.append(toLoadFrom)
    self.addSpecializedReadingSettings()
    sourceType = None
    print('DATAS         : Constructiong data type "' +self.type +'" named "'+ self.name + '" from:')
    try:    
      sourceType =  self.toLoadFromList[-1].type
      print(' '*16 +'Object type "' + self.toLoadFromList[-1].type + '" named "' + self.toLoadFromList[-1].name+ '"')
    except AttributeError: 
      print(' '*16 +'CSV "' + toLoadFrom + '"')
  
    if(sourceType == 'HDF5'): tupleVar = self.toLoadFromList[-1].retrieveData(self.dataParameters)
    #else:                     tupleVar = ld().csvLoadData(self.toLoadFromList,self.dataParameters) 
    else:                     tupleVar = ld().csvLoadData([toLoadFrom],self.dataParameters) 
    
    for hist in tupleVar[0].keys():
      if type(tupleVar[0][hist]) == dict:
        for key in tupleVar[0][hist].keys(): self.updateInputValue(key, tupleVar[0][hist][key])
      else: self.updateInputValue(hist, tupleVar[0][hist])
    for hist in tupleVar[1].keys():
      if type(tupleVar[1][hist]) == dict:
        for key in tupleVar[1][hist].keys(): self.updateOutputValue(key, tupleVar[1][hist][key])
      else: self.updateOutputValue(hist, tupleVar[1][hist]) 
    #self.inpParametersValues = copy.deepcopy(tupleVar[0])
    #self.outParametersValues = copy.deepcopy(tupleVar[1])
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
    
  def extractValue(self,varTyp,varName,varID=None,stepID=None):
    '''
    this a method that is used to extract a value (both array or scalar) attempting an implicit conversion for scalars
    the value is returned without link to the original
    @in varType is the requested type of the variable to be returned (bool, int, float, numpy.ndarray, etc)
    @in varName is the name of the variable that should be recovered
    @in varID is the ID of the value that should be retrieved within a set
      if varID.type!=tuple only one point along sampling of that variable is retrieved
        else:
          if varID=(int,int) the slicing is [varID[0]:varID[1]]
          if varID=(int,None) the slicing is [varID[0]:]
    @in stepID determine the slicing of an history.
        if stepID.type!=tuple only one point along the history is retrieved
        else:
          if stepID=(int,int) the slicing is [stepID[0]:stepID[1]]
          if stepID=(int,None) the slicing is [stepID[0]:]
    '''
    myType=self.type
    if   varName in self.dataParameters['inParam' ]: inOutType = 'input'
    elif varName in self.dataParameters['outParam']: inOutType = 'output'
    else: raise 'the variable named '+varName+' was not found in the data: '+self.name
    self.__extractValueLocal__(myType,inOutType,varTyp,varName,varID=None,stepID=None)
  
  @abc.abstractmethod
  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None):
    '''this method has to be override to implement the specialization of extractValue for each data class'''
    pass

class TimePoint(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except AttributeError: sourceType = None
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
    myFile = open(filenameLocal + '.csv', 'wb')
    
    #Print input values
    inpKeys   = self.inpParametersValues.keys()
    inpValues = self.inpParametersValues.values()
    for i in range(len(inpKeys)):
      myFile.write(',' + inpKeys[i])
    myFile.write('\n')
    
    for i in range(len(inpKeys)):
      myFile.write(',' + str(inpValues[i][0]))
    myFile.write('\n')
    
    #Print time + output values
    outKeys   = self.outParametersValues.keys()
    outValues = self.outParametersValues.values()
    for i in range(len(outKeys)):
      myFile.write(',' + outKeys[i])
    myFile.write('\n')
    
    for i in range(len(outKeys)):
      myFile.write(',' + str(outValues[i][0]))
    myFile.write('\n')
    
    myFile.close()
  
  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None):
    '''override of the method in the base class Datas'''
    if varID!=None or stepID!=None: raise 'seeking to extract a slice from a TimePoint type of data is not possible. Data name: '+self.name+' variable: '+varName
    if varTyp!='numpy.ndarray':exec ('return varTyp(self.getParam(inOutType,'+varName+')[0])')
    else: return self.getParam(inOutType,varName)

class TimePointSet(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except AttributeError: sourceType = None
    if('HDF5' == sourceType):
      self.dataParameters['histories'] = self.toLoadFromList[0].getEndingGroupNames()
      self.dataParameters['filter'   ] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data TimePointSet
    '''
    try:   sourceType = self.toLoadFromList[0].type
    except AttributeError:sourceType = None
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

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None):
    '''override of the method in the base class Datas'''
    if stepID!=None: raise 'seeking to extract a history slice over an TimePointSet type of data is not possible. Data name: '+self.name+' variable: '+varName    
    if varTyp!='numpy.ndarray':
      if varID!=None: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[varID]')
      else: raise 'trying to extract a scalar value from a time point set without an index'
    else: return self.getParam(inOutType,varName)


class History(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except AttributeError: sourceType = None
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
    myFile = open(filenameLocal + '.csv', 'wb')
    
    #Print input values
    inpKeys   = self.inpParametersValues.keys()
    inpValues = self.inpParametersValues.values()
    for i in range(len(inpKeys)):
      myFile.write(',' + inpKeys[i])
    myFile.write('\n')
    
    for i in range(len(inpKeys)):
      myFile.write(',' + str(inpValues[i][0]))
    myFile.write('\n')
    
    #Print time + output values
    outKeys   = self.outParametersValues.keys()
    outValues = self.outParametersValues.values()
    for i in range(len(outKeys)):
      myFile.write(',' + outKeys[i])
    myFile.write('\n')

    for j in range(outValues[0].size):
      for i in range(len(outKeys)):
        myFile.write(',' + str(outValues[i][j]))
      myFile.write('\n')
    
    myFile.close()

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None):
    '''override of the method in the base class Datas'''
    if varID!=None: raise 'seeking to extract a slice over number of parameters an History type of data is not possible. Data name: '+self.name+' variable: '+varName
    if varTyp!='numpy.ndarray':
      if varName in self.dataParameters['inParam']: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[0])')
      else:
        if stepID!=None and type(stepID)!=tuple: exec ('return self.getParam('+inOutType+','+varName+')['+str(stepID)+']')
        else: raise 'To extract a scalar from an history a step id is needed. Variable: '+varName+', Data: '+self.name
    else:
      if stepID==None : return self.getParam(inOutType,varName)
      elif stepID!=None and type(stepID)==tuple: return self.getParam(inOutType,varName)[stepID[0]:stepID[1]]
      else: raise 'trying to extract variable '+varName+' from '+self.name+' the id coordinate seems to be incoherent: stepID='+str(stepID)
    

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
    
    inpValues = list(self.inpParametersValues.values())
    outKeys   = self.outParametersValues.keys()
    outValues = list(self.outParametersValues.values())
    
    for n in range(len(outKeys)):
      myFile = open(filenameLocal + '_'+ str(n) + '.csv', 'wb')
  
      inpKeys_h   = list(inpValues[n].keys())
      inpValues_h = list(inpValues[n].values())
      outKeys_h   = list(outValues[n].keys())
      outValues_h = list(outValues[n].values())

      for i in range(len(inpKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        myFile.write(prefix + utils.toBytes(inpKeys_h[i]))
      myFile.write(b'\n')
      
      for i in range(len(inpKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        myFile.write(prefix + utils.toBytes(str(inpValues_h[i][0])))
      myFile.write(b'\n')
      
      #Print time + output values
      for i in range(len(outKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        myFile.write(utils.toBytes(outKeys_h[i]) + b',')
      myFile.write(b'\n')
  
      for j in range(outValues_h[0].size):
        for i in range(len(outKeys_h)):
          if i == 0 : prefix = b''
          else:       prefix = b','
          myFile.write(prefix+ utils.toBytes(str(outValues_h[i][j])))
        myFile.write(b'\n')    
      
      myFile.close()
      
  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None):
    '''override of the method in the base class Datas'''
    if varTyp!='numpy.ndarray':
      if varName in self.dataParameters['inParam']:
        if varID!=None: exec ('return varTyp(self.getParam('+inOutType+','+str(varID)+')[varName]')
        else: raise 'to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID to identify the history (varID missed)'
      else:
        if varID!=None:
          if stepID!=None and type(stepID)!=tuple: exec ('return varTyp(self.getParam('+inOutType+','+str(varID)+')[varName][stepID]')
          else: raise 'to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used and a time coordinate (time or timeID missed or tuple)'
        else: raise 'to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used (varID missed)'
    else:
      if varName in self.dataParameters['inParam']:
        myOut=np.zeros(len(self.getInpParametersValues().keys()))
        for key in self.getInpParametersValues().keys():
          myOut[int(key)]=self.getParam(inOutType,key)[varName][0]
        return myOut
      else:
        if varID!=None:
          if stepID==None:
            return self.getParam(inOutType,varID)[varName]
          elif type(stepID)==tuple:
            if stepID[1]==None: return self.getParam(inOutType,varID)[varName][stepID[0]:]
            else: return self.getParam(inOutType,varID)[varName][stepID[0]:stepID[1]]
          else: return self.getParam(inOutType,varID)[varName][stepID]
        else:
          if stepID==None: raise 'more info needed trying to extract '+varName+' from data '+self.name
          elif type(stepID)==tuple:
            if stepID[1]!=None:
              myOut=np.zeros((len(self.getOutParametersValues().keys()),stepID[1]-stepID[0]))
              for key in self.getOutParametersValues().keys():
                myOut[int(key),:]=self.getParam(inOutType,key)[varName][stepID[0]:stepID[1]]
            else: raise 'more info needed trying to extract '+varName+' from data '+self.name
          else:
            myOut=np.zeros(len(self.getOutParametersValues().keys()))
            for key in self.getOutParametersValues().keys():
              myOut[int(key)]=self.getParam(inOutType,key)[varName][stepID]
            return myOut
       
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

