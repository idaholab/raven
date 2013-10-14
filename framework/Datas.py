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
#from hdf5_manager import hdf5Manager as AAFManager
#import h5py as h5

class Data(BaseType):
  def __init__(self,inParamValues = None, outParamValues = None):
    BaseType.__init__(self)
    self.dataParameters = {}                # in here we store all the data parameters (inputs params, output params,etc) 
    if inParamValues: 
      if type(inParamValues) != 'dict':
        raise ConstructError('ERROR in __init__  in Datas of type ' + self.type + ' . inParamValues is not a dictionary')
      self.inpParametersValues = inParamValues
    else:
      self.inpParametersValues   = {}         # input parameters as keys, corresponding values 
    if outParamValues: 
      if type(outParamValues) != 'dict':
        raise ConstructError('ERROR in __init__  in Datas of type ' + self.type + ' . outParamValues is not a dictionary')
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
    except:self.dataParameters['history'] = None
    try:
      # check if time information are present... in case, store it
      time = xmlNode.attrib['time']
      if time == 'end' or time == 'all': self.dataParameters['time'] = time 
      else:
        try:   self.dataParameters['time'] = float(time)
        except:self.dataParameters['time'] = float(time.split(','))
    except:self.dataParameters['time'] = None
    try:
      self.print_CSV = bool(xmlNode.attrib['printCSV'])
    except:self.print_CSV = False
    
    try:
      self.CSVfilename = xmlNode.attrib['CSVfilename']    
    except:self.CSVfilename = None

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
    raise NotImplementedError('The data of type '+self.type+' seems not to have a addSpecializedReadingSettings method overloaded!!!!')

  def checkConsistency(self):
    '''
      This function checks the consistency of the data structure... every specialized data needs to overwrite it!!!!!
    '''
    raise NotImplementedError('The data of type '+self.type+' seems not to have a checkConsistency method overloaded!!!!')

  def printCSV(self):
    # print content of data in a .csv format
    print('=======================')
    print('DATAS: print on file(s)')
    print('=======================')
    
    if (self.print_CSV):
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
    except: pass
    
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
      else: raise Exception("parameter " + keyword + " not found in inpParametersValues dictionary. Function: Data.getParam")    
    elif typeVar == "output":
      if keyword in self.outParametersValues.keys(): return self.outParametersValues[keyword]    
      else: raise Exception("parameter " + keyword + " not found in outParametersValues dictionary. Function: Data.getParam")
    else: raise Exception("type " + typeVar + " is not a valid type. Function: Data.getParam")

class TimePoint(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except: sourceType = None
    if('HDF5' == sourceType):
      if(not self.dataParameters['history']): raise IOError('DATAS     : ERROR: In order to create a TimePoint data, history name must be provided')
      self.dataParameters['filter'] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data TimePoint
    '''
    for key in self.inpParametersValues.keys():
      if (self.inpParametersValues[key].size) != 1:
        raise NotConsistentData('The input parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self.inpParametersValues[key])))
    for key in self.outParametersValues.keys():
      if (self.outParametersValues[key].size) != 1:
        raise NotConsistentData('The output parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self.outParametersValues[key])))

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
          raise NotConsistentData('The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self.inParametersValues[key].size))
      for key in self.outParametersValues.keys():
        if (self.outParametersValues[key].size) != len(eg):
          raise NotConsistentData('The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self.outParametersValues[key].size))
    else: 
      for key in self.inpParametersValues.keys():
        if (self.inpParametersValues[key].size) != len(self.toLoadFromList):
          raise NotConsistentData('The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(self.toLoadFromList)) + '.Actual size is ' + str(self.inParametersValues[key].size))
      for key in self.outParametersValues.keys():
        if (self.outParametersValues[key].size) != len(self.toLoadFromList):
          raise NotConsistentData('The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(self.toLoadFromList)) + '.Actual size is ' + str(self.outParametersValues[key].size))

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
    
    inpKeys   = self.inpParametersValues.keys()
    inpValues = self.inpParametersValues.values()
    
    outKeys   = self.outParametersValues.keys()
    outValues = self.outParametersValues.values()
    myFile = open(filenameLocal + '.csv', 'wb')
    myFile.write('counter')
    for i in range(len(inpKeys)):
        myFile.write(',' + inpKeys[i])
    for i in range(len(outKeys)):
        myFile.write(',' + outKeys[i])
    myFile.write('\n')
    
    for j in range(outValues[0].size):
      myFile.write(str(j))
      for i in range(len(inpKeys)):
        myFile.write(',' + str(inpValues[i][j]))
      for i in range(len(outKeys)):
        myFile.write(',' + str(outValues[i][j]))
      myFile.write('\n')
      
    myFile.close()

class History(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try: sourceType = self.toLoadFromList[0].type
    except: sourceType = None
    if('HDF5' == sourceType):
      if(not self.dataParameters['history']): raise IOError('DATAS     : ERROR: In order to create a History data, history name must be provided')
      self.dataParameters['filter'] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data History
    '''
    for key in self.inpParametersValues.keys():
      if (self.inpParametersValues[key].size) != 1:
        raise NotConsistentData('The input parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(len(self.inpParametersValues[key])))
    for key in self.outParametersValues.keys():
      if (self.outParametersValues[key].ndim) != 1:
        raise NotConsistentData('The output parameter value, for key ' + key + ' has not a consistent shape for History ' + self.name + '!! It should be an 1D array.' + '.Actual dimension is ' + str(self.outParametersValues[key].ndim))

  def updateSpecializedInputValue(self,name,value):
    if name in self.inpParametersValues.keys():
      popped = self.inpParametersValues.pop(name)
      self.inpParametersValues[name] = copy.deepcopy(np.concatenate((np.array(popped), np.atleast_1d(np.array(value)))))
    else:
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
    except: sourceType = None
    if('HDF5' == sourceType):
      self.dataParameters['filter'   ] = "whole"

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data Histories
    '''
    try: sourceType = self.toLoadFromList[0].type
    except: sourceType = None
    
    if('HDF5' == sourceType):
      eg = self.toLoadFromList[0].getEndingGroupNames()
      if(len(eg) != len(self.inpParametersValues.keys())):
        raise NotConsistentData('Number of Histories contained in Histories data ' + self.name + ' != number of loading sources!!! ' + str(len(eg)) + ' !=' + str(len(self.inpParametersValues.keys())))
    else:
      if(len(self.toLoadFromList) != len(self.inpParametersValues.keys())):
        raise NotConsistentData('Number of Histories contained in Histories data ' + self.name + ' != number of loading sources!!! ' + str(len(self.toLoadFromList)) + ' !=' + str(len(self.inpParametersValues.keys())))
    for key in self.inpParametersValues.keys():
      for key2 in self.inpParametersValues[key].keys():
        if (self.inpParametersValues[key][key2].size) != 1:
          raise NotConsistentData('The input parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in Histories ' +self.name+ '!! It should be a single value.' + '.Actual size is ' + str(len(self.inpParametersValues[key][key2])))
    for key in self.outParametersValues.keys():
      for key2 in self.outParametersValues[key].keys():
        if (self.outParametersValues[key][key2].ndim) != 1:
          raise NotConsistentData('The output parameter value, for key ' + key2 + ' has not a consistent shape for History ' + key + ' contained in Histories ' +self.name+ '!! It should be an 1D array.' + '.Actual dimension is ' + str(self.outParametersValues[key][key2].ndim))

  def updateSpecializedInputValue(self,name,value):
    #FIXME... The Actual structure is not ok.. fix it
    pass

  def updateSpecializedOutputValue(self,name,value):
    #FIXME... The Actual structure is not ok.. fix it
    pass

  def specializedPrintCSV(self,filenameLocal):
    
    inpKeys   = self.inpParametersValues.keys()
    inpValues = self.inpParametersValues.values()
    outKeys   = self.outParametersValues.keys()
    outValues = self.outParametersValues.values()
    
    for n in range(len(outKeys)):
      file = open(filenameLocal + '_'+ str(n) + '.csv', 'wb')
  
      inpKeys_h   = inpValues[n].keys()
      inpValues_h = inpValues[n].values()
      outKeys_h   = outValues[n].keys()
      outValues_h = outValues[n].values()


      for i in range(len(inpKeys_h)):
        file.write(',' + inpKeys_h[i])
      file.write('\n')
      
      for i in range(len(inpKeys_h)):
        file.write(',' + str(inpValues_h[i]))
      file.write('\n')
      
      #Print time + output values
      for i in range(len(outKeys_h)):
        file.write(outKeys_h[i] + ',')
      file.write('\n')
  
      for j in range(outValues_h[0].size):
        for i in range(len(outKeys_h)):
          file.write(',' + str(outValues_h[i][j]))
        file.write('\n')    
      
      file.close()
   
'''
 Interface Dictionary (factory) (private)
'''

base = 'Data'
__InterfaceDict = {}
__InterfaceDict['TimePoint'   ] = TimePoint
__InterfaceDict['TimePointSet'] = TimePointSet
__InterfaceDict['History'     ] = History
__InterfaceDict['Histories'   ] = Histories

def returnInstance(Type):
  try:
    if Type in __InterfaceDict.keys():
      return __InterfaceDict[Type]()
  except:
    raise NameError('not known '+base+' type'+Type)
  
# 
  
