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
#from hdf5_manager import hdf5Manager as AAFManager
#import h5py as h5

class Data(BaseType):
  def __init__(self):
    BaseType.__init__(self)
    self.dataParameters = {}                # in here we store all the data parameters (inputs params, output params,etc) 
    self.inpParametersValues   = {}         # input parameters as keys, corresponding values 
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

  def addInitParams(self,tempDict):
    for i in range(len(self.dataParameters['inParam' ])):  tempDict['Input_'+str(i)]  = self.dataParameters['inParam' ][i]
    for i in range(len(self.dataParameters['outParam'])): tempDict['Output_'+str(i)] = self.dataParameters['outParam'][i]
    tempDict['Time'] = self.dataParameters['time']
    return tempDict
  
  def addSpecializedReadingSettings(self):
    raise NotImplementedError('The data of type '+self.type+' seems not to have a addSpecializedReadingSettings method overloaded!!!!')

  def checkConsistency(self):
    raise NotImplementedError('The data of type '+self.type+' seems not to have a checkConsistency method overloaded!!!!')

  def addOutput(self,toLoadFrom):
    # this function adds the file name/names to the
    # filename list
    print('DATAS       : toLoadFrom -> ')
    print(toLoadFrom)
    self.toLoadFromList.append(toLoadFrom)
    self.addSpecializedReadingSettings()
    sourceType = None
    try:    sourceType =  self.toLoadFromList[0].type
    except: pass
    
    if(sourceType == 'HDF5'): tupleVar = self.toLoadFromList[0].retrieveData(self.dataParameters)
    else:                     tupleVar = ld().csvLoadData(self.toLoadFromList,self.dataParameters) 
    self.inpParametersValues = tupleVar[0]
    self.outParametersValues = tupleVar[1]
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
    try:    
      if('HDF5' in self.toLoadFromList[0].type):
         if(not self.dataParameters['history']): raise IOError('DATAS     : ERROR: In order to create a TimePoint data, history name must be provided')
         self.dataParameters['filter'] = "whole"
    except: 
      pass

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

class TimePointSet(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try:    
      if('HDF5' in self.toLoadFromList[0].type):
         self.dataParameters['histories'] = self.toLoadFromList[0].getEndingGroupNames()
         self.dataParameters['filter'   ] = "whole"
    except: 
      pass

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data TimePointSet
    '''
    try:    
      if('HDF5' in self.toLoadFromList[0].type):
        eg = self.toLoadFromList[0].getEndingGroupNames()
        for key in self.inpParametersValues.keys():
          if (self.inpParametersValues[key].size) != len(eg):
            raise NotConsistentData('The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self.inParametersValues[key].size))
        for key in self.outParametersValues.keys():
          if (self.outParametersValues[key].size) != len(eg):
            raise NotConsistentData('The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(eg)) + '.Actual size is ' + str(self.outParametersValues[key].size))
    except: 
      for key in self.inpParametersValues.keys():
        if (self.inpParametersValues[key].size) != len(self.toLoadFromList):
          raise NotConsistentData('The input parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(self.toLoadFromList)) + '.Actual size is ' + str(self.inParametersValues[key].size))
      for key in self.outParametersValues.keys():
        if (self.outParametersValues[key].size) != len(self.toLoadFromList):
          raise NotConsistentData('The output parameter value, for key ' + key + ' has not a consistent shape for TimePointSet ' + self.name + '!! It should be an array of size ' + str(len(self.toLoadFromList)) + '.Actual size is ' + str(self.outParametersValues[key].size))

class History(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try:    
      if('HDF5' in self.toLoadFromList[0].type):
         if(not self.dataParameters['history']): raise IOError('DATAS     : ERROR: In order to create a History data, history name must be provided')
         self.dataParameters['filter'] = "whole"
    except: 
      pass

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


class Histories(Data):
  def addSpecializedReadingSettings(self):
    self.dataParameters['type'] = self.type # store the type into the dataParameters dictionary
    try:    
      if('HDF5' in self.toLoadFromList[0].type):
         self.dataParameters['filter'   ] = "whole"
    except: 
      pass

  def checkConsistency(self):
    '''
      Here we perform the consistency check for the structured data Histories
    '''
    try:    
      if('HDF5' in self.toLoadFromList[0].type):
        eg = self.toLoadFromList[0].getEndingGroupNames()
        if(len(eg) != len(self.inpParametersValues.keys())):
          raise NotConsistentData('Number of Histories contained in Histories data ' + self.name + ' != number of loading sources!!! ' + str(len(eg)) + ' !=' + str(len(self.inpParametersValues.keys())))
    except:
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
   
def returnInstance(Type):
  base = 'Data'
  InterfaceDict = {}
  InterfaceDict['TimePoint'   ] = TimePoint
  InterfaceDict['TimePointSet'] = TimePointSet
  InterfaceDict['History'     ] = History
  InterfaceDict['Histories'   ] = Histories
  try:
    if Type in InterfaceDict.keys():
      return InterfaceDict[Type]()
  except:
    raise NameError('not known '+base+' type'+Type)
  
  
  
