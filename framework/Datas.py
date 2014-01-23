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
    '''
    Function to read the xml input block.
    @ In, xmlNode, xml node
    '''
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
      self.dataParameters['input_ts'] = int(xmlNode.attrib['input_ts'])
    except KeyError:self.dataParameters['input_ts'] = None
    
  def addInitParams(self,tempDict):
    '''
    Function to get the input params that belong to this class
    @ In, tempDict, temporary dictionary
    '''
    for i in range(len(self.dataParameters['inParam' ])):  tempDict['Input_'+str(i)]  = self.dataParameters['inParam' ][i]
    for i in range(len(self.dataParameters['outParam'])): tempDict['Output_'+str(i)] = self.dataParameters['outParam'][i]
    tempDict['Time'] = self.dataParameters['time']
    return tempDict
  
  def removeInputValue(self,name):
    '''
    Function to remove a value from the dictionary inpParametersValues
    @ In, name, parameter name
    '''
    if name in self.inpParametersValues.keys(): self.inpParametersValues.pop(name)
   
  def removeOutputValue(self,name):
    '''
    Function to remove a value from the dictionary outParametersValues
    @ In, name, parameter name
    '''
    if name in self.outParametersValues.keys(): self.outParametersValues.pop(name)
  
  def updateInputValue(self,name,value):
    '''
    Function to update a value from the dictionary inParametersValues
    @ In, name, parameter name
    @ In, value, the new value
    '''
    self.updateSpecializedInputValue(name,value)

  def updateOutputValue(self,name,value):
    '''
    Function to update a value from the dictionary outParametersValues
    @ In, name, parameter name
    @ In, value, the new value
    '''
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

  def printCSV(self,options=None):
    '''
    Function used to dump the data into a csv file
    Every class must implement the specializedPrintCSV method
    that is going to be called from here
    @ In, OPTIONAL, options, dictionary of options... it can contain the filename to be used, the parameters need to be printed....
    '''
    options_int = {}
    # print content of data in a .csv format
    if self.debug:
      print('=======================')
      print('DATAS: print on file(s)')
      print('=======================')
    if options:
      if ('filenameroot' in options.keys()): filenameLocal = options['filenameroot']
      else: filenameLocal = self.name + '_dump'
      if 'variables' in options.keys():
        variables_to_print = []
        for var in options['variables'].split(','):
          if   var.lower() == 'input' : 
            if type(list(self.inpParametersValues.values())[0]) == dict: 
              for invar in list(self.inpParametersValues.values())[0].keys(): variables_to_print.append('input|'+str(invar))  
            else: 
              for invar in self.inpParametersValues.keys(): variables_to_print.append('input|'+str(invar))
          elif var.lower() == 'output': 
            if type(list(self.outParametersValues.values())[0]) == dict:
              for outvar in list(self.outParametersValues.values())[0].keys(): variables_to_print.append('output|'+str(outvar))  
            else:
              for outvar in self.outParametersValues.keys(): variables_to_print.append('output|'+str(outvar))
          elif '|' in var:
            if var.split('|')[0].lower() == 'input':
              if type(list(self.inpParametersValues.values())[0]) == dict:
                if var.split('|')[1] not in list(self.inpParametersValues.values())[0].keys(): raise Exception("DATAS     : ERROR -> variable " + var.split('|')[1] + " is not present among the Inputs of Data " + self.name)
                else: variables_to_print.append('input|'+str(var.split('|')[1]))
              else:
                if var.split('|')[1] not in self.inpParametersValues.keys(): raise Exception("DATAS     : ERROR -> variable " + var.split('|')[1] + " is not present among the Inputs of Data " + self.name)
                else: variables_to_print.append('input|'+str(var.split('|')[1]))
            elif var.split('|')[0].lower() == 'output':
              if type(list(self.outParametersValues.values())[0]) == dict:
                if var.split('|')[1] not in list(self.outParametersValues.values())[0].keys(): raise Exception("DATAS     : ERROR -> variable " + var.split('|')[1] + " is not present among the Outputs of Data " + self.name)
                else: variables_to_print.append('output|'+str(var.split('|')[1]))
              else:
                if var.split('|')[1] not in self.outParametersValues.keys(): raise Exception("DATAS     : ERROR -> variable " + var.split('|')[1] + " is not present among the Outputs of Data " + self.name)
                else: variables_to_print.append('output|'+str(var.split('|')[1]))
          else: raise Exception("DATAS     : ERROR -> variable " + var + " is unknown in Data " + self.name + ". You need to specify an input or a output")
        options_int['variables'] = variables_to_print            
    else:   filenameLocal = self.name + '_dump'
    
    self.specializedPrintCSV(filenameLocal,options_int)

  def addOutput(self,toLoadFrom):
    ''' 
      Function to construct a data from a source
      @ In, toLoadFrom, loading source, it can be an HDF5 database, a csv file and in the future a xml file
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
    self.checkConsistency()
    return

  def getParametersValues(self,typeVar):
    '''
    Functions to get the parameter values
    @ In, variable type (input or output)
    '''
    if typeVar.lower() == "input":    return self.getInpParametersValues()
    elif typeVar.lower() == "output": return self.getOutParametersValues()
    else: raise Exception("DATAS     : ERROR -> type " + typeVar + " is not a valid type. Function: Data.getParametersValues")

  def getInpParametersValues(self):
    '''
    Function to get a reference to the input parameter dictionary
    @, In, None
    @, Out, Reference to self.inpParametersValues
    '''
    return self.inpParametersValues  

  def getOutParametersValues(self):
    '''
    Function to get a reference to the output parameter dictionary
    @, In, None
    @, Out, Reference to self.outParametersValues
    '''
    return self.outParametersValues 
  
  def getParam(self,typeVar,keyword):
    '''
    Function to get a reference to an output or input parameter
    @ In, typeVar, input or output
    @ In, keyword, keyword 
    @ Out, Reference to the parameter
    '''
    if typeVar.lower() == "input":
      if keyword in self.inpParametersValues.keys(): return self.inpParametersValues[keyword]
      else: raise Exception("DATAS     : ERROR -> parameter " + keyword + " not found in inpParametersValues dictionary. Function: Data.getParam")    
    elif typeVar.lower() == "output":
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
    else: raise Exception('the variable named '+varName+' was not found in the data: '+self.name)
    return self.__extractValueLocal__(myType,inOutType,varTyp,varName,varID,stepID)
  
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
        raise NotConsistentData('DATAS     : ERROR -> The input parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(self.inpParametersValues[key].size))
    for key in self.outParametersValues.keys():
      if (self.outParametersValues[key].size) != 1:
        raise NotConsistentData('DATAS     : ERROR -> The output parameter value, for key ' + key + ' has not a consistent shape for TimePoint ' + self.name + '!! It should be a single value.' + '.Actual size is ' + str(self.outParametersValues[key].size))

  def updateSpecializedInputValue(self,name,value):
    if name in self.inpParametersValues.keys():
      self.inpParametersValues.pop(name)
    self.inpParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def updateSpecializedOutputValue(self,name,value):
    if name in self.inpParametersValues.keys():
      self.outParametersValues.pop(name)
    self.outParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal,options):
    
    inpKeys   = []
    inpValues = []
    outKeys   = []
    outValues = []
    #Print input values
    if 'variables' in options.keys():
      for var in options['variables']:
        if var.split('|')[0] == 'input': 
          inpKeys.append(var.split('|')[1])
          inpValues.append(self.inpParametersValues[var.split('|')[1]])
        if var.split('|')[0] == 'output': 
          outKeys.append(var.split('|')[1])
          outValues.append(self.outParametersValues[var.split('|')[1]])
    else:
      inpKeys   = self.inpParametersValues.keys()
      inpValues = self.inpParametersValues.values()
      outKeys   = self.outParametersValues.keys()
      outValues = self.outParametersValues.values()
    
    if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'wb')
    else: return
    
    for item in inpKeys:
      myFile.write(b',' + utils.toBytes(item))
    if len(inpKeys) > 0: myFile.write(b'\n')
    
    for item in inpValues:
      myFile.write(b',' + utils.toBytes(str(item[0])))
    if len(inpValues) > 0: myFile.write(b'\n')
    
    #Print time + output values
    for item in outKeys:
      myFile.write(b',' + utils.toBytes(item))
    if len(outKeys) > 0: myFile.write(b'\n')
    
    for item in outValues:
      myFile.write(b',' + utils.toBytes(str(item[0])))
    if len(outValues) > 0: myFile.write(b'\n')
    
    myFile.close()
  
  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None):
    '''override of the method in the base class Datas'''
    if varID!=None or stepID!=None: raise Exception('seeking to extract a slice from a TimePoint type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':exec ('return '+varTyp+'(self.getParam(inOutType,varName)[0])')
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
      if name not in self.dataParameters['inParam']: self.dataParameters['inParam'].append(name)
      self.inpParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def updateSpecializedOutputValue(self,name,value):
    if name in self.outParametersValues.keys():
      popped = self.outParametersValues.pop(name)
      self.outParametersValues[name] = copy.deepcopy(np.concatenate((np.array(popped), np.atleast_1d(np.array(value)))))
    else:
      if name not in self.dataParameters['outParam']: self.dataParameters['outParam'].append(name)
      self.outParametersValues[name] = copy.deepcopy(np.atleast_1d(np.array(value)))

  def specializedPrintCSV(self,filenameLocal,options): 
    inpKeys   = []
    inpValues = []
    outKeys   = []
    outValues = []
    #Print input values
    if 'variables' in options.keys():
      for var in options['variables']:
        if var.split('|')[0] == 'input': 
          inpKeys.append(var.split('|')[1])
          inpValues.append(self.inpParametersValues[var.split('|')[1]])
        if var.split('|')[0] == 'output': 
          outKeys.append(var.split('|')[1])
          outValues.append(self.outParametersValues[var.split('|')[1]])
    else:
      inpKeys   = list(self.inpParametersValues.keys())
      inpValues = list(self.inpParametersValues.values())
      outKeys   = list(self.outParametersValues.keys())
      outValues = list(self.outParametersValues.values())
    
    if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'wb')
    else: return

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
    if stepID!=None: raise Exception('seeking to extract a history slice over an TimePointSet type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':
      if varID!=None: 
        exec('aa ='+varTyp +'(self.getParam(inOutType,varName)[varID])')
        return aa
      #if varID!=None: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[varID])')
      else: raise Exception('trying to extract a scalar value from a time point set without an index')
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

  def specializedPrintCSV(self,filenameLocal,options):
    inpKeys   = []
    inpValues = []
    outKeys   = []
    outValues = []
    #Print input values
    if 'variables' in options.keys():
      for var in options['variables']:
        if var.split('|')[0] == 'input': 
          inpKeys.append(var.split('|')[1])
          inpValues.append(self.inpParametersValues[var.split('|')[1]])
        if var.split('|')[0] == 'output': 
          outKeys.append(var.split('|')[1])
          outValues.append(self.outParametersValues[var.split('|')[1]])
    else:
      inpKeys   = self.inpParametersValues.keys()
      inpValues = self.inpParametersValues.values()
      outKeys   = self.outParametersValues.keys()
      outValues = self.outParametersValues.values()
    
    if len(inpKeys) > 0 or len(outKeys) > 0: myFile = open(filenameLocal + '.csv', 'wb')
    else: return

    for i in range(len(inpKeys)):
      myFile.write(b',' + utils.toBytes(inpKeys[i]))
    if len(inpKeys) > 0: myFile.write(b'\n')
    
    for i in range(len(inpKeys)):
      myFile.write(b',' + utils.toBytes(str(inpValues[i][0])))
    if len(inpKeys) > 0: myFile.write(b'\n')
    
    #Print time + output values
    for i in range(len(outKeys)):
      myFile.write(b',' + utils.toBytes(outKeys[i]))
    if len(outKeys) > 0: 
      myFile.write(b'\n')
      for j in range(outValues[0].size):
        for i in range(len(outKeys)):
          myFile.write(b',' + utils.toBytes(str(outValues[i][j])))
        myFile.write(b'\n')
    
    myFile.close()

  def __extractValueLocal__(self,myType,inOutType,varTyp,varName,varID=None,stepID=None):
    '''override of the method in the base class Datas'''
    if varID!=None: raise Exception('seeking to extract a slice over number of parameters an History type of data is not possible. Data name: '+self.name+' variable: '+varName)
    if varTyp!='numpy.ndarray':
      if varName in self.dataParameters['inParam']: exec ('return varTyp(self.getParam('+inOutType+','+varName+')[0])')
      else:
        if stepID!=None and type(stepID)!=tuple: exec ('return self.getParam('+inOutType+','+varName+')['+str(stepID)+']')
        else: raise Exception('To extract a scalar from an history a step id is needed. Variable: '+varName+', Data: '+self.name)
    else:
      if stepID==None : return self.getParam(inOutType,varName)
      elif stepID!=None and type(stepID)==tuple: return self.getParam(inOutType,varName)[stepID[0]:stepID[1]]
      else: raise Exception('trying to extract variable '+varName+' from '+self.name+' the id coordinate seems to be incoherent: stepID='+str(stepID))
    

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
      
  def specializedPrintCSV(self,filenameLocal,options):
    
    inpValues = list(self.inpParametersValues.values())
    outKeys   = self.outParametersValues.keys()
    outValues = list(self.outParametersValues.values())
    
    for n in range(len(outKeys)):
      inpKeys_h   = []
      inpValues_h = []
      outKeys_h   = []
      outValues_h = []
      if 'variables' in options.keys():
        for var in options['variables']:
          if var.split('|')[0] == 'input': 
            inpKeys_h.append(var.split('|')[1])
            inpValues_h.append(inpValues[n][var.split('|')[1]])
          if var.split('|')[0] == 'output': 
            outKeys_h.append(var.split('|')[1])
            outValues_h.append(outValues[n][var.split('|')[1]])
      else:
        inpKeys_h   = list(inpValues[n].keys())
        inpValues_h = list(inpValues[n].values())
        outKeys_h   = list(outValues[n].keys())
        outValues_h = list(outValues[n].values())
    
      if len(inpKeys_h) > 0 or len(outKeys_h) > 0: myFile = open(filenameLocal + '_'+ str(n) + '.csv', 'wb')
      else: return
      
      for i in range(len(inpKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        myFile.write(prefix + utils.toBytes(inpKeys_h[i]))
      if len(inpKeys_h) > 0: myFile.write(b'\n')
      
      for i in range(len(inpKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        myFile.write(prefix + utils.toBytes(str(inpValues_h[i][0])))
      if len(inpKeys_h) > 0: myFile.write(b'\n')
      
      #Print time + output values
      for i in range(len(outKeys_h)):
        if i == 0 : prefix = b''
        else:       prefix = b','
        myFile.write(utils.toBytes(outKeys_h[i]) + b',')
      if len(outKeys_h) > 0:
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
        else: raise Exception('to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID to identify the history (varID missed)')
      else:
        if varID!=None:
          if stepID!=None and type(stepID)!=tuple: exec ('return varTyp(self.getParam('+inOutType+','+str(varID)+')[varName][stepID]')
          else: raise Exception('to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used and a time coordinate (time or timeID missed or tuple)')
        else: raise Exception('to extract a scalar ('+varName+') form the data '+self.name+', it is needed an ID of the input set used (varID missed)')
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
          if stepID==None: raise Exception('more info needed trying to extract '+varName+' from data '+self.name)
          elif type(stepID)==tuple:
            if stepID[1]!=None:
              myOut=np.zeros((len(self.getOutParametersValues().keys()),stepID[1]-stepID[0]))
              for key in self.getOutParametersValues().keys():
                myOut[int(key),:]=self.getParam(inOutType,key)[varName][stepID[0]:stepID[1]]
            else: raise Exception('more info needed trying to extract '+varName+' from data '+self.name)
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

