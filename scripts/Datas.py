'''
Created on Feb 16, 2013

@author: alfoa
'''
import xml.etree.ElementTree as ET
from BaseType import BaseType
from Csv_loader import CsvLoader 
#import h5py as h5


class Data(BaseType):
  def __init__(self):
    BaseType.__init__(self)
    self.inputs  = []   # input  parameters
    self.outputs = []   # output parameters
    self.inpParametersValues   = {}  # input parameters as keys, corresponding values 
    self.outParametersValues   = {}  # output variables as keys, corresponding values
    self.toLoadFromList = []
  def readMoreXML(self,xmlNode):
    self.inputs  = xmlNode.find('Input' ).text.split(',')
    self.outputs = xmlNode.find('Output').text.split(',')
    try:
      time = xmlNode.attrib['time']
      if time == 'end' or time == 'all':
        self.time = time 
      else:
        try: self.time = float(time)
        except:self.time = float(time.split(','))
    except:self.time = None

  def addInitParams(self,tempDict):
    for i in len(self.inputs): 
      tempDict['Input_'+str(i)] = self.inputs[i]
    for i in len(self.outputs): 
      tempDict['Output_'+str(i)] = self.outputs[i]
    tempDict['Time'] = self.time
    return tempDict

  def finalizeOutput(self):
    pass 
  def addOutput(self,toLoadFrom):
    # this function adds the file name/names to the
    # filename list
    self.toLoadFromList.append(toLoadFrom)
    
  def getInpParametersValues(self):
    return self.inpParametersValues  

  def getOutParametersValues(self):
    return self.outParametersValues 
  
  def getParam(self,type,keyword):
    if type == "input":
      if keyword in self.inpParametersValues.keys():
        return self.inpParametersValues[keyword]
      else:
        raise("parameter " + keyword + 
              " not found in inpParametersValues dictionary. Function: Data.getParam")    
    elif type == "output":
      if keyword in self.outParametersValues.keys():
        return self.outParametersValues[keyword]    
      else:
        raise("parameter " + keyword + 
              " not found in outParametersValues dictionary. Function: Data.getParam")
    else:
      raise("type " + type + " is not a valid type. Function: Data.getParam")
class TimePoint(Data):
  def finalizeOutput(self):
    try:
      type = toLoadFromList[0].type
      #add here the specialization for loading from other source
    except:
      tuple = ld.csvLoaderForTimePoint(self.toLoadFromList[0],self.time,self.inputs,self.outputs)
      self.inpParametersValues = tuple[0]
      self.outParametersValues = tuple[1]
    
class TimePointSet(Data):
  def finalizeOutput(self):
    try:
      types = []
      types = toLoadFromList[:].type
      #add here the specialization for loading from other source
    except:      
      tuple = ld.csvLoaderForTimePointSet(self.toLoadFromList,self.time,self.inputs,self.outputs)
      self.inpParametersValues = tuple[0]
      self.outParametersValues = tuple[1]

class History(Data):
  def finalizeOutput(self):
    try:
      type = toLoadFromList[0].type
      #add here the specialization for loading from other source
    except:      
      tuple = ld.csvLoaderForHistory(self.toLoadFromList[0],self.time,self.inputs,self.outputs)
      self.inpParametersValues = tuple[0]
      self.outParametersValues = tuple[1]

class Histories(Data):
  def __init__(self):
    Data.__init__(self)
#    self.vectorOfHistory = []
  def finalizeOutput(self):
    try:
      type = toLoadFromList[0].type
      #add here the specialization for loading from other source
    except:  
      for index in len(self.toLoadFromList):
        tuple = ld.csvLoaderForHistory(self.toLoadFromList[index],self.time,self.inputs,self.outputs)
        self.vectorOfHistory.append(History())
        # dictionary of dictionary key = i => ith history ParameterValues dictionary
        self.inpParametersValues[index] = tuple[0]
        self.inpParametersValues[index] = tuple[1]
#        self.vectorOfHistory[index].inpParametersValues = tuple[0]
#        self.vectorOfHistory[index].outParametersValues = tuple[1]
        del tuple

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
  
  
  
