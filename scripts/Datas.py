'''
Created on Feb 16, 2013

@author: alfoa
'''
#import CsvLoader as loader 
import xml.etree.ElementTree as ET
from BaseType import BaseType

class Data(BaseType):
  def __init__(self):
    BaseType.__init__(self)
    self.inputs  = []   # input  parameters
    self.outputs = []   # output parameters
    self.inpParametersValues   = {}  # input parameters as keys, corresponding values 
    self.outParametersValues   = {}  # output variables as keys, corresponding values    
  def readMoreXML(self,xmlNode):
    self.inputs  = xmlNode.find('Input' ).text.split(',')
    self.outputs = xmlNode.find('Output').text.split(',')
    try:
      time = xmlNode.attrib['time']
      if time == 'end':
        self.time = time
      else:
        try: self.time = float(time)
        except:self.time = float(time.split(','))
    except:self.time = 'end'
    # set keys into dictionaries of values
    for key in self.inputs:
      self.inpParametersValues[key] = None
    for key in self.outputs:
      self.outParametersValues[key] = None    
  def addInitParams(self,tempDict):
    counter = 0
    for key in self.inpParametersValues.keys(): 
      tempDict['Input'+str(counter)] = key
      counter += 1
    counter = 0
    for key in self.outParametersValues.keys(): 
      tempDict['Output'+str(counter)] = key
      counter += 1
    tempDict['Time'] = self.time

  def load(self):
    return #.loaderDictionary(self.type)

class TimePoint(Data):
  pass

class TimePointSet(TimePoint):
  def load(self,fileNameRoot,numberSimulation):
    # we construct the list of files from which the data must be collected
    files = []
    for iSims in numberSimulation:
      files[iSims] = fileNameRoot + '_' + str(iSims)  + '.csv'   
    Data.load(files,self.time_filter,self.inpParametersValues,self.outParametersValues)

class History(Data):
  pass

class Histories(Data):
  def __init__(self):
    Data.__init__(self)
    self.vectorOfHistory = []
    self.vectorOfHistory.append(History())
    self.vectorOfHistory[0].name                = self.name
    self.vectorOfHistory[0].type                = self.type
    self.vectorOfHistory[0].inputs              = self.inputs
    self.vectorOfHistory[0].outputs             = self.outputs
    self.vectorOfHistory[0].inpParametersValues = self.inpParametersValues
    self.vectorOfHistory[0].outParametersValues = self.outParametersValues
  def load(self,fileNameRoot,numberSimulation):
    # we create a list of History type (size() == numberSimulation)
    for iSims in xrange(numberSimulation):
      self.vectorOfHistory.append(self.vectorOfHistory[0])
      filename = fileNameRoot + '_' + str(iSims)  + '.csv'
      Data.load(filename,self.vectorOfHistory[iSims].time_filter,self.vectorOfHistory[iSims].inpParametersValues,self.vectorOfHistory[iSims].outVariableValues)

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
  
  
  
