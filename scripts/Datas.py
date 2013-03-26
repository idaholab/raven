'''
Created on Feb 16, 2013

@author: alfoa
'''
import xml.etree.ElementTree as ET
from BaseType import BaseType
import CsvLoader as ld 
#import h5py as h5


class Data(BaseType):
  def __init__(self):
    BaseType.__init__(self)
    self.inputs  = []   # input  parameters
    self.outputs = []   # output parameters
    self.inpParametersValues   = {}  # input parameters as keys, corresponding values 
    self.outParametersValues   = {}  # output variables as keys, corresponding values
    self.filenames = []
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
    return 
  def addOutput(self,filename):
    # this function adds the file name/names to the
    # filename list
    self.filenames.append(filename)
    
  def getInpParametersValues(self):
    return self.inpParametersValues  

  def getOutParametersValues(self):
    return self.outParametersValues 

class TimePoint(Data):
  def finalizeOutput(self):
    tuple = ld.csvLoaderForTimePoint(self.filenames[0],self.time,self.inputs,self.outputs)
    self.inpParametersValues = tuple[0]
    self.outParametersValues = tuple[1]
    
class TimePointSet(Data):
  def finalizeOutput(self,fileNameRoot,numberSimulation):
    tuple = ld.csvLoaderForTimePointSet(self.filenames,self.time,self.inputs,self.outputs)
    self.inpParametersValues = tuple[0]
    self.outParametersValues = tuple[1]

class History(Data):
  def finalizeOutput(self):
    tuple = ld.csvLoaderForHistory(self.filenames[0],self.time,self.inputs,self.outputs)
    self.inpParametersValues = tuple[0]
    self.outParametersValues = tuple[1]

class Histories(Data):
  def __init__(self):
    Data.__init__(self)
#    self.vectorOfHistory = []
  def finalizeOutput(self):
    for ifiles in len(self.filenames):
      tuple = ld.csvLoaderForHistory(self.filenames[ifiles],self.time,self.inputs,self.outputs)
      self.vectorOfHistory.append(History())
      # dictionary of dictionary key = i => ith history ParameterValues dictionary
      self.inpParametersValues[ifiles] = tuple[0]
      self.inpParametersValues[ifiles] = tuple[1]
#      self.vectorOfHistory[ifiles].inpParametersValues = tuple[0]
#      self.vectorOfHistory[ifiles].outParametersValues = tuple[1]
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
  
  
  
