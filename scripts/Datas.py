import CsvLoader as loader 
import xml.etree.ElementTree as ET

class Data:
  def __init__(self,xmlNode):
    
    self.name    = " "  # name of this data (alias)
    self.inputs  = []   # input  parameters
    self.outputs = []   # output parameters
    self.inpParametersValues   = {}  # input parameters as keys, corresponding values 
    self.outParametersValues   = {}  # output variables as keys, corresponding values    
    '''
      Constructor for the template class
    '''
    self.readXml(xmlNode)
    '''
      Read the XML input in order to get the information needed by
      each Data type
    '''
  def readXml(self,xmlNode):
    
    xmlNode = ET.parse("lupo.xml")
    root = xmlNode.getroot()
    
    self.name = root.get("name")
    
    try:
      inputs = root.find("Input").text
      # token-ize the string
      self.inputs = inputs.split()
      
    except:
      raise IOError('not found element "Input" in data' + self)
    try:
      outputs = root.find("Output").text
      # token-ize the string
      self.outputs = outputs.split()
    except:
      raise IOError('not found element "Output" in data' + self)
  
    # set keys into dictionaries of values
    for key in self.inputs:
      self.inpParametersValues[key] = None
    for key in self.outputs:
      self.outParametersValues[key] = None    
    
  
class TimePoint(Data):
  '''
  one dimensional point (in the output space)
  characterized by its input values in the m-dimensional space (and might be time value)
  '''
  def __init__(self,xmlNode):
    # let's initialize the base class (the readXml
    # of the base class is called in the constructor)
    Data.__init__(self, xmlNode)
    # read the specialization of this particular data type
    self.readXml(xmlNode)
    
  def readXml(self,xmlNode):
    root = xmlNode.getroot()
    self.time_filter = root.get("Time")

  def load(self,filein):
    
    loader.csvLoaderForTimePoint(filein,self.time_filter,self.inpParametersValues,self.outParametersValues)
    
  def takePointfromHistories(self):
    return
  
  def takePointfromHistory(self):
    return
  
  def takePointfromPointSet(self):
    return

class TimePointSet(Data):
  '''
  a set of n dimensional points (in the output space)
  characterized by its input values in the m-dimensional space (and might be time value)
  '''
  def __init__(self,xmlNode):
    #"self.outVariableValues"    keys are the name of the recorded variable, 
    #                            they correspond to vectors of values (size()==n dimensional points)
    #"self.inpParametersValues"  keys are the name of the input parameters, 
    #                            they correspond to the input values (size()==n dimensional points)
    
    # let's initialize the base class (the readXml
    # of the base class is called in the constructor)
    Data.__init__(self, xmlNode)
    
    self.readXml(xmlNode)
    
  def readXml(self,xmlNode):
    root = xmlNode.getroot()
    self.time_filter = root.get("Time")
    
  def load(self,fileNameRoot,numberSimulation):
    # we construct the list of files from which the data must be collected
    files = []
    for iSims in numberSimulation:
      files[iSims] = fileNameRoot + '_' + str(iSims)  + '.csv'   
      
    loader.csvLoaderForTimePointSet(files,self.time_filter,self.inpParametersValues,self.outParametersValues)
    return
  def takePointSetfromHistories(self):
    return
  def generatePointSetfromPoints(self):
    return

      
class History(Data):
  '''
  class used to store one time history of one n-dimensional trajectory in the phase space 
  '''
  def __init__(self,xmlNode):
    #"self.outVariableValues"    keys are the name of the recorded variable, 
    #                            they correspond to vectors of values (size()==n time steps)
    #"self.inpParametersValues"  keys are the name of the input parameters, 
    #                            they correspond to the input values (size()==n time steps)

    # let's initialize the base class (the readXml
    # of the base class is called in the constructor)
    Data.__init__(self, xmlNode)
    
    self.readXml(xmlNode)
    
  def readXml(self,xmlNode):
    root = xmlNode.getroot()  
    self.time_filter = root.get("Time")    
            
  def load(self,fileName):
    '''
    open and read the file
    filter the data if present oneTrajectoryTimefilter
    allocate numpy arrays and store the info
    place the keys and the arrays in the dictionary
    '''
    loader.csvLoaderForHistory(filein,self.time_filter,self.inpParametersValues,self.outVariableValues)
    return 

class Histories(Data):
  '''
  class used to store time histories of one n-dimensional trajectory in the phase space 
  It contains a vector(list) of History data type
  '''
  def __init__(self,xmlNode):
    #"self.outVariableValues"    keys are the name of the recorded variable, 
    #                            they correspond to vectors of values (size()==n time steps)
    #"self.inpParametersValues"  keys are the name of the input parameters, 
    #                            they correspond to the input values (size()==n time steps)
    Data.__init__(self, xmlNode)
    
    self.readXml(xmlNode)
    
    self.vectorOfHistory = []
    self.vectorOfHistory.append(History(xmlNode))

  def readXml(xmlNode):
    root = xmlNode.getroot()      
    self.time_filter = root.get("Time")

  def load(self,fileNameRoot,numberSimulation):
    '''
    ''' 
    # we create a list of History type (size() == numberSimulation)
    for iSims in xrange(numberSimulation):
      self.vectorOfHistory.append(self.vectorOfHistory[0])
     
    for iSims in xrange(numberSimulation):
      filename = fileNameRoot + '_' + str(iSims)  + '.csv'
      loader.csvLoaderForHistory(filename,self.vectorOfHistory[iSims].time_filter,self.vectorOfHistory[iSims].inpParametersValues,self.vectorOfHistory[iSims].outVariableValues)
#      loader.csvLoaderForHistories(files,self.time_filter,self.inpParametersValues,self.outParametersValues)
    return 

def generateDataClass(xmlNode):
  '''
    This function generates an instance of class Data (specialized)
  '''
  dataTypeDictionary["TimePoint"]    = TimePoint
  dataTypeDictionary["TimePointSet"] = TimePointSet
  dataTypeDictionary["History"]      = History
  dataTypeDictionary["Histories"]    = Histories
  
  
  return dataTypeDictionary[xmlNode.tag](xmlNode)

         