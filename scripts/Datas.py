import CsvLoader as loader 

class TimePoint:
  '''
  one dimensional point (in the output space)
  characterized by its input values in the m-dimensional space (and might be time value)
  '''
  def __init__(self,name=None):
    if name:
      self.name = name
    else:
      self.name = None
    self.inpParametersValues = {}  # input parameters as keys, corresponding values (time is placed here eventually as a parameter)
    self.outVariableValues   = {}  # output variables as keys, corresponding values
    
  def load(self,filein,time,inpParamsDict,outParamsDict):
    
    self.inpParametersValues = inpParamsDict
    self.outVariableValues = outParamsDict
    
    loader.csvLoaderForTimePoint(filein,time,self.inpParametersValues,self.outVariableValues)
    
  def takePointfromHistories(self):
    return
  
  def takePointfromHistory(self):
    return
  
  def takePointfromPointSet(self):
    return

class TimePointSet:
  '''
  a set of n dimensional points (in the output space)
  characterized by its input values in the m-dimensional space (and might be time value)
  '''
  def __init__(self,name=None):
    if name:
      self.name = name
    else:
      self.name = None
    self.inpParametersValues = {}  #input parameters as keys, corresponding a vector of values (time is eventually placed here as a parameter)
    self.outVariableValues   = {}  #output variables as keys, corresponding values vectors
    
  def load(self,fileNameRoot,numberSimulation,time,inpParamsDict,outParamsDict):
    
    self.inpParametersValues = inpParamsDict
    self.outVariableValues = outParamsDict
    
    # we construct the list of files from which the data must be collected
    files = []
    for iSims in numberSimulation:
      files[iSims] = fileNameRoot + '_' + str(iSims)  + '.csv'   
      
    loader.csvLoaderForTimePointSet(files,time,self.inpParametersValues,self.outVariableValues)
    
    return
  def takePointSetfromHistories(self):
    return
  def generatePointSetfromPoints(self):
    return

      
class History:
  '''
  class used to store one time history of one n-dimensional trajectory in the phase space 
  '''
  def __init__(self,name=None):
    if name:
      self.name = name
    else:
      self.name = None
      
    self.outVariableValues   = {} #keys are the name of the recorded variable, they correspond to vectors of values
    self.inpParametersValues = {} #keys are the name of the input parameters, they correspond to the input values
    
  def load(self,fileName,inpParamsDict,outParamsDict,oneTrajectoryTimefilter=None):
    '''
    open and read the file
    filter the data if present oneTrajectoryTimefilter
    allocate numpy arrays and store the info
    place the keys and the arrays in the dictionary
    '''
    
    self.inpParametersValues = inpParamsDict
    self.outVariableValues = outParamsDict
    loader.csvLoaderForHistory(filein,oneTrajectoryTimefilter,self.inpParametersValues,self.outVariableValues)
    return 

class Histories:
  '''
  class used to store time histories of one n-dimensional trajectory in the phase space 
  '''
  def __init__(self,name=None):
    if name:
      self.name = name
    else:
      self.name = None
    self.containerOut = {} # key are the name of the recorded variable, they correspond to matrices of values (simulation numbers)x(time step)
    self.containerIn  = {}
    
  def load(self,fileNameRoot,numberSimulation,inpParamsDict,outParamsDict,TrajectorySetTimefilter=None):
    '''
    '''
    self.inpParametersValues = inpParamsDict
    self.outVariableValues = outParamsDict
    
    # we construct the list of files from which the data must be collected
    files = []
    for iSims in numberSimulation:
      files[iSims] = fileNameRoot + '_' + str(iSims)  + '.csv'   
      
    loader.csvLoaderForHistories(files,time,self.inpParametersValues,self.outVariableValues)
    
    return 
 
class DataInterface:
  def __init__(self):
    self.dataMap   = {}
    self.dataMap['Histories']    = Histories
    self.dataMap['History']      = History
    self.dataMap['TimePointSet'] = TimePointSet
    self.dataMap['TimePoint']    = TimePoint
    
         