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
    
  def load(self,filein,time,paramList=None):
    if time == 'end':
      return #get the ending state
    else:
      #read the time point with linear interpolation
      return
    
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
    
  def load(self,filein,time,number,paramList=None):
    if time == 'end':
      return #get the ending state
    else:
      #read the time point with linear interpolation
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

    self.container = {} #keys are the name of the recorded variable, they correspond to vectors of values
    self.input     = {} #keys are the name of the input parameters, they correspond to the input values
    
  def load(self,fileName,oneTrajectoryTimefilter=None):
    '''
    open and read the file
    filter the data if present oneTrajectoryTimefilter
    allocate numpy arrays and store the info
    place the keys and the arrays in the dictionary
    '''
    if filter:
      return
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
    self.container = {} # key are the name of the recorded variable, they correspond to matrices of values (simulation numbers)x(time step)
    
  def load(self,fileNameRoot,numberSimulation=None,TrajectorySetTimefilter=None):
    '''
    '''
    if filter:
      return
    return 
 
class DataInterface:
  def __init__(self):
    self.dataMap   = {}
    self.dataMap['Histories']    = Histories
    self.dataMap['History']      = History
    self.dataMap['TimePointSet'] = TimePointSet
    self.dataMap['TimePoint']    = TimePoint
    
         