'''
Created on December 1, 2015

'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
#External Modules End--------------------------------------------------------------------------------

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase

class RiskMeasuresDiscrete(PostProcessorInterfaceBase):
  """ This class implements the four basic risk-importance measures
      This class inherits form the base class PostProcessorInterfaceBase and it contains three methods:
      - initialize
      - run
      - readMoreXML
  """

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'PointSet'
    self.outputFormat = 'PointSet'
    
  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    self.variables = {}
    self.target    = {}
    
    for child in xmlNode:
      if child.tag == 'measures':
        self.measures = child.text.split(',')
        
      elif child.tag == 'variable':
        variableID = child.text
        self.variables[variableID] = {}
        if 'R0values' in child.attrib.keys():
          values = child.attrib['R0values'].split(',')
          if len(values)>2 or len(values)==1:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R0 for XML node: ' + str(child) + ' has one or more than two values')
          val1 = float(values[0])
          val2 = float(values[1])
          self.variables[variableID]['R0low']  = min(val1,val2)
          self.variables[variableID]['R0high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R0 is not present for XML node: ' + str(child) )
        if 'R1values' in child.attrib.keys():
          values = child.attrib['R1values'].split(',')
          if len(values)>2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R1 for XML node: ' + str(child) + ' has more than two values')
          val1 = float(values[0])
          val2 = float(values[1])
          self.variables[variableID]['R1low']  = min(val1,val2)
          self.variables[variableID]['R1high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R1 is not present for XML node: ' + str(child) )
      
      elif child.tag == 'target':
        self.target['targetID'] = child.text
        if 'values' in child.attrib.keys():
          values = child.attrib['values'].split(',')
          if len(values)>2 or len(values)==1:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node values for XML node: ' + str(child) + ' has one or more than two values')
          val1 = float(values[0])
          val2 = float(values[1])
          self.target['low']  = min(val1,val2)
          self.target['high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node values is not present for XML node: ' + str(child) )
        
      elif child.tag !='method':
        self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')
        
    if not set(self.measures).issubset(['B','FV','RAW','RRW']):
      self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : measures ' 
                        + str(set(self.measures).issubset([B,FV,RAW,RRW])) + ' are not recognized')

  def run(self,inputDic):
    """
    This method perform the actual calculation of the risk measures
     @ In,  inputDic,  dict, dictionary which contains the data inside the input DataObject
     @ Out, outputDic, dict, dictionary which contains the risk measures

    """
        
    if 'metadata' in inputDic.keys():     
      if 'ProbabilityWeight' in inputDic['metadata'][0].keys():
        pbPresent = True
      else:
        pbPresent = False
    else:
      pbPresent = False

    if not pbPresent:
      pbWeights= np.asarray([1.0 / len(inputDic['targets'][self.parameters['targets'][0]])]*len(inputDic['targets'][self.parameters['targets'][0]]))
    else: 
      pbWeights = inputDic['metadata'][0]['ProbabilityWeight']/np.sum(inputDic['metadata'][0]['ProbabilityWeight'])

    N = pbWeights.size

    # 1.0 = failure
    # 0.0 = OK
    
    outputDic = {}
    outputDic['data'] = {}
    outputDic['data']['output'] = {}
    outputDic['data']['input']  = {}
    
    print('======= Risk Measures =============')
    
    for variable in self.variables:
      data=np.zeros((3,N))
      data[0] = pbWeights
      data[1] = inputDic['data']['input'][variable]
      data[2] = inputDic['data']['output'][self.target['targetID']]

      ''' Calculate R0, Rminus, Rplus '''
      
      indexSystemFailure = np.where(np.logical_or(data[2,:]<self.target['low'], data[2,:]>self.target['high'])) 
      dataSystemFailure  = np.delete(data, indexSystemFailure,  axis=1)
      
      minusValues     = np.where(np.logical_or(dataSystemFailure[1,:]<self.variables[variable]['R1low'], dataSystemFailure[1,:]>self.variables[variable]['R1high']))  
      plusValues      = np.where(np.logical_or(dataSystemFailure[1,:]<self.variables[variable]['R0low'], dataSystemFailure[1,:]>self.variables[variable]['R0high'])) 
      dataSystemMinus = np.delete(dataSystemFailure, minusValues, axis=1)
      dataSystemPlus  = np.delete(dataSystemFailure, plusValues , axis=1)
      
      indexComponentFailureMinus = np.where(np.logical_or(data[1,:]<self.variables[variable]['R1low'], data[1,:]>self.variables[variable]['R1high']))
      indexComponentFailurePlus  = np.where(np.logical_or(data[1,:]<self.variables[variable]['R0low'], data[1,:]>self.variables[variable]['R0high']))
      dataComponentMinus = np.delete(data, indexComponentFailureMinus, axis=1)
      dataComponentPlus  = np.delete(data, indexComponentFailurePlus,  axis=1)

      R0     = np.sum(dataSystemFailure[0,:])
      Rminus = np.sum(dataSystemMinus[0,:])/np.sum(dataComponentMinus[0,:])
      Rplus  = np.sum(dataSystemPlus[0,:]) /np.sum(dataComponentPlus[0,:])
      
      print('--> ' + str(variable) + ' Rminus = ' + str(Rminus))
      print('--> ' + str(variable) + ' Rplus  = ' + str(Rplus))
      print('--> ' + str(variable) + ' R0     = ' + str(R0))
      
      ''' Calculate RRW, RAW, FV, B '''
      RRW = R0/Rminus
      RAW = Rplus/R0
      FV  = (R0-Rminus)/R0
      B   = Rplus+Rminus
            
      if 'RRW' in self.measures:
        outputDic['data']['output'][variable + '_RRW'] = np.asanyarray([RRW])
        print('--> ' + str(variable) + 'RRW = ' + str(RRW))
      if 'RAW' in self.measures:
        outputDic['data']['output'][variable + '_RAW'] = np.asanyarray([RAW])
        print('--> ' + str(variable) + 'RAW = ' + str(RAW))
      if 'FV' in self.measures:
        outputDic['data']['output'][variable + '_FV']  = np.asanyarray([FV])
        print('--> ' + str(variable) + 'FV = ' + str(FV))
      if 'B' in self.measures:
        outputDic['data']['output'][variable + '_B']   = np.asanyarray([B])
        print('--> ' + str(variable) + 'B  = ' + str(B))
      
      outputDic['data']['input'][variable + '_avg'] = np.asanyarray([np.sum(dataSystemMinus[0,:])])
   
    outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])   
   
    return outputDic


