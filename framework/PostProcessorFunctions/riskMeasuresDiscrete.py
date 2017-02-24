"""
Created on November 2016

@author: mandd
"""
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

class riskMeasuresDiscrete(PostProcessorInterfaceBase):
  """ This class implements the four basic risk-importance measures
      This class inherits form the base class PostProcessorInterfaceBase and it contains three methods:
      - initialize
      - run
      - readMoreXML
  """

  def initialize(self):
    """
      Method to initialize the Interfaced Post-processor
      @ In, None
      @ Out, None

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
    
    self.IEdata = {}

    for child in xmlNode:
      if child.tag == 'measures':
        self.measures = child.text.split(',')

      elif child.tag == 'variable':
        variableID = child.text
        self.variables[variableID] = {}
        if 'R0values' in child.attrib.keys():
          values = child.attrib['R0values'].split(',')
          if len(values) != 2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R0 for XML node: ' + str(child) + ' has one or more than two values')
          try:
            val1 = float(values[0])
            val2 = float(values[1])
          except:
            self.raiseAnError(IOError,' Wrong R0values associated to riskMeasuresDiscrete Post-Processor')
          self.variables[variableID]['R0low']  = min(val1,val2)
          self.variables[variableID]['R0high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R0 is not present for XML node: ' + str(child) )
        if 'R1values' in child.attrib.keys():
          values = child.attrib['R1values'].split(',')
          if len(values)>2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R1 for XML node: ' + str(child) + ' has more than two values')
          try:
            val1 = float(values[0])
            val2 = float(values[1])
          except:
            self.raiseAnError(IOError,' Wrong R1values associated to riskMeasuresDiscrete Post-Processor')
          self.variables[variableID]['R1low']  = min(val1,val2)
          self.variables[variableID]['R1high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R1 is not present for XML node: ' + str(child) )

      elif child.tag == 'target':
        self.target['targetID'] = child.text
        if 'values' in child.attrib.keys():
          values = child.attrib['values'].split(',')
          if len(values) != 2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node values for XML node: ' + str(child) + ' has one or more than two values')
          try:
            val1 = float(values[0])
            val2 = float(values[1])
          except:
            self.raiseAnError(IOError,' Wrong target values associated to riskMeasuresDiscrete Post-Processor')
          self.target['low']  = min(val1,val2)
          self.target['high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node values is not present for XML node: ' + str(child) )
      
      elif child.tag == 'data':
        self.IEdata[child.text] = float(child.attrib['freq'])
      
      elif child.tag !='method':
        self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if not set(self.measures).issubset(['B','FV','RAW','RRW']):
      self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : measures '
                        + str(set(self.measures).issubset([B,FV,RAW,RRW])) + ' are not recognized')

  def run(self,inputDic):
    """
     This method perform the actual calculation of the risk measures
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ Out, outputDic, dict, dictionary which contains the risk measures

    """

    Rvalues = {}
    
    for inp in inputDic:
      print(inp['name'])
      Rvalues[inp['name']] = {}
      pbPresent = False
      if 'metadata' in inp.keys() and 'ProbabilityWeight' in inp['metadata'].keys():
        pbPresent = True

      if not pbPresent:
        pbWeights= np.asarray([1.0 / len(inp['targets'][self.parameters['targets'][0]])]*len(inp['targets'][self.parameters['targets'][0]]))
      else:
        pbWeights = inp['metadata']['ProbabilityWeight']/np.sum(inp['metadata']['ProbabilityWeight'])

      N = pbWeights.size

      outputDic = {}
      outputDic['data'] = {}
      outputDic['data']['output'] = {}
      outputDic['data']['input']  = {}
      
      for variable in self.variables:
        Rvalues[inp['name']][variable] = {}
        
        if variable in inp['data']['input'].keys():
          data=np.zeros((3,N))
          data[0] = pbWeights
          data[1] = inp['data']['input'][variable]
          data[2] = inp['data']['output'][self.target['targetID']]
  
          #Calculate R0, Rminus, Rplus
  
          # Step 1: Retrieve points that contain system failure
          indexSystemFailure = np.where(np.logical_or(data[2,:]<self.target['low'], data[2,:]>self.target['high']))
          dataSystemFailure  = np.delete(data, indexSystemFailure,  axis=1)
          
          # Step 2: Retrieve points from Step 1 that contain component reliability values equal to 1 (minusValues) and 0 (plusValues)
          minusValues     = np.where(np.logical_or(dataSystemFailure[1,:]<self.variables[variable]['R1low'], dataSystemFailure[1,:]>self.variables[variable]['R1high']))
          plusValues      = np.where(np.logical_or(dataSystemFailure[1,:]<self.variables[variable]['R0low'], dataSystemFailure[1,:]>self.variables[variable]['R0high']))
          dataSystemMinus = np.delete(dataSystemFailure, minusValues, axis=1)
          dataSystemPlus  = np.delete(dataSystemFailure, plusValues , axis=1)
  
          # Step 3: Retrieve points from original dataset that contain component reliability values equal to 1 (minusValues) and 0 (plusValues)
          indexComponentFailureMinus = np.where(np.logical_or(data[1,:]<self.variables[variable]['R1low'], data[1,:]>self.variables[variable]['R1high']))
          indexComponentFailurePlus  = np.where(np.logical_or(data[1,:]<self.variables[variable]['R0low'], data[1,:]>self.variables[variable]['R0high']))
          dataComponentMinus = np.delete(data, indexComponentFailureMinus, axis=1)
          dataComponentPlus  = np.delete(data, indexComponentFailurePlus,  axis=1)
  
          # Step 4: Sum pb weights for the subsets retrieved in Steps 1 2 and 3
  
          # R0 = pb of system failure
          Rvalues[inp['name']][variable]['R0']     = np.sum(dataSystemFailure[0,:])
          # Rminus = pb of system failure given component reliability is 1
          Rvalues[inp['name']][variable]['Rminus'] = np.sum(dataSystemMinus[0,:])/np.sum(dataComponentMinus[0,:])
          # Rplus = pb of system failure given component reliability is 0
          Rvalues[inp['name']][variable]['Rplus']  = np.sum(dataSystemPlus[0,:]) /np.sum(dataComponentPlus[0,:])
        else:
          data=np.zeros((2,N))
          data[0] = pbWeights
          data[1] = inp['data']['output'][self.target['targetID']]
          
          indexSystemFailure = np.where(np.logical_or(data[1,:]<self.target['low'], data[1,:]>self.target['high']))
          dataSystemFailure  = np.delete(data, indexSystemFailure,  axis=1) 
          Rvalues[inp['name']][variable] ['R0'] = Rvalues[inp['name']][variable]['Rminus'] = Rvalues[inp['name']][variable]['Rplus'] = np.sum(dataSystemFailure[0,:]) 
        
        print(str(variable) + " R0 : " + str(Rvalues[inp['name']][variable]['R0']))
        print(str(variable) + " R- : " + str(Rvalues[inp['name']][variable]['Rminus']))
        print(str(variable) + " R+ : " + str(Rvalues[inp['name']][variable]['Rplus']))
          
        # Step 5: Calculate RRW, RAW, FV, B for each variable and for each data set
        #measures[inp['name']][variable]['RRW'] = R0/Rminus
        #measures[inp['name']][variable]['RAW'] = Rplus/R0
        #measures[inp['name']][variable]['FV']  = (R0-Rminus)/R0
        #measures[inp['name']][variable]['B']   = Rplus-Rminus
      
    #Step 6: Determine global values for RRW, RAW, FV, B 
    RvaluesMacro = {}
    for variable in self.variables:
      RvaluesMacro[variable] = {}
      RvaluesMacro[variable]['R0']  = 0
      RvaluesMacro[variable]['Rminus'] = 0
      RvaluesMacro[variable]['Rplus']  = 0
      for inp in inputDic:
        if inp['name'] in self.IEdata.keys():
          multiplier = self.IEdata[inp['name']]
        else:
          multiplier = 1.0
          self.raiseAWarning('RiskMeasuresDiscrete Interfaced Post-Processor: the dataObject ' + str (inp['name']) + ' does not have the frequency of the IE specified. It is assumed that the frequency of the IE is 1.0')
        RvaluesMacro[variable]['R0']     += multiplier * Rvalues[inp['name']][variable]['R0']
        RvaluesMacro[variable]['Rminus'] += multiplier * Rvalues[inp['name']][variable]['Rminus']
        RvaluesMacro[variable]['Rplus']  += multiplier * Rvalues[inp['name']][variable]['Rplus']         
        
    
      if 'RRW' in self.measures:
        RRW = outputDic['data']['output'][variable + '_RRW'] = np.asanyarray([RvaluesMacro[variable]['R0']/RvaluesMacro[variable]['Rminus']])
        self.raiseADebug(str(variable) + ' RRW = ' + str(RRW))
      if 'RAW' in self.measures:
        RAW = outputDic['data']['output'][variable + '_RAW'] = np.asanyarray([RvaluesMacro[variable]['Rplus']/RvaluesMacro[variable]['R0']])
        self.raiseADebug(str(variable) + ' RAW = ' + str(RAW))
      if 'FV' in self.measures:
        FV = outputDic['data']['output'][variable + '_FV']  = np.asanyarray([(RvaluesMacro[variable]['R0']-RvaluesMacro[variable]['Rminus'])/RvaluesMacro[variable]['R0']])
        self.raiseADebug( str(variable) + ' FV = ' + str(FV))
      if 'B' in self.measures:
        B = outputDic['data']['output'][variable + '_B']   = np.asanyarray([RvaluesMacro[variable]['Rplus']-RvaluesMacro[variable]['Rminus']])
        self.raiseADebug(str(variable) + ' B  = ' + str(B))

      outputDic['data']['input'][variable + '_avg'] = np.asanyarray([np.sum(dataSystemMinus[0,:])])

    outputDic['metadata'] = copy.deepcopy(inp['metadata'])


    return outputDic


