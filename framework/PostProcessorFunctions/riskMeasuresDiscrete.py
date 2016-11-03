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
    for child in xmlNode:
      if child.tag == 'measures':
        self.measures = child.text.split(',')
        print(self)
      elif child.tag == 'variables':
        self.variables = child.text.split(',')
      elif child.tag == 'target':
        self.target = child.text
      elif child.tag !='method':
        self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : XML node ' 
                          + str(child) + ' is not recognized')
        
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
      R0     = 0.0
      Rminus = 0.0
      Rplus  = 0.0

      data=np.zeros((3,N))
      data[0] = pbWeights
      data[1] = inputDic['data']['output'][variable]
      data[2] = inputDic['data']['output'][self.target]
      
      ''' Calculate R0, Rminus, Rplus '''
      minusValues = np.argwhere(data[1,:] == 0.0)
      plusValues  = np.argwhere(data[1,:] == 1.0)
      
      dataMinus = np.delete(data, minusValues, axis=1)
      dataPlus  = np.delete(data, plusValues , axis=1)
      
      indexMinus = np.argwhere(dataMinus[2,:] == 1.0)
      indexPlus  = np.argwhere(dataPlus[2,:]  == 1.0)
      indexZero  = np.argwhere(data[2,:]      == 1.0)
      
      dataReducedMinus = np.delete(dataMinus, indexMinus, axis=1)
      dataReducedPlus  = np.delete(dataPlus,  indexMinus, axis=1)
      dataReducedZero  = np.delete(data,      indexZero,  axis=1)
      
      Rminus = np.sum(dataReducedMinus[0,:])
      Rplus  = np.sum(dataReducedPlus[0,:])
      R0     = np.sum(dataReducedZero[0,:])
      
      print('--> ' + str(variable) + 'Rminus = ' + str(Rminus))
      print('--> ' + str(variable) + 'Rplus  = ' + str(Rplus))
      print('--> ' + str(variable) + 'R0     = ' + str(R0))
      
      ''' Calculate RRW, RAW, FV, B '''
      RRW = R0/Rminus
      RAW = Rplus/R0
      FV  = (R0-Rminus)/R0
      B   = Rplus + Rminus
            
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
      
      outputDic['data']['input'][variable + '_avg'] = np.asanyarray([np.sum(dataMinus[0,:])])
      
      
   
    outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])   
   
    return outputDic


