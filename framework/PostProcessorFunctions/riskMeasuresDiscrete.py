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

import numpy as np

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
        self.target = child.text.split(',')
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
    #print(inputDic['data']['input'])
    print(inputDic['metadata'])
    
    if 'metadata' in inputDic.keys():
      pbPresent = 'ProbabilityWeight' in inputDic['metadata'][0].keys() if 'metadata' in inputDic.keys() else False
    if not pbPresent:
      pbWeights['realization'] = np.asarray([1.0 / len(inputDic['targets'][self.parameters['targets'][0]])]*len(inputDic['targets'][self.parameters['targets'][0]]))
    else: 
      pbWeights['realization'] = inputDic['metadata']['ProbabilityWeight']/np.sum(inputDic['metadata']['ProbabilityWeight'])
    print(pbWeights['realization'])
    for variable in self.variables:
      R0     = 0.0
      Rminus = 0.0
      Rplus  = 0.0
      
      ''' Calculate R0, Rminus, Rplus '''
      R0 = np.inner()
      
      ''' Calculate RRW, RAW, FV, B '''
      RRW = R0/Rminus
      RAW = Rplus/R0
      FV  = (R0-Rminus)/R0
      B   = Rplus + Rminus
      
      outputDic[variable + '_RRW'] = RRW
      outputDic[variable + '_RAW'] = RAW
      outputDic[variable + '_FV']  = FV
      outputDic[variable + '_B']   = B
      
    return outputDic


