'''
Created on Mar 7, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import xml.etree.ElementTree as ET
import scipy.stats.distributions  as dist
import numpy as np
from BaseType import BaseType

class Distribution(BaseType):
  ''' 
  a general class containing the distributions
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.upperBoundUsed = False
    self.lowerBoundUsed = False
    self.upperBound       = 0.0
    self.lowerBound       = 0.0
    self.adjustmentType   = ''
    
  def readMoreXML(self,xmlNode):
    if xmlNode.find('upperBound') !=None:
      self.upperBound = float(xmlNode.find('upperBound').text)
      self.upperBoundUsed = True
        
    if xmlNode.find('lowerBound')!=None:
      self.lowerBound = float(xmlNode.find('lowerBound').text)
      self.lowerBoundUsed = True
    if xmlNode.find('adjustment') !=None:
      self.adjustment = xmlNode.find('adjustment').text
    else:
      self.adjustment = 'scaling'

  def addInitParams(self,tempDict):
    tempDict['upperBoundUsed'] = self.upperBoundUsed
    tempDict['lowerBoundUsed'] = self.lowerBoundUsed
    tempDict['upperBound'    ] = self.upperBound
    tempDict['lowerBound'    ] = self.lowerBound
    tempDict['adjustmentType'] = self.adjustmentType

  def rvsWithinCDFbounds(self,LowerBound,upperBound):
    point = np.random.rand(1)*(upperBound-LowerBound)+LowerBound
    return self.distribution.ppt(point)

  def rvsWithinbounds(self,LowerBound,upperBound):
    CDFupper = self.distribution.cdf(upperBound)
    CDFlower = self.distribution.cdf(LowerBound)
    return self.rvsWithinCDFbounds(CDFlower,CDFupper)



class Normal(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.mean  = 0.0
    self.sigma = 0.0

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.mean  = float(xmlNode.find('mean' ).text)
    except: raise Exception('mean value needed for normal distribution')
    try: self.sigma = float(xmlNode.find('sigma').text)
    except: raise Exception('sigma value needed for normal distribution')
    self.inDistr()
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma
    
  def inDistr(self):
    if self.upperBoundUsed == False and self.lowerBoundUsed == False:
      self.distribution = dist.norm(loc=self.mean,scale=self.sigma)
    else:
      if self.lowerBoundUsed == False: a = -sys.float_info[max]
      else:a = (self.lowerBound - self.mean) / self.sigma
      if self.upperBoundUsed == False: b = sys.float_info[max]
      else:b = (self.upperBound - self.mean) / self.sigma
      self.distribution = dist.truncnorm(a,b,loc=self.mean,scale=self.sigma)
    
class Triangular(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.apex = 0.0
    self.min  = 0.0
    self.max  = 0.0
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.apex = float(xmlNode.find('apex').text)
    except: raise Exception('apex value needed for normal distribution')
    try: self.min = float(xmlNode.find('min').text)
    except: raise Exception('min value needed for normal distribution')
    try: self.max = float(xmlNode.find('max').text)
    except: raise Exception('max value needed for normal distribution')
    self.inDistr()
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['apex' ] = self.apex
    tempDict['min'  ] = self.min
    tempDict['max'  ] = self.max
  def inDistr(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      c = (self.apex-self.min)/(self.max-self.min)
      self.distribution = dist.triang(c,loc=self.min,scale=(self.max-self.min))
    else:
      raise IOError ('Truncated triangular not yet implemented')

def returnInstance(Type):
  base = 'Distribution'
  InterfaceDict = {}
  InterfaceDict['Normal'   ]  = Normal
  InterfaceDict['Triangular'] = Triangular
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
  
  
