'''
Created on Dec 2, 2014

@author: talbpw
'''
#for future compatibility with Python 3-----------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3-------------------------------------------------

#External Modules---------------------------------------------------------------------
import numpy as np
import scipy.special.orthogonal as quads
from scipy.fftpack import ifft
from scipy.misc import factorial
#External Modules End-----------------------------------------------------------------

#Internal Modules
from BaseClasses import BaseType
from utils import returnPrintTag, returnPrintPostTag, find_distribution1D
#Internal Modules End-----------------------------------------------------------------


class QuadratureSet(BaseType):
  '''Base class to produce standard quadrature points and weights.
     Points and weights are obtained as

     myQuad = Legendre()
     pts,wts = myQuad(n)
  '''
  def __init__(self):
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    self.debug = False

  def __call__(self,order):
    '''Defines operations to return correct pts, wts'''
    pts,wts = self.rule(order,*self.params)
    pts = np.around(pts,decimals=14) #helps with nesting, might not be desirable
    if self.debug: print(self.printTag,'TODO this could probably be optimized further')
    return pts,wts

  def whatDoINeed(self):
    needDict = self._localWhatDoINeed()
    for val in self.assemblerObjects.values():
      for value in val:
        if value[0] not in needDict.keys(): needDict[value[0]] = []
        needDict[value[0]].append((value[1],value[2]))
    return needDict

  def _localWhatDoINeed(self):
    return {}

  def generateAssembler(self,initDict):
    self._localGenerateAssembler(initDict)

  def _localGenerateAssembler(self,initDict):
    pass

  def initialize(self,*params):
    self.rule   = None   # Function as rule(n) that takes an integer order and returns pts, wts
    self.params = params # list of additional parameters necessary for some quadratures

  def _readMoreXML(self,xmlNode):
    if self.debug:print('Quadrature: need to fix _readMoreXML')
    self._localReadMoreXML(xmlNode)
    return
    try:
      self.type = xmlNode.tag
      self.name = xmlNode.attrib['name']
      self.printTag = self.type.ljust(25)
      if 'debug' in xmlNode.attrib.keys(): self.debug = bool(xmlNode.attrib['debug'])
      #TODO assembler stuff
      if self.debug:print('TODO Quadrature needs to implement assembler stuff in readXML')
    except: pass

  def _localReadMoreXML(self,xmlNode):
    pass


class Legendre(QuadratureSet):
  def initialize(self):
    self.rule   = quads.p_roots
    self.params = []

class CDF(Legendre): #added just for name distinguish; equiv to Legendre
  pass

class Hermite(QuadratureSet):
  def initialize(self):
    self.rule   = quads.he_roots
    self.params = []

class Laguerre(QuadratureSet):
  def initialize(self):
    self.rule   = quads.la_roots

  def _localReadMoreXML(self,xmlNode):
    self.params=[]
    if xmlNode.find('alpha') != None:
      alpha = float(xmlNode.find('alpha').text)
    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Laguerre quadrature requires alpha keyword; not found.')
    self.params = [alpha-1]

class Jacobi(QuadratureSet):
  def initialize(self):
    self.rule   = quads.j_roots
    self.params = []

  def _localReadMoreXML(self,xmlNode):
    if xmlNode.find('alpha') != None:
      self.params.append(float(xmlNode.find('alpha').text))
    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Laguerre quadrature requires alpha keyword; not found.')
    if xmlNode.find('beta') != None:
      self.params.append(float(xmlNode.find('beta').text))
    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Laguerre quadrature requires beta keyword; not found.')
    self.params = [alpha,beta]


class ClenshawCurtis(QuadratureSet):
  def initialize(self):
    self.rule = self.cc_roots
    self.params = []

  def cc_roots(self,o):
    '''Computes Clenshaw Curtis nodes and weights for given order n=2^o+1'''
    n1=2**o+1 #assures nested
    if o==1:
      return np.array([np.array([0]),np.array([2])])
    else:
      #n1=2**o+1 #assures nested
      n = n1-1
      C = np.zeros((n1,2))
      k = 2*(1+np.arange(np.floor(n/2)))
      C[::2,0] = 2/np.hstack((1,1-k*k))
      C[1,1]=-n
      V = np.vstack((C,np.flipud(C[1:n,:])))
      F = np.real(ifft(V,n=None,axis=0))
      x = F[0:n1,1]
      w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
    return x,w



'''
 Interface Dictionary (factory) (private)
'''
__base = 'QuadratureSet'
__interFaceDict = {}
__interFaceDict['Legendre'] = Legendre
__interFaceDict['Cdf'] = CDF
__interFaceDict['Hermite'] = Hermite
__interFaceDict['Laguerre'] = Laguerre
__interFaceDict['Jacobi'] = Jacobi
__interFaceDict['ClenshawCurtis'] = ClenshawCurtis
__knownTypes = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  '''
    function used to generate a Filter class
    @ In, Type : Filter type
    @ Out,Instance of the Specialized Filter class
  '''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)

