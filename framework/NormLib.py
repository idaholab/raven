'''
Created on Nov 21, 2013

@author: crisr
'''

def RelL2Norm():
  pass

__base                   = 'AdaptiveAlgorithms'
__interFaceDict          = {}
__interFaceDict['relL2'] = RelL2Norm
__knownTypes                     = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  
