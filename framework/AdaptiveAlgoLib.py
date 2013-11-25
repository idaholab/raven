'''
Created on Nov 21, 2013

@author: crisr
'''
def TimePoint():
  pass

__base                          = 'AdaptiveAlgorithms'
__interFaceDict                  = {}
__interFaceDict['leastDistance'] = TimePoint
__knownTypes                     = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  
