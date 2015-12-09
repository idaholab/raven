"""
Created on December 1, 2015

"""

#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
from glob import glob
import inspect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
#Internal Modules End--------------------------------------------------------------------------------

__moduleInterfaceList = []
startDir = os.path.join(os.path.dirname(__file__),'PostProcessorFunctions')
for dirr,_,_ in os.walk(startDir):
  __moduleInterfaceList.extend(glob(os.path.join(dirr,"*.py")))
  utils.add_path(dirr)
__moduleImportedList = []

'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'PostProcessor'
__interFaceDict                 = {}
for moduleIndex in range(len(__moduleInterfaceList)):
  if 'class' in open(__moduleInterfaceList[moduleIndex]).read():
    __moduleImportedList.append(utils.importFromPath(__moduleInterfaceList[moduleIndex],False))
    for key,modClass in inspect.getmembers(__moduleImportedList[-1], inspect.isclass):
      if 'run' in modClass.__dict__.keys():__interFaceDict[key] = modClass
__knownTypes      = list(__interFaceDict.keys())

def knownTypes(): 
  return __knownTypes

def returnPostProcessorInterface(Type,caller):
  if Type not in knownTypes(): caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
  return __interFaceDict[Type]()

