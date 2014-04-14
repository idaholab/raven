'''
Created on April 14, 2014

@author: alfoa

comment: The CodeInterface Module is an Handler. 
         It inquires all the modules contained in the folder './CodeInterfaces'
         and load them, constructing a '__interFaceDict' on the fly
'''

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
from glob import glob
import imp
import inspect
import utils

__moduleInterfaceList = []
start_dir = os.path.join(os.path.dirname(__file__),'CodeInterfaces')
for dir,_,_ in os.walk(start_dir): __moduleInterfaceList.extend(glob(os.path.join(dir,"*.py")))
__moduleImportedList = []

'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'Code'
__interFaceDict                 = {}
for moduleIndex in range(len(__moduleInterfaceList)):
  __moduleImportedList.append(utils.importFromPath(__moduleInterfaceList[moduleIndex]))
  #__moduleImportedList.append(imp.load_module(str(moduleIndex),__moduleInterfaceList[moduleIndex]))
  for key,modClass in inspect.getmembers(__moduleImportedList[moduleIndex], inspect.isclass): 
    __interFaceDict[key.replace("Interface","")] = modClass 
__knownTypes      = list(__interFaceDict.keys())

def knonwnTypes():
  return __knownTypes

def returnCodeInterface(Type):
  print(__interFaceDict)
  '''this allow to the code(model) class to interact with a specific
     code for which the interface is present in the CodeInterfaces module'''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
