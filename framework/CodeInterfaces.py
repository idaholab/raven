'''
Created on April 14, 2014

@author: alfoa

comment: The CodeInterface Module is an Handler.
         It inquires all the modules contained in the folder './CodeInterfaces'
         and load them, constructing a '__interFaceDict' on the fly
'''
#for future compatibility with Python 3--------------------------------------------------------------
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
start_dir = os.path.join(os.path.dirname(__file__),'CodeInterfaces')
for dirr,_,_ in os.walk(start_dir):
  __moduleInterfaceList.extend(glob(os.path.join(dirr,"*.py")))
  utils.add_path(dirr)
__moduleImportedList = []

'''
 Interface Dictionary (factory) (private)
'''
__base                          = 'Code'
__interFaceDict                 = {}
for moduleIndex in range(len(__moduleInterfaceList)):
  if 'class' in open(__moduleInterfaceList[moduleIndex]).read():
    __moduleImportedList.append(utils.importFromPath(__moduleInterfaceList[moduleIndex]))
    for key,modClass in inspect.getmembers(__moduleImportedList[moduleIndex], inspect.isclass):
      if 'createNewInput' in modClass.__dict__.keys():__interFaceDict[key.replace("Interface","")] = modClass
__knownTypes      = list(__interFaceDict.keys())

def knonwnTypes():
  return __knownTypes

def returnCodeInterface(Type):
  '''this allow to the code(model) class to interact with a specific
     code for which the interface is present in the CodeInterfaces module'''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
