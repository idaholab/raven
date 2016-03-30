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
 Interfaced Post Processor
 Here all the Interfaced Post-Processors located in the raven/framework/PostProcessorFunctions folder are parsed and their instance is returned
'''

__base                          = 'PostProcessor'
__interFaceDict                 = {}
for moduleIndex in range(len(__moduleInterfaceList)):
  if 'class' in open(__moduleInterfaceList[moduleIndex]).read():
    __moduleImportedList.append(utils.importFromPath(__moduleInterfaceList[moduleIndex],False))
    for key,modClass in inspect.getmembers(__moduleImportedList[-1], inspect.isclass):
      # in this way we can get all the class methods
      classMethods = [method for method in dir(modClass) if callable(getattr(modClass, method))]
      if 'run' in classMethods: __interFaceDict[key] = modClass
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  """
    This function returns the types of interfaced post-processors available
    @ In, None,
    @ Out, __knownTypes, list, list of recognized post-processors
  """
  return __knownTypes

def returnPostProcessorInterface(Type,caller):
  """
    This function returns interfaced post-processors interface
    @ In, Type, string, type of Interfaced PostProcessor to run
    @ In, caller, instance of the PostProcessor class
    @ Out, __interFaceDict[Type](), dict, interfaced PostProcessor dictionary
  """
  if Type not in knownTypes():
    caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
  return __interFaceDict[Type](caller.messageHandler)

