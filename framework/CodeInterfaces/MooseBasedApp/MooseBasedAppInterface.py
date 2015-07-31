'''
Created on April 14, 2014

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
from CodeInterfaceBaseClass import CodeInterfaceBase

class MooseBasedAppInterface(CodeInterfaceBase):
  '''this class is used as part of a code dictionary to specialize Model.Code for RAVEN'''
  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    '''seek which is which of the input files and generate According the running command'''
    found = False
    for index, inputFile in enumerate(inputFiles):
      if '.'+inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    outputfile = 'out~'+inputFiles[index].getBase()
    executeCommand = (executable+' -i '+inputFiles[index].getFilename() +
                        ' Outputs/file_base='+ outputfile +
                        ' Outputs/interval=1'+ ' Outputs/output_initial=true' + ' Outputs/csv=true')

    return executeCommand,outputfile

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler has been chosen'''
    import MOOSEparser
    self._samplersDictionary                          = {}
    self._samplersDictionary['MonteCarlo'           ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['Grid'                 ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['Stratified'           ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['DynamicEventTree'     ] = self.dynamicEventTreeForMooseBasedApp
    self._samplersDictionary['StochasticCollocation'] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['FactorialDesign'      ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['ResponseSurfaceDesign'] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['Adaptive']              = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['SparseGridCollocation'] = self.pointSamplerForMooseBasedApp
    print('WARNING: Sampler type not found in MooseBasedApp Interface dictionaries; continuing with default...')
    found = False
    for index, inputFile in enumerate(currentInputFiles):
      inputFile = inputFile.getAbsFile()
      if inputFile.endswith(self.getInputExtension()):
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index].getAbsFile())
    modifDict = self._samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,False)
    temp = str(oriInputFiles[index])[:]
    newInputFiles = copy.deepcopy(currentInputFiles)
    #TODO fix this? storing unwieldy amounts of data in 'prefix'
    if type(Kwargs['prefix']) in [str,type("")]:#Specifing string type for python 2 and 3
      #newInputFiles[index] = os.path.join(os.path.split(temp)[0],Kwargs['prefix']+"~"+os.path.split(temp)[1])
      newInputFiles[index].setBase(Kwargs['prefix']+"~"+currentInputFiles[index].getBase())
    else:
      newInputFiles[index].setBase(str(Kwargs['prefix'][1][0])+"~"+currentInputFiles[index].getBase())
      #newInputFiles[index] = os.path.join(os.path.split(temp)[0],str(Kwargs['prefix'][1][0])+"~"+os.path.split(temp)[1])
    parser.printInput(newInputFiles[index].getAbsFile())
    return newInputFiles

  def pointSamplerForMooseBasedApp(self,**Kwargs):
    listDict  = []
    modifDict = {}
    # the position in, eventually, a vector variable is not available yet...
    # the MOOSEparser needs to be modified in order to accept this variable type
    # for now the position (i.e. ':' at the end of a variable name) is discarded
    #return self.expandVarNames(**Kwargs)
    dum =  self.expandVarNames(**Kwargs)
    print('dum: ',dum)
    return dum

  def dynamicEventTreeForMooseBasedApp(self,**Kwargs):
    raise IOError('dynamicEventTreeForMooseBasedApp not yet implemented')
    listDict = []
    return listDict
