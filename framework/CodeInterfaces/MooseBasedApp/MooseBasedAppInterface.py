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
import MooseData
import csvUtilities

class MooseBasedAppInterface(CodeInterfaceBase):
  '''this class is used as part of a code dictionary to specialize Model.Code for RAVEN'''
  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
    See base class.  Collects all the clargs and the executable to produce the command-line call.
    Returns tuple of commands and base file name for run.
    Commands are a list of tuples, indicating parallel/serial and the execution command to use.
    @ In, inputFiles, the input files to be used for the run
    @ In, executable, the executable to be run
    @ In, clargs, command-line arguments to be used
    @ In, fargs, in-file changes to be made
    @Out, tuple( list(tuple(serial/parallel, exec_command)), outFileRoot string)
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    outputfile = 'out~'+inputFiles[index].getBase()
    executeCommand = [('parallel',executable+' -i '+inputFiles[index].getFilename() +
                        ' Outputs/file_base='+ outputfile +
                        ' Outputs/csv=true')]

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
    newInputFiles = copy.deepcopy(currentInputFiles)
    #TODO fix this? storing unwieldy amounts of data in 'prefix'
    if type(Kwargs['prefix']) in [str,type("")]:#Specifing string type for python 2 and 3
      newInputFiles[index].setBase(Kwargs['prefix']+"~"+currentInputFiles[index].getBase())
    else:
      newInputFiles[index].setBase(str(Kwargs['prefix'][1][0])+"~"+currentInputFiles[index].getBase())
    parser.printInput(newInputFiles[index].getAbsFile())
    self.vectorPPFound, self.vectorPPDict = parser.vectorPostProcessor()
    return newInputFiles

  def pointSamplerForMooseBasedApp(self,**Kwargs):
    listDict  = []
    modifDict = {}
    # the position in, eventually, a vector variable is not available yet...
    # the MOOSEparser needs to be modified in order to accept this variable type
    # for now the position (i.e. ':' at the end of a variable name) is discarded
    return self.expandVarNames(**Kwargs)

  def dynamicEventTreeForMooseBasedApp(self,**Kwargs):
    raise IOError('dynamicEventTreeForMooseBasedApp not yet implemented')
    listDict = []
    return listDict

  def finalizeCodeOutput(self,command,output,workingDir):
    """ this method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
        It can be used for those codes, that do not create CSV files to convert the whaterver output formato into a csv
        @ command, Input, the command used to run the just ended job (NOT Used at the moment)
        @ output, Input, the Output name root (string)
        @ workingDir, Input, actual working dir (string)
        @ return is optional, in case the root of the output file gets changed in this method.
    """

    if self.vectorPPFound:
      if len(self.vectorPPDict['rings']) == 1: return self.__mergeTime(output,workingDir)[0]
      else:
        ringFiles = self.__mergeTime(output,workingDir)
        for i in range(len(ringFiles)): ringFiles[i] = os.path.join(ringFiles[i],'.csv')
        outputObj = csvUtilities.csvUtilityClass(ringFiles)
        outputFileName = os.path.join(workingDir,str(output+'_VPP.csv'))
        options = {'variablesToExpandFrom': ["timeStep"]}
        outputObj.mergeCSV(outputFileName, options)
        return  outputFileName
    else: return

  def __mergeTime(self,output,workingDir):
    """
    merges the vector PP output files created with the MooseApp
    @ In, output: the Output name root (string)
    @ In, workingDir: Actual working dir (string)
    @ Out, vppFiles: the files merged from the outputs of the vector PP
    """
    files2Merge = []
    vppFiles = []
    if 'JIntegral' in self.vectorPPDict['integrals']:
      integral = 'J'
    elif 'KIntegral' in self.vectorPPDict['integrals']: integral = 'K' #TODO: This is an example KIntegral could be something different did not check it fwith MOOSE yet.
    else: integral = '' #TODO: implement an error or warning message
    for ring in range(len(self.vectorPPDict['rings'])):
      files2Merge.append([])
      for time in range(int(self.vectorPPDict['timeStep'][0])):
        files2Merge[ring].append(os.path.join(workingDir,str(output+'_'+integral+'_'+str(ring+1)+'_'+("%04d" % (time+1))+'.csv')))
      outputObj = MooseData.mooseData(files2Merge[ring],workingDir,output+str(ring+1),ring)
      vppFiles.append(os.path.join(workingDir,str(outputObj.vppFiles)))
    return vppFiles

