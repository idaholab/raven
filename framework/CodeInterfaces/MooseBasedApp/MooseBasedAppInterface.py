"""
Created on April 14, 2014

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
from CodeInterfaceBaseClass import CodeInterfaceBase
import MooseData
import csvUtilities

class MooseBasedAppInterface(CodeInterfaceBase):
  """
    this class is used as part of a code dictionary to specialize Model.Code for RAVEN
  """
  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False
    self.mooseVPPFile = ''
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    if fargs['moosevpp'] != '' : self.mooseVPPFile = fargs['moosevpp']
    outputfile = 'out~'+inputFiles[index].getBase()
    executeCommand = [('parallel',executable+' -i '+inputFiles[index].getFilename() +
                        ' Outputs/file_base='+ outputfile +
                        ' Outputs/csv=true')]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      this generates a new input file depending on which sampler has been chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
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
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for point samplers (Grid, Stratified, etc.)
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    # the position in, eventually, a vector variable is not available yet...
    # the MOOSEparser needs to be modified in order to accept this variable type
    # for now the position (i.e. ':' at the end of a variable name) is discarded
    listDict = self.expandVarNames(**Kwargs)
    return listDict

  def dynamicEventTreeForMooseBasedApp(self,**Kwargs):
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for DET samplers (DynamicEventTree, AdaptiveDynamicEventTree, etc.)
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    listDict = []
    raise IOError('dynamicEventTreeForMooseBasedApp not yet implemented')
    return listDict

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      this method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output formats into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, returnOut, string, optional, present in case the root of the output file gets changed in this method.
    """
    returnOut = output
    if self.vectorPPFound: returnOut = self.__mergeTime(output,workingDir)[0]
    return returnOut

  def __mergeTime(self,output,workingDir):
    """
      Merges the vector PP output files created with the MooseApp
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, vppFiles, list, the list of files merged from the outputs of the vector PP
    """
    files2Merge, vppFiles  = [], []
    for time in range(int(self.vectorPPDict['timeStep'][0])):
      files2Merge.append(os.path.join(workingDir,str(output+self.mooseVPPFile+("%04d" % (time+1))+'.csv')))
      outputObj = MooseData.mooseData(files2Merge,workingDir,output,self.mooseVPPFile)
      vppFiles.append(os.path.join(workingDir,str(outputObj.vppFiles)))
    return vppFiles
