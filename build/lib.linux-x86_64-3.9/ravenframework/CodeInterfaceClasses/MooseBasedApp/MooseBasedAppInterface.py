# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on April 14, 2014

@author: alfoa
"""

import os
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..Generic import GenericParser
from . import MooseData
from . import MOOSEparser

class MooseBasedApp(CodeInterfaceBase):
  """
    this class is used as part of a code dictionary to specialize Model.Code for RAVEN
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.outputPrefix = 'out~'
    self.vectorPPFound = None # Indicates if a MOOSE vector postprocessor is in use
    self.vectorPPDict = None  # Contains information about the postprocessor used

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False
    self.mooseVPPFile = ''
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    if fargs['moosevpp'] != '':
      self.mooseVPPFile = fargs['moosevpp']
    outputfile = self.outputPrefix+inputFiles[index].getBase()
    executeCommand = [('parallel',executable+' -i '+inputFiles[index].getFilename())]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      this generates a new input file depending on which sampler has been chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    # TODO not currently maintained: dynamic event tree sampling for MOOSE applications

    found = False
    genericInput, genericOriInput = [], []
    # identify modifyable input files
    for i, inputFile in enumerate(currentInputFiles):
      inFile = inputFile.getAbsFile()
      if inFile.endswith(self.getInputExtension()):
        index = i
        found = True
      if inputFile.getType().lower() == "generic":
        genericInput.append(inputFile)
        genericOriInput.append(oriInputFiles[i])
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    # build output file names # TODO this probably isn't necessary thanks to file structures anymore
    outName = self.outputPrefix + currentInputFiles[index].getBase()
    # get a parser for the input file
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index].getAbsFile())
    # apply the requested modifications
    modifDict = self._expandVarNames(**Kwargs)
    ### set up output to place in a csv
    modifDict.append({'csv':'true', 'name':['Outputs', 'csv']})
    modifDict.append({'file_base': outName, 'name':['Outputs', 'file_base']})
    ### do modifications
    modified = parser.modifyOrAdd(modifDict)
    # write new input
    parser.printInput(currentInputFiles[index].getAbsFile(), modified)
    # I don't know what this is.
    self.vectorPPFound, self.vectorPPDict = parser.vectorPostProcessor()
    # or this.
    if genericInput:
      parser = GenericParser.GenericParser(genericInput)
      parser.modifyInternalDictionary(**Kwargs)
      parser.writeNewInput(genericInput, genericOriInput)

    return currentInputFiles

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      this method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output formats into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, returnOut, string, optional, present in case the root of the output file gets changed in this method.
    """
    returnOut = output
    if self.vectorPPFound:
      returnOut = self.__mergeTime(output,workingDir)[0]
    return returnOut

  def __mergeTime(self, output, workingDir):
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

  def _expandVarNames(self, **Kwargs):
    """
      This method will assure the full proper variable names are returned in a dictionary.
      @ In, Kwargs, dict, keyworded dictionary. Arguments include:
          - SampleVars, short name -> sampled value dictionary
      @ Out, requests, list(dict), dictionaries contain:
               'name': [path,to,name],
               short varname: var value
    """
    requests = []
    for var in Kwargs['SampledVars']:
      modifDict = {}
      # colon is used when we want to perturb element in the vector of given variable
      elemLoc = None
      if ":" in var:
        request, elemLoc = (v.strip() for v in var.split(':'))
      else:
        request = var
      if '|' not in request:
        # what modifications don't have the path in them?
        continue
      pathedName = request.split('|')
      modifDict['name'] = pathedName
      if elemLoc is not None:
        modifDict[pathedName[-1]] = (int(elemLoc), Kwargs['SampledVars'][var])
      else:
        modifDict[pathedName[-1]] = Kwargs['SampledVars'][var]
      requests.append(modifDict)
    return requests

  # TODO neither of these are used here, but in the RELAP7 interface. Maybe they should be moved?
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
    listDict = self._expandVarNames(**Kwargs)
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
