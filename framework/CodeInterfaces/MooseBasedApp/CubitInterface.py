"""
created on July 16, 2015

@author: tompjame
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

import os
import copy
import sys
import re
import collections
from utils import toBytes, toStrish, compare
from CodeInterfaceBaseClass import CodeInterfaceBase

class CubitInterface(CodeInterfaceBase):
  """This class is used to couple raven to Cubit journal files for input to generate
     meshes (usually to run in another simulation)"""

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None):
    """Generate a command to run cubit using an input with sampled variables to output
       the perturbed mesh as an exodus file.
       @ In, inputFiles, the perturbed input files (list of Files) along with pass-through files from RAVEN.
       @ In, executable, the Cubit executable to run (string)
       @ In, clargs, command line arguments
       @ In, fargs, file-based arguments
       @Out, (string, string), execution command and output file name
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if '.'+inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    executeCommand = (executable+ ' -batch ' + inputFiles[index].getFilename())
    return executeCommand, self.outputfile

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """Generates new perturbed input files.
       @ In, currentInputFiles, list of Files objects, most recently perturbed files
       @ In, originInputFiles, the template input files originally shown
       @ In, samplerType, the sampler type used (not used in this algorithm)
       @ In, Kwargs, dictionary of key-val pairs
       @Out, list of perturbed files
    """
    import CUBITparser
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getExt() == self.getInputExtension():
        break
    parser = CUBITparser.CUBITparser(currentInputFiles[index])
    # Copy original mesh generation input file and write new input from sampled vars
    newInputFiles = copy.deepcopy(currentInputFiles)
    newInputFiles[index].close()
    newInputFiles[index].setBase(currentInputFiles[index].getBase()+'_'+Kwargs['prefix'])
    #newInputFiles[index] = os.path.join(os.path.split(temp)[0], os.path.split(temp)[1].split('.')[0] \
    #    +'_'+Kwargs['prefix']+'.'+os.path.split(temp)[1].split('.')[1])
    self.outputfile = 'mesh~'+newInputFiles[index].getBase()
    Kwargs['SampledVars']['Cubit|out_name'] = "\"'"+self.outputfile+".e'\""
    # Copy dictionary of sampled vars sent to interface and change name of alias (if it exists)
    sampledDict = copy.deepcopy(Kwargs['SampledVars'])
    for alias,var in Kwargs['alias'].items():
      sampledDict[var] = Kwargs['SampledVars'][alias]
      del sampledDict[alias]
    parser.modifyInternalDictionary(**sampledDict)
    # Write new input files
    parser.writeNewInput(newInputFiles[index].getAbsFile())
    return newInputFiles

  def getInputExtension(self):
    """Returns the output extension of input files to be perturbed as a string."""
    return(".jou")

  def finalizeCodeOutput(self, command, output, workingDir):
    """Cleans up files in the working directory that are not needed after the run
       @ In, command, (string), command used to run the just ended job
       @ In, output, (string), the Output name root
       @ In, workingDir, (string), the current working directory
    """
    # Append wildcard strings to workingDir for files wanted to be removed
    cubitjour_files = os.path.join(workingDir,'cubit*')
    exodus_meshes = os.path.join(workingDir,'*.e')
    # Inform user which files will be removed
    print('files being removed:\n'+cubitjour_files+'\n'+exodus_meshes)
    # Remove Cubit generated journal files
    self.rmUnwantedFiles(cubitjour_files)
    # Remove exodus mesh (.e) files TODO create an optional node to allow user to keep .e files
    self.rmUnwantedFiles(exodus_meshes)

  def rmUnwantedFiles(self, path_to_files):
    """Method to remove unwanted files after completing the run
       @ In, path_to_files, (string), path to the files to be removed
    """
    success = os.system('rm '+path_to_files)
    if success != 0:
      print(success,"Error removing ",path_to_files)
