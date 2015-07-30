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
import MessageHandler
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
      if inputFile.endswith(self.getInputExtension()):
        found = True
        break
    if not found: self.raiseAnError(IOError,'None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    executeCommand = (executable+ ' -batch ' + os.path.split(inputFiles[index])[1])
    self.raiseADebug(executeCommand)
    return executeCommand, self.outputfile
    # CHECK THIS DEFINITION

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
      if inputFile.endswith(self.getInputExtension()):
        break
    parser = CUBITparser.CUBITparser(self.messageHandler, currentInputFiles[index])
    # Copy dictionary of sampled vars sent to interface and change name ot alias (if it exists)
    sampledDict = copy.deepcopy(Kwargs['SampledVars'])
    for alias,var in Kwargs['alias'].items():
      sampledDict[var] = Kwargs['SampledVars'][alias]
      del sampledDict[alias]
    parser.modifyInternalDictionary(**sampledDict)
    parser.modifyInternalDictionary(**Kwargs['SampledVars'])
    # Copy original mesh generation input file and write new input from sampled vars
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.copy(currentInputFiles)
    newInputFiles[index] = os.path.join(os.path.split(temp)[0], os.path.split(temp)[1].split('.')[0] \
        +'_'+Kwargs['prefix']+'.'+os.path.split(temp)[1].split('.')[1])
    self.outputfile = ('mesh~'+os.path.split(temp)[1].split('.')[0])+'_'+Kwargs['prefix']
    Kwargs['SampledVars']['Cubit|out_name'] = "\"'"+self.outputfile+".e'\""
    parser.writeNewInput(newInputFiles[index])
    return newInputFiles
    # CHECK THIS DEFINITION

  def getInputExtension(self):
    """Returns the output extension of input files to be perturbed as a string."""
    return(".jou")
