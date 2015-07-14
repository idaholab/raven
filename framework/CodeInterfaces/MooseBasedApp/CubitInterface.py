"""
Created July 14, 2015

@author: talbpaul
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('defalts',DeprecationWarning)

import os
import copy
import Files
from CodeInterfaceBaseClass import CodeInterfaceBase

class CubitInterface(CodeInterfaceBase):
  """This class provides the mean to perturb inputs for Cubit-generating Python scripts that
     are commonly used by the MOOSE herd."""

  def findInps(self,inputFiles):
    """Locates the input file to perturb.
    @ In, inputFiles, list of Files objects
    @Out, File, desired input file
    """
    if len(inputFiles)==1 and inputFiles[0].rsplit('.')=='py': return inputFiles[0] #targeted case
    for inFile in inputFiles:
      if inFile.getType()=='input': return inFile
    self.raiseAnError(IOError,'Cubit input file with type "input" was not found!')

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """Generates a command that generates a mesh.
    @ In, inputFiles, the perturbed input Files and pass-throughs
    @ In, executable, the execution string
    @ In, clargs, dictionary of CL arguments
    @ In, fargs, dictionary of file arguments
    @ Out, (String,string), exec command and output filename
    """
    inp = self.findInps(inputFiles)
    outfilename = 'cubit_out_'+inp.getBase()
    execcom = ('python '+executable+' -i '+inp.getAbsFile()) #FIXME how to set output file?

  def createNewInput(self,curInps,origInps,samplerType,**Kwargs):
    import CubitParser
    inp = self.findInps(curInps)
    moddict = [] #TODO
