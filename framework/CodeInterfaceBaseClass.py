"""
Created on Jan 28, 2015

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#External Modules------------------------------------------------------------------------------------
import abc
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils    import returnPrintTag, metaclass_insert, stringsThatMeanTrue
#Internal Modules End--------------------------------------------------------------------------------

class CodeInterfaceBase(metaclass_insert(abc.ABCMeta,object)):
  """
  Code Interface base class. This class should be the base class for all the code interfaces.
  In this way some methods are forced to be implemented and some automatic checking features
  are available (checking of the inputs if no executable is available), etc.
  NOTE: As said, this class SHOULD be the base class of the code interfaces. However, the developer
        of a newer code interface can decide to avoid to inherit from this class if he does not want
        to exploit the automatic checking of the code interface's functionalities
  """
  def genCommand(self,inputFiles,executable,flags=None):
    """
      This method is used to retrieve the command (in string format) needed to launch the Code.
      This method checks a bolean enviroment variable called 'RAVENinterfaceCheck':
      if true, the subcodeCommand is going to be overwritten with an empty string. In this way we can check the functionality of the interface without having an executable.
      See Driver.py to understand how this Env variable is set
      @ In , inputFiles, list, List of input files (lenght of the list depends on the number of inputs have been added in the Step is running this code)
      @ In , executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In , flags, string, a string containing the flags the user can specify in the input (e.g. under the node <Code> <executable> <flags>-u -r</flags> </executable> </Code>)
      @ Out, string, string containing the full command that the internal JobHandler is going to use to run the Code this interface refers to
    """
    subcodeCommand,outputfileroot = self.generateCommand(inputFiles,executable,flags)
    if os.environ['RAVENinterfaceCheck'].lower() in stringsThatMeanTrue(): return '',outputfileroot
    return subcodeCommand,outputfileroot

  @abc.abstractmethod
  def generateCommand(self,inputFiles,executable,flags=None):
    """
      This method is used to retrieve the command (in string format) needed to launch the Code.
      @ In , inputFiles, list, List of input files (lenght of the list depends on the number of inputs have been added in the Step is running this code)
      @ In , executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In , flags, string, a string containing the flags the user can specify in the input (e.g. under the node <Code> <executable> <flags>-u -r</flags> </executable> </Code>)
      @ Out, string, string containing the full command that the internal JobHandler is going to use to run the Code this interface refers to
    """
    return

  @abc.abstractmethod
  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
    This method is used to generate an input based on the information passed in.
    @ In , currentInputFiles, list,  list of current input files (input files from last this method call)
    @ In , oriInputFiles, list, list of the original input files
    @ In , samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
    @ In , Kwargs, kwarded dictionary, dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
           where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
    @ Out, newInputFiles, list of newer input files, list of the new input files (modified and not)
    """
    pass

  def finalizeCodeOutput(self,currentInputFiles,output,workingDir):
    """
    this method is called by the RAVEN code at the end of each run (if the method is present).
    It can be used for those codes, that do not create CSV files to convert the whaterver output formato into a csv
    @ currentInputFiles, currentInputFiles, list,  list of current input files (input files from last this method call)
    @ output, Input, the Output name root (string)
    @ workingDir, Input, actual working dir (string)
    @ return string, optional, present in case the root of the output file gets changed in this method.
    """
    return output
