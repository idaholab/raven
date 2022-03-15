# Copyright 2017 University of Rome La Sapienza and Battelle Energy Alliance, LLC
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
  Created on April 18, 2017
  @author: Matteo Donorio (University of Rome La Sapienza)
  This is a utility class for the creation of the command.
  It is not an interface that can work on its own
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import os
from ravenframework.utils import utils

class MelgenApp():
  """
    This class is the CodeInterface for MELGEN (a sub-module of Melcor)
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self.inputExtensions = ['i','inp']

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None,preExec=None):
    """
      Generate a command to run MELGEN (a sub-module of Melcor)
      Collects all the clargs and the executable to produce the command-line call.
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
    outputfile = 'OUTPUT_MELCOR'
    found = False

    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.inputExtensions:
        found = True
        break
    if not found:
      raise IOError("Unknown input extension. Expected input file extensions are "+ ",".join(self.inputExtensions))
    if clargs:
      precommand = executable + clargs['text']
    else     :
      precommand = executable
    executeCommand = [('serial',precommand + ' '+inputFiles[index].getFilename())]
    returnCommand = executeCommand, outputfile
    return returnCommand



