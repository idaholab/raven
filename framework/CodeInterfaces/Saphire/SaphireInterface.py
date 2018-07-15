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
Created on July 12, 2018

@author: wangc

comments: Interface for SAPHIRE Simulation
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy

from GenericCodeInterface import GenericCode
from SaphireData import SaphireData

class Saphire(GenericCode):
  """
    SAPHIRE Interface.
  """
  def __init__(self):
    """
      Initializes the SAPHIRE code interface
      @ In, None
      @ Out, None
    """
    GenericCode.__init__(self)
    self.codeOutputs = {} # the root of the outputs

  def addDefaultExtension(self):
    """
      The Generic code interface does not accept any default input types.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['mac'])

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this class and initialize some members
      based on inputs
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node
      @ Out, None
    """
    GenericCode._readMoreXML(self, xmlNode)
    for outNode in xmlNode.findall('codeOutput'):
      outType = outNode.get('type').strip()
      outName = outNode.text.strip()
      self.codeOutputs[outName] = outType

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run. It will convert the SAPHIRE outputs into RAVEN
      compatible csv file.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
    """
    filesIn = []
    for outName, outType in self.codeOutputs.items():
      filesIn.append((os.path.join(workingDir, outName), outType))
    outputParser = SaphireData(filesIn)
    outputParser.writeCSV(os.path.join(workingDir, output))

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by RAVEN at the end of each run if the return code is == 0.
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    return failure
