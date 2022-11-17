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
import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os

def convert(tree,fileName=None):
  """
    Converts the file type of miscellaneous inputs in MAMMOTH interface MultiRun to
    ancillary input as of merge request 728.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @ Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  # Check if this is RAVEN input
  if simulation.tag!='Simulation' and simulation.tag!='ExternalModel': return tree
  # Check if this is MAMMOTH interface input
  mammothInput = False
  mammothMultiRunNames = []
  for code in simulation.iter('Code'):
    if code.attrib['subType'].lower == 'mammoth':
      mammothInput = True
      mammothMultiRunNames.append(code.attrib['name'])
    for codeChild in code:
      if codeChild.tag == 'alias':
        checkAliases = True
        break
  if not mammothInput:
    return tree
  # Find all Inputs with class=Files from Mammoth MultiRun
  inputFileNames = []
  for mammothMultiRun in mammothMultiRunNames:
    for iput in simulation.findall("./Steps/MultiRun/[Model='"+mammothMultiRun+"']/Input"):
      if iput.attrib['class'] == 'Files':
        inputFileNames.append(iput.text)
  # Change any found Mammoth MultiRun Input Files' with no type to ancillaryInput
  for inputFileName in inputFileNames:
    for blankInputFileType in simulation.findall("./Files/Input/[@name='"+inputFileName+"']/[@type='']"):
      blankInputFileType.set('type', 'ancillaryInput')

  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
