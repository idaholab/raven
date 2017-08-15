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
    Converts input files to be compatible with merge request 191, where some
    of the less used mpi mode features were moved to mpilegacy
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  runinfo = simulation.find('RunInfo')
  if runinfo is None:
    return tree #No RunInfo, no need to change.
  if runinfo.find('mode') is not None:
    mode = runinfo.find('mode')
    foundLegacy = False
    for child in mode:
      if child.tag.lower() == "nosplitnode":
        foundLegacy = True
      if child.tag.lower() == "limitnode":
        foundLegacy = True
    if foundLegacy and mode.text.strip() == "mpi":
      print("found mpi legacy features in mpi mode")
      mode.text = "mpilegacy"
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
