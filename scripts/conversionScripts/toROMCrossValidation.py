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
    Converts input files to be compatible with merge request #1016
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  roms = models.findall('ROM')
  hybridModel = models.find('HybridModel')
  if roms is not None and hybridModel is not None:
    cv = hybridModel.find('CV')
    romHybrid = hybridModel.find('ROM')
    if cv is not None and romHybrid is not None:
      for rom in roms:
        if rom.attrib['name'] == romHybrid.text.strip():
          rom.append(cv)
          hybridModel.remove(cv)
    TestInfo = simulation.find('TestInfo')
    if TestInfo is not None:
      revisions = TestInfo.find('revisions')
      hasRev = True
      if revisions is None:
        revisions = ET.Element('revisions')
        hasRev = False
      rev = ET.Element('revision')
      rev.attrib['author'] = 'wangc'
      rev.attrib['date'] = '2019-07-09'
      rev.text = 'Move cross validation asssemble from HybridModel to ROM'
      revisions.append(rev)
      if not hasRev:
        TestInfo.append(revisions)
  else:
    return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
