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

def convert(tree,fileName=None):
  """
    Converts input files to be compatible with merge request #785:
      Where ARMA exists, removes <Pmax>, <Pmin>, <Qmax>, and <Qmin>, and adds <P> and <Q>
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  if models is not None:
    for child in models:
      if child.tag == 'ROM' and child.attrib['subType'] == 'SyntheticHistory':
        arma = child.find('arma')
        if arma is None:
          return tree
        # store outdated nodes if existing
        signalLag = arma.find('SignalLag')
        noiseLag  = arma.find('NoiseLag')
        # replace outdated nodes with new node
        if signalLag is not None:
          newP = ET.Element('P')
          newP.text = signalLag.text
          arma.remove(signalLag)
          arma.append(newP)
        if noiseLag is not None:
          newQ = ET.Element('Q')
          newQ.text = noiseLag.text
          arma.remove(noiseLag)
          arma.append(newQ)
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
