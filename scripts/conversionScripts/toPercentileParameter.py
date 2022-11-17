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
    Converts percentiles to be a tag with a parameter, instead of just a
    tag with the percentile inside it.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  for child in simulation.findall("./Models/PostProcessor[@subType='BasicStatistics']"):
    toRemove = []
    toAdd = []
    for childchild in child:
      if childchild.tag.startswith('percentile'):
        if "_" in childchild.tag:
          oldPercentileNode = childchild
          percent = oldPercentileNode.tag.split("_")[1]
          toRemove.append(oldPercentileNode)
          newPercentileNode = ET.Element('percentile')
          newPercentileNode.text = oldPercentileNode.text.replace("%","")
          newPercentileNode.attrib['percent'] = percent
          toAdd.append(newPercentileNode)
    #Need to do this here, because can't do it while iterating.
    for r in toRemove:
      child.remove(r)
    for a in toAdd:
      child.append(a)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
