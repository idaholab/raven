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
    Converts input files to be compatible with merge request #269 (wangc/nd_dist_dev).  Removes the <data_filename> and <workingDir> node
    from the <MultivariateNormal> block, add <covariance> child node to the parent node <MultivariateNormal>.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  for child in simulation:
    if child.tag == 'Distributions':
      MVNNode = child.find('MultivariateNormal')
      if MVNNode is not None:
        dataFileNameNode = MVNNode.find('data_filename')
        covFileName = dataFileNameNode.text
        dataWorkingDirNode = MVNNode.find('workingDir')
        covFileDir = dataWorkingDirNode.text
        if '~' in covFileDir: covFileDir = os.path.expanduser(covFileDir)
        if os.path.isabs(covFileDir): covFileDir = covFileDir
        elif "runRelative" in dataWorkingDirNode.attrib:
          covFileDir = os.path.abspath(covFileName)
        else:
          if covFileDir == None: raise IOError('Relative working directory is requested but the given name is None' )
          covFileDir = os.path.join(os.getcwd(),covFileDir.strip())
        covFileName = os.path.join(covFileDir,covFileName.strip())
        MVNNode.remove(dataFileNameNode)
        MVNNode.remove(dataWorkingDirNode)
        covData = ''
        if os.path.isfile(covFileName):
          for line in file(covFileName,'r'):
            covData += line.rstrip() + ' '
        else:
          print 'Error! The following file is not exist: ', covFileName
        covNode = ET.Element('covariance')
        covNode.text = covData
        MVNNode.append(covNode)
      else:
        print 'No conversion needed'

  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
