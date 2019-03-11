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
 Created on Sept 20, 2018

 @author: alfoa
"""
import xml.etree.ElementTree as ET
from collections import OrderedDict
import sys

def writeLatexFromDictionaryOfRequirements(applicationName,requirementGroups,outputFileName):
  """
    Method to read the XML containing the requirements for a certain application
    @ In, outputFileName, string, the latex file name
    @ In, applicationName, string, the application name
    @ In, requirementGroups, dict, the dictionary containing the requirements
    @ Out, None
  """
  sections = ["section","subsection","subsubsection","paragraph"]
  indexSec = 0
  fileObject = open(outputFileName,"w+")
  fileObject.write("\\"+sections[0].strip()+"{System Requirements: " +applicationName.strip()+"}\n")
  for group, value in requirementGroups.items():
    fileObject.write("\\"+sections[1].strip()+"{" +group.strip()+"}\n")
    for reqSet, requirements in value.items():
      # construct table
      fileObject.write("\\"+sections[2].strip()+"{" +reqSet.strip()+"}\n")
      # start write the requirements for this set
      for req, content in requirements.items():
        fileObject.write("\\"+sections[3].strip()+"{" +req.strip()+"} \n")
        fileObject.write(content['description'].strip() + "\n")
  fileObject.close()


def readRequirementsXml(fileName):
  """
    Method to read the XML containing the requirements for a certain application
    @ In, fileName, string, the xml file name
    @ Out, (applicationName, requirementGroups), tuple, the application name and the dictionary containing the requirements
  """
  try:
    root = ET.parse(fileName).getroot()
  except Exception as e:
    print('file :'+fileName+'\nXML Parsing error!',e,'\n')
    raise IOError('\nError in '+fileName+'\n')
  if root.tag != 'requirements_specification':
    raise IOError('The root node is not requirements_specification for file '+fileName+'\n')
  applicationName = root.attrib.get("application")
  if applicationName is None:
    raise IOError('the requirements_specification node must contain the attribute "application"!')
  requirementGroups = OrderedDict()
  allGroups = root.findall('.//requirement_group')
  if len(allGroups) == 0:
    raise IOError('No requirement_group node has been found in file  '+fileName+'\n')
  for group in allGroups:
    groupName = group.attrib.get("id")
    if groupName is None:
      raise IOError('the attribute "id" must be present for any <requirement_group>!')
    if groupName in requirementGroups:
      raise IOError('the requirement_group with "id"='+groupName+ ' has been already inputted!')
    requirementGroups[groupName] = OrderedDict()
    # find requirement sets
    allRequirementSets = group.findall('.//requirement_set')
    if len(allRequirementSets) == 0:
      raise IOError('No requirement_set node has been found for requirement_group "'+groupName+'"')
    for rset in allRequirementSets:
      setName = rset.attrib.get("caption")
      if setName is None:
        raise IOError('the attribute "caption" must be present for any <requirement_set>!')
      if setName in requirementGroups[groupName]:
        raise IOError('the requirement_set with "caption"='+ setName + ' in requirement_group ' +   groupName+ ' has been already inputted!')
      requirementGroups[groupName][setName] = OrderedDict()
      # find all requirements for this set
      allRequirements = rset.findall('.//requirement')
      if len(allRequirements) == 0:
        raise IOError('No requirement node has been found for requirement_group "'+groupName+'" in requirement_set "'+setName+'"!')
      for req in allRequirements:
        reqName = req.attrib.get("id_code")
        if reqName is None:
          raise IOError('the attribute "id_code" must be present for any <requirement>!')
        if reqName in requirementGroups[groupName][setName]:
          raise IOError('the requirement with "id_code"='+reqName+' in the requirement_set with "caption"='+ setName + ' in requirement_group ' +   groupName+ ' has been already inputted!')
        requirementGroups[groupName][setName][reqName] = OrderedDict.fromkeys(['description','source'])
        description = req.find('.//description')
        source = req.find('.//source')
        if description is None:
          raise IOError('the <description> node has not been found in the requirement with "id_code"='+reqName+
                        ' in the requirement_set with "caption"='+ setName + ' in requirement_group ' +groupName+
                        ' has been already inputted!')
        requirementGroups[groupName][setName][reqName]['description'] = description.text.strip()
        if source is not None:
          requirementGroups[groupName][setName][reqName]['source'] = source.text.strip().split(";")
  return applicationName, requirementGroups

if __name__ == '__main__':
  try:
    index = sys.argv.index("-i")
    requirementFile = sys.argv[index+1].strip()
  except ValueError:
    raise ValueError("Not found command line argument -i")
  try:
    index = sys.argv.index("-o")
    outputLatex = sys.argv[index+1].strip()
  except ValueError:
    raise ValueError("Not found command line argument -o")

  app, groups = readRequirementsXml(requirementFile)
  writeLatexFromDictionaryOfRequirements(app,groups,outputLatex)
