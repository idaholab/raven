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
Created on Sept 10, 2017

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import shutil
import copy
import numpy as np
from collections import OrderedDict

from utils import xmlUtils, mathUtils
import MessageHandler # to give VariableGroups a messageHandler and handle messages

class RAVENparser():
  """
    Import the RAVEN input as xml tree, provide methods to add/change entries and print it back
  """
  def __init__(self, inputFile):
    """
      Constructor
      @ In, inputFile, string, input file name
      @ Out, None
    """
    self.printTag  = 'RAVEN_PARSER' # print tag
    self.inputFile = inputFile      # input file name
    self.outStreamsNames = {}       # {'outStreamName':[DataObjectName,DataObjectType]}
    self.varGroups = {}             # variable groups, names and values
    if not os.path.exists(inputFile):
      raise IOError(self.printTag+' ERROR: Not found RAVEN input file')
    try:
      tree = ET.parse(open(inputFile,'r'))
    except IOError as e:
      raise IOError(self.printTag+' ERROR: Input Parsing error!\n' +str(e)+'\n')
    self.tree = tree.getroot()

    # expand the ExteranlXML nodes
    cwd = os.path.dirname(inputFile)
    xmlUtils.expandExternalXML(self.tree,cwd)

    # get the NAMES of the variable groups
    variableGroupNode = self.tree.find('VariableGroups')
    if variableGroupNode is not None:
      # make a messageHandler and messageUsesr to handle variable group creation
      ## if made generally available to this parser, this can be relocated and used generally
      messageHandler = MessageHandler.MessageHandler()
      messageHandler.initialize({'verbosity':'quiet'})
      messageUser = MessageHandler.MessageUser()
      self.varGroups = xmlUtils.readVariableGroups(variableGroupNode,messageHandler,messageUser)

    # do some sanity checks
    sequence = [step.strip() for step in self.tree.find('.//RunInfo/Sequence').text.split(",")]
    # firstly no multiple sublevels of RAVEN can be handled now
    for code in self.tree.findall('.//Models/Code'):
      if 'subType' not in code.attrib:
        raise IOError(self.printTag+' ERROR: Not found subType attribute in <Code> XML blocks!')
      if code.attrib['subType'].strip() == 'RAVEN':
        raise IOError(self.printTag+' ERROR: Only one level of RAVEN runs are allowed (Not a chain of RAVEN runs). Found a <Code> of subType RAVEN!')
    # find steps and check if there are active outstreams (Print)
    foundOutStreams = False
    for step in self.tree.find('.//Steps'):
      if step.attrib['name'] in sequence:
        for role in step:
          if role.tag.strip() == 'Output':
            mainClass, subType = role.attrib['class'].strip(), role.attrib['type'].strip()
            if mainClass == 'OutStreams' and subType == 'Print':
              outStream = self.tree.find('.//OutStreams/Print[@name="'+role.text.strip()+ '"]'+'/source')
              if outStream is None:
                raise IOError(self.printTag+' ERROR: The OutStream of type "Print" named "'+role.text.strip()+'" has not been found!')
              dataObjectType = None
              linkedDataObjectPointSet = self.tree.find('.//DataObjects/PointSet[@name="'+outStream.text.strip()+ '"]')
              if linkedDataObjectPointSet is None:
                linkedDataObjectHistorySet = self.tree.find('.//DataObjects/HistorySet[@name="'+outStream.text.strip()+ '"]')
                if linkedDataObjectHistorySet is None:
                  # try dataset
                  linkedDataObjectHistorySet = self.tree.find('.//DataObjects/DataSet[@name="'+outStream.text.strip()+ '"]')
                  if linkedDataObjectHistorySet is None:
                    raise IOError(self.printTag+' ERROR: The OutStream of type "Print" named "'+role.text.strip()+'" is linked to not existing DataObject!')
                dataObjectType, xmlNode = "HistorySet", linkedDataObjectHistorySet
              else:
                dataObjectType, xmlNode = "PointSet", linkedDataObjectPointSet
              self.outStreamsNames[role.text.strip()] = [outStream.text.strip(),dataObjectType,xmlNode]
              foundOutStreams = True
    if not foundOutStreams:
      raise IOError(self.printTag+' ERROR: at least one <OutStreams> of type "Print" needs to be inputted in the active Steps!!')
    # Now we grep the paths of all the inputs the SLAVE RAVEN contains in the workind directory.
    self.workingDir = self.tree.find('.//RunInfo/WorkingDir').text.strip()
    # Find the Files
    self.slaveInputFiles = []
    filesNode = self.tree.find('.//Files')
    if filesNode is not None:
      for child in self.tree.find('.//Files'):
        subDirectory = child.attrib['subDirectory'] if 'subDirectory' in child.attrib else None
        if subDirectory:
          self.slaveInputFiles.append(os.path.expanduser(os.path.join(subDirectory,child.text.strip())))
        else:
          self.slaveInputFiles.append(os.path.expanduser(child.text.strip()))

    externalModels = self.tree.findall('.//Models/ExternalModel')
    if len(externalModels) > 0:
      for extModel in externalModels:
        if 'ModuleToLoad' in extModel.attrib:
          moduleToLoad = extModel.attrib['ModuleToLoad']
          if not moduleToLoad.endswith("py"):
            moduleToLoad += ".py"
          if self.workingDir not in moduleToLoad:
            self.slaveInputFiles.append(os.path.expanduser(os.path.join(self.workingDir,moduleToLoad)))
          else:
            self.slaveInputFiles.append(os.path.expanduser(moduleToLoad))
        else:
          if 'subType' not in extModel.attrib or len(extModel.attrib['subType']) == 0:
            raise IOError(self.printTag+' ERROR: ExternalModel "'+extModel.attrib['name']+'" does not have any attribute named "ModuleToLoad" or "subType" with an available plugin name!')

    externalFunctions = self.tree.findall('.//Functions/External')
    if len(externalFunctions) > 0:
      for extFunct in externalFunctions:
        if 'file' in extFunct.attrib:
          moduleToLoad = extFunct.attrib['file']
          if not moduleToLoad.endswith("py"):
            moduleToLoad += ".py"
          if self.workingDir not in moduleToLoad:
            self.slaveInputFiles.append(os.path.expanduser(os.path.join(self.workingDir,moduleToLoad)))
          else:
            self.slaveInputFiles.append(os.path.expanduser(moduleToLoad))
        else:
          raise IOError(self.printTag+' ERROR: Functions/External ' +extFunct.attrib['name']+ ' does not have any attribute named "file"!!')

  def returnOutstreamsNamesAnType(self):
    """
      Method to return the Outstreams names and linked DataObject name
      @ In, None
      @ Out, outStreamsNames, dict, the dictionary of outstreams of type print {'outStreamName':[DataObjectName,DataObjectType]}
    """
    return self.outStreamsNames

  def returnVarGroups(self):
    """
      Method to return the variable groups'
      @ In, None
      @ Out, varGroups, list, the list of var group names
    """
    return self.varGroups

  def copySlaveFiles(self,currentDirName):
    """
      Method to copy the slave input files
      @ In, currentDirName, str, the current directory (destination of the copy procedure)
      @ Out, None
    """
    # the dirName is actually in workingDir/StepName/prefix => we need to go back 2 dirs
    dirName = os.path.join(currentDirName, ".."+os.path.sep+".."+os.path.sep)
    # copy SLAVE raven files in case they are needed
    for slaveInput in self.slaveInputFiles:
      # full path
      slaveInputFullPath = os.path.abspath(os.path.join(dirName,slaveInput))
      # check if exists
      if os.path.exists(slaveInputFullPath):
        slaveInputBaseDir = os.path.dirname(slaveInput)
        slaveDir = os.path.join(currentDirName,slaveInputBaseDir.replace(currentDirName,""))
        if not os.path.exists(slaveDir):
          os.makedirs(slaveDir)
        shutil.copy(slaveInputFullPath,slaveDir)
      else:
        raise IOError(self.printTag+' ERROR: File "' +slaveInputFullPath+'" has not been found!!!')


  def printInput(self,rootToPrint,outfile=None):
    """
      Method to print out the new input
      @ In, rootToPrint, xml.etree.ElementTree.Element, the Element containing the input that needs to be printed out
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    xmlObj = xml.dom.minidom.parseString(ET.tostring(rootToPrint))
    inputAsString = xmlObj.toprettyxml()
    inputAsString = "".join([s for s in inputAsString.strip().splitlines(True) if s.strip()])
    if outfile==None:
      outfile =self.inputfile
    IOfile = open(outfile,'w+')
    IOfile.write(inputAsString)
    IOfile.close()

  def modifyOrAdd(self,modiDictionary={},save=True, allowAdd = False):
    """
      modiDictionary a dict of dictionaries of the required addition or modification
      {"variableToChange":value }
      @ In, modiDictionary, dict, dictionary of variables to modify
            syntax:
            {'Node|SubNode|SubSubNode:value1','Node|SubNode|SubSubNode@attribute:attributeValue|SubSubSubNode':value2
                      'Node|SubNode|SubSubNode@attribute':value3}
             TODO: handle added XML nodes
      @ In, save, bool, optional, True if the original tree needs to be saved
      @ In, allowAdd, bool, optional, True if the nodes that are not found should be added (additional piece of input)
      @ Out, returnElement, xml.etree.ElementTree.Element, the tree that got modified
    """
    if save:
      returnElement = copy.deepcopy(self.tree)            #make a copy if save is requested
    else:
      returnElement = self.tree                           #otherwise return the original modified

    for fullNode, val in modiDictionary.items():
      # might be comma-separated ("fully correlated") variables
      nodes = [x.strip() for x in fullNode.split(',')]
      for node in nodes:
        # make sure node is XML-tree-parsable
        if "|" not in node:
          raise IOError(self.printTag+' ERROR: the variable '+node.strip()+' does not contain "|" separator and can not be handled!!')
        changeTheNode = True
        allowAddNodes, allowAddNodesPath = [], OrderedDict()
        if "@" in node:
          # there are attributes that are needed to locate the node
          splittedComponents = node.split("|")
          # check the first
          pathNode = './'
          attribName = ''
          for cnt, subNode in enumerate(splittedComponents):
            splittedComp = subNode.split("@")
            component = splittedComp[0]
            attribPath = ""
            attribConstruct = OrderedDict()
            if "@" in subNode:
              # more than an attribute locator
              for attribComp in splittedComp[1:]:
                attribValue = None
                if ":" in attribComp.strip():
                  # it is a locator
                  attribName  = attribComp.split(":")[0].strip()
                  attribValue = attribComp.split(":")[1].strip()
                  attribPath +='[@'+attribName+('="'+attribValue+'"]')
                else:
                  # it is actually the attribute that needs to be changed
                  # check if it is the last component
                  if cnt+1 != len(splittedComponents):
                    raise IOError(self.printTag+' ERROR: the variable '+node.strip()+' follows the syntax "Node|SubNode|SubSubNode@attribute"'+
                                                ' but the attribute is not the last component. Please check your input!')
                  attribName = attribComp.strip()
                  attribPath +='[@'+attribName+']'
                if allowAdd:
                  attribConstruct[attribName]  = attribValue
            pathNode += "/" + component.strip()+attribPath
            if allowAdd:
              if len(returnElement.findall(pathNode)) > 0:
                allowAddNodes.append(pathNode)
              else:
                allowAddNodes.append(None)
              allowAddNodesPath[component.strip()] = attribConstruct
          if pathNode.endswith("]") and list(attribConstruct.values())[-1] is None:
            changeTheNode = False
          else:
            changeTheNode = True
        else:
          # there are no attributes that are needed to track down the node to change
          pathNode = './/' + node.replace("|","/").strip()
          if allowAdd:
            pathNodeTemp = './'
            for component in node.replace("|","/").split("/"):
              pathNodeTemp += '/'+component
              if len(returnElement.findall(pathNodeTemp)) > 0:
                allowAddNodes.append(pathNodeTemp)
              else:
                allowAddNodes.append(None)
              allowAddNodesPath[component.strip()] = None
        # look for the node with XPath directives
        foundNodes = returnElement.findall(pathNode)
        if len(foundNodes) > 1:
          raise IOError(self.printTag+' ERROR: multiple nodes have been found corresponding to path -> '+node.strip()+'. Please use the attribute identifier "@" to nail down to a specific node !!')
        if len(foundNodes) == 0 and not allowAdd:
          raise IOError(self.printTag+' ERROR: no node has been found corresponding to path -> '+node.strip()+'. Please check the input!!')
        if len(foundNodes) == 0:
          # this means that the allowAdd is true (=> no error message has been raised)
          indexFirstUnknownNode = allowAddNodes.index(None)
          if indexFirstUnknownNode == 0:
            raise IOError(self.printTag+' ERROR: at least the main XML node should be present in the RAVEN template input -> '+node.strip()+'. Please check the input!!')
          getFirstElement = returnElement.findall(allowAddNodes[indexFirstUnknownNode-1])[0]
          for i in range(indexFirstUnknownNode,len(allowAddNodes)):
            nodeWithAttributeName = allowAddNodesPath.keys()[i]
            if not allowAddNodesPath[nodeWithAttributeName]:
              subElement =  ET.Element(nodeWithAttributeName)
            else:
              subElement =  ET.Element(nodeWithAttributeName, attrib=allowAddNodesPath[nodeWithAttributeName])
            getFirstElement.append(subElement)
            getFirstElement = subElement
          # in the event of vector entries, handle those here
          if mathUtils.isSingleValued(val):
            val = str(val).strip()
          else:
            if len(val.shape) > 1:
              raise IOError(self.printTag+'ERROR: RAVEN interface is not prepared to handle matrix value passing yet!')
            val = ','.join(str(i) for i in val)
          if changeTheNode:
            subElement.text = val
          else:
            subElement.attrib[attribConstruct.keys()[-1]] = val

        else:
          nodeToChange = foundNodes[0]
          pathNode     = './/'
          # in the event of vector entries, handle those here
          if mathUtils.isSingleValued(val):
            val = str(val).strip()
          else:
            if len(val.shape) > 1:
              raise IOError(self.printTag+'ERROR: RAVEN interface is not prepared to handle matrix value passing yet!')
            val = ','.join(str(i) for i in val)
          if changeTheNode:
            nodeToChange.text = val
          else:
            nodeToChange.attrib[attribName] = val
    return returnElement
