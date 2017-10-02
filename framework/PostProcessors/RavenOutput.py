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
Created on July 10, 2013

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import xmlUtils
import Files
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class RavenOutput(PostProcessor):
  """
    This postprocessor collects the outputs of RAVEN runs (XML format) and turns entries into a PointSet
    Someday maybe it should do history sets too.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inputSpecification = super(RavenOutput, cls).getInputSpecification()

    ## TODO: Fill this in with the appropriate tags

    return inputSpecification


  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR RAVENOUTPUT'
    self.IDType = 'int'
    self.files = {}
      # keyed by ID, which gets you to... (self.files[ID])
      #   name: RAVEN name for file (from input)
      #   fileObject: FileObject
      #   paths: {varName:'path|through|xml|to|var'}
    self.dynamic = False #if true, reading in pivot as input and values as outputs

  def initialize(self,runInfo,inputs,initDict):
    """
      Method to initialize pp
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)
    #assign File objects to their proper place
    for id,fileDict in self.files.items():
      found = False
      for i,input in enumerate(inputs):
        #skip things that aren't files
        if not isinstance(input,Files.File):
          continue
        #assign pointer to file object if match found
        if input.name == fileDict['name']:
          self.files[id]['fileObject'] = input
          found = True
          break
      if not found:
        self.raiseAnError(IOError,'Did not find file named "%s" among the Step inputs!' % (input.name))

  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    # paramInput = RavenOutput.getInputSpecification()()
    # paramInput.parseNode(xmlNode)

    #check if in dynamic mode; default is False
    dynamicNode = xmlNode.find('dynamic')
    if dynamicNode is not None:
      #could specify as true/false or just have the node present
      text = dynamicNode.text
      if text is not None:
        if text not in utils.stringsThatMeanFalse():
          self.dynamic = True
      else:
        self.dynamic = True
    numberOfSources = 0
    for child in xmlNode:
      #if dynamic, accept a single file as <File ID="1" name="myOut.xml">
      #if not dynamic, accept a list of files
      if child.tag == 'File':
        numberOfSources += 1
        if 'name' not in child.attrib.keys():
          self.raiseAnError(IOError,'Each "File" must have an associated "name"; missing for',child.tag,child.text)
        #make sure you provide an ID and a file name
        if 'ID' not in child.attrib.keys():
          id = 0
          while id in self.files.keys():
            id += 1
          self.raiseAWarning(IOError,'Each "File" entry must have an associated "ID"; missing for',child.tag,child.attrib['name'],'so ID is set to',id)
        else:
          #assure ID is a number, since it's going into a data object
          id = child.attrib['ID']
          try:
            id = float(id)
          except ValueError:
            self.raiseAnError(IOError,'ID for "'+child.text+'" is not a valid number:',id)
          #if already used, raise an error
          if id in self.files.keys():
            self.raiseAnError(IOError,'Multiple File nodes have the same ID:',child.attrib('ID'))
        #store id,filename pair
        self.files[id] = {'name':child.attrib['name'].strip(), 'fileObject':None, 'paths':{}}
        #user provides loading information as <output name="variablename">ans|pearson|x</output>
        for cchild in child:
          if cchild.tag == 'output':
            #make sure you provide a label for this data array
            if 'name' not in cchild.attrib.keys():
              self.raiseAnError(IOError,'Must specify a "name" for each "output" block!  Missing for:',cchild.text)
            varName = cchild.attrib['name'].strip()
            if varName in self.files[id]['paths'].keys():
              self.raiseAnError(IOError,'Multiple "output" blocks for "%s" have the same "name":' %self.files[id]['name'],varName)
            self.files[id]['paths'][varName] = cchild.text.strip()
    #if dynamic, only one File can be specified currently; to fix this, how do you handle different-lengthed times in same data object?
    if self.dynamic and numberOfSources > 1:
      self.raiseAnError(IOError,'For Dynamic reading, only one "File" node can be specified!  Got',numberOfSources,'nodes.')
    # check there are entries for each
    if len(self.files)<1:
      self.raiseAWarning('No files were specified to read from!  Nothing will be done...')
    # if no outputs listed, remove file from list and warn
    toRemove=[]
    for id,fileDict in self.files.items():
      if len(fileDict['paths'])<1:
        self.raiseAWarning('No outputs were specified for File with ID "%s"!  No extraction will be performed for this file...' %str(id))
        toRemove.append(id)
    for rem in toRemove:
      del self.files[id]

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    # outputs are realizations that will got into data object
    outputDict={'realizations':[]}
    if self.dynamic:
      #outputs are basically a point set with pivot as input and requested XML path entries as output
      fileName = self.files.values()[0]['fileObject'].getAbsFile()
      root,_ = xmlUtils.loadToTree(fileName)
      #determine the pivot parameter
      pivot = root[0].tag
      numPivotSteps = len(root)
      #read from each iterative pivot step
      for p,pivotStep in enumerate(root):
        realization = {'inputs':{},'outputs':{},'metadata':{'loadedFromRavenFile':fileName}}
        realization['inputs'][pivot] = float(pivotStep.attrib['value'])
        for name,path in self.files.values()[0]['paths'].items():
          desiredNode = self._readPath(pivotStep,path,fileName)
          realization['outputs'][name] = float(desiredNode.text)
        outputDict['realizations'].append(realization)
    else:
      # each ID results in a realization for the requested attributes
      for id,fileDict in self.files.items():
        realization = {'inputs':{'ID':id},'outputs':{},'metadata':{'loadedFromRavenFile':str(fileDict['fileObject'])}}
        for varName,path in fileDict['paths'].items():
          #read the value from the file's XML
          root,_ = xmlUtils.loadToTree(fileDict['fileObject'].getAbsFile())
          desiredNode = self._readPath(root,path,fileDict['fileObject'].getAbsFile())
          realization['outputs'][varName] = float(desiredNode.text)
        outputDict['realizations'].append(realization)
    return outputDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")

    outputDictionary = evaluation[1]
    realizations = outputDictionary['realizations']
    for real in realizations:
      for key in output.getParaKeys('inputs'):
        if key not in real['inputs'].keys():
          self.raiseAnError(RuntimeError, 'Requested input variable '+key+' has not been extracted. Check the consistency of your input')
        output.updateInputValue(key,real['inputs'][key])
      for key in output.getParaKeys('outputs'):
        if key not in real['outputs'].keys():
          self.raiseAnError(RuntimeError, 'Requested output variable '+key+' has not been extracted. Check the consistency of your input')
        output.updateOutputValue(key,real['outputs'][key])
      for key,val in real['metadata'].items():
        output.updateMetadata(key,val)

  def _readPath(self,root,inpPath,fileName):
    """
      Reads in values from XML tree.
      @ In, root, xml.etree.ElementTree.Element, node to start from
      @ In, inPath, string, |-separated list defining path from root (not including root)
      @ In, fileName, string, used in error
      @ Out, desiredNode, xml.etree.ElementTree.Element, desired node
    """
    #improve path format
    path = '|'.join(c.strip() for c in inpPath.strip().split('|'))
    desiredNode = xmlUtils.findPath(root,path)
    if desiredNode is None:
      self.raiseAnError(RuntimeError,'Did not find "%s|%s" in file "%s"' %(root.tag,path,fileName))
    return desiredNode
