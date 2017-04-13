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
  This module contains the Custom sampling strategy

  Created on May 21, 2016
  @author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .ForwardSampler import ForwardSampler
#Internal Modules End--------------------------------------------------------------------------------

class CustomSampler(ForwardSampler):
  """
    Custom Sampler
  """
  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    ForwardSampler.__init__(self)
    self.pointsToSample = {}
    self.infoFromCustom = {}
    self.addAssemblerObject('Source','1',True)
    self.printTag = 'SAMPLER CUSTOM'

  def _readMoreXMLbase(self,xmlNode):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ Out, None
    """
    self.readSamplerInit(xmlNode)
    for child in xmlNode:
      if child.tag == 'variable':
        self.toBeSampled[child.attrib['name']] = 'custom'
      if child.tag == 'Source'  :
        if child.attrib['class'] not in ['Files','DataObjects']: self.raiseAnError(IOError, "Source class attribute must be either 'Files' or 'DataObjects'!!!")
        if child.attrib['class'] == 'DataObjects' and child.attrib['type'] != 'PointSet': self.raiseAnError(IOError, "Source type attribute must be 'PointSet' if class attribute is 'DataObjects'!!!")
    if len(self.toBeSampled.keys()) == 0: self.raiseAnError(IOError,"no variables got inputted!!!!!!")

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed (in this case it is empty, since no distrubtions are needed and the Source is loaded automatically)
    """
    return {}

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    #it is called for the ensemble sampler
    for key, value in self.assemblerObjects.items():
      if key == 'Source':
        self.assemblerDict[key] =  []
        for interface in value: self.assemblerDict[key].append([interface[0],interface[1],interface[2],initDict[interface[0]][interface[2]]])
    if len(self.assemblerDict.keys()) == 0: self.raiseAnError(IOError,"No Source object has been found!")

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    # check the source
    if self.assemblerDict['Source'][0][0] == 'Files':
      csvFile = self.assemblerDict['Source'][0][3]
      csvFile.open(mode='r')
      headers = [x.replace("\n","") for x in csvFile.readline().split(",")]
      data = np.loadtxt(self.assemblerDict['Source'][0][3], dtype=np.float, delimiter=',', skiprows=1, ndmin=2)
      csvFile.close()
      for var in self.toBeSampled.keys():
        if var not in headers: self.raiseAnError(IOError, "variable "+ var+ " not found in the file "+csvFile.getFilename())
        self.pointsToSample[var] = data[:,headers.index(var)]
      if 'PointProbability' in headers: self.infoFromCustom['PointProbability'] = data[:,headers.index('PointProbability')]
      if 'ProbabilityWeight' in headers: self.infoFromCustom['ProbabilityWeight'] = data[:,headers.index('ProbabilityWeight')]
    else:
      dataObj = self.assemblerDict['Source'][0][3]
      for var in self.toBeSampled.keys():
        if var not in dataObj.getParaKeys('input') + dataObj.getParaKeys('output'): self.raiseAnError(IOError,"the variable "+ var+ " not found in "+dataObj.type +" "+dataObj.name)
        self.pointsToSample[var] = dataObj.getParam('input', var, nodeId = 'ending') if var in dataObj.getParaKeys('input') else dataObj.getParam('output', var, nodeId = 'ending')
      if 'PointProbability'  in dataObj.getParaKeys('metadata'): self.infoFromCustom['PointProbability'] = dataObj.getMetadata('PointProbability',nodeId='ending')
      if 'ProbabilityWeight' in dataObj.getParaKeys('metadata'): self.infoFromCustom['ProbabilityWeight'] = dataObj.getMetadata('ProbabilityWeight',nodeId='ending')
    self.limit = len(self.pointsToSample.values()[0])
    #TODO: add restart capability here!
    if self.restartData: self.raiseAnError(IOError,"restart capability not implemented for CustomSampler yet!")
#       self.counter+=len(self.restartData)
#       self.raiseAMessage('Number of points from restart: %i' %self.counter)
#       self.raiseAMessage('Number of points needed:       %i' %(self.limit-self.counter))

  def localGenerateInput(self,model,myInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    # create values dictionary
    for var in self.toBeSampled.keys():                   self.values[var] = self.pointsToSample[var][self.counter-1]
    if 'PointProbability' in self.infoFromCustom.keys():  self.inputInfo['PointProbability'] = self.infoFromCustom['PointProbability'][self.counter-1]
    if 'ProbabilityWeight' in self.infoFromCustom.keys(): self.inputInfo['ProbabilityWeight'] = self.infoFromCustom['ProbabilityWeight'][self.counter-1]
    self.inputInfo['SamplerType'] = 'Custom'
