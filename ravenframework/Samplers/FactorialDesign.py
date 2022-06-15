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
  This module contains the Factorial Design sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Grid import Grid
from ..contrib import pyDOE as doe
from ..utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class FactorialDesign(Grid):
  """
    Samples the model on a given (by input) set of points
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
    inputSpecification = super(FactorialDesign, cls).getInputSpecification()

    factorialSettingsInput = InputData.parameterInputFactory("FactorialSettings")
    algorithmTypeInput = InputData.parameterInputFactory("algorithmType", contentType=InputTypes.StringType)
    factorialSettingsInput.addSub(algorithmTypeInput)

    factorialSettingsInput.addSub(InputData.parameterInputFactory("gen", contentType=InputTypes.StringType))
    factorialSettingsInput.addSub(InputData.parameterInputFactory("genMap", contentType=InputTypes.StringType))

    inputSpecification.addSub(factorialSettingsInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Grid.__init__(self)
    self.printTag = 'SAMPLER FACTORIAL DESIGN'
    # accepted types. full = full factorial, 2levelFract = 2-level fractional factorial, pb = Plackett-Burman design. NB. full factorial is equivalent to Grid sampling
    self.acceptedTypes = ['full','2levelfract','pb'] # accepted factorial types
    self.factOpt       = {}                          # factorial options (type,etc)
    self.designMatrix  = None                        # matrix container

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ Out, None
    """
    #TODO remove using xmlNode
    Grid.localInputAndChecks(self,xmlNode, paramInput)
    factsettings = xmlNode.find("FactorialSettings")
    if factsettings == None:
      self.raiseAnError(IOError,'FactorialSettings xml node not found!')
    facttype = factsettings.find("algorithmType")
    if facttype == None:
      self.raiseAnError(IOError,'node "algorithmType" not found in FactorialSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedTypes:
      self.raiseAnError(IOError,' "type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedTypes))
    self.factOpt['algorithmType'] = facttype.text.lower()
    if self.factOpt['algorithmType'] == '2levelfract':
      self.factOpt['options'] = {}
      self.factOpt['options']['gen'] = factsettings.find("gen")
      self.factOpt['options']['genMap'] = factsettings.find("genMap")
      if self.factOpt['options']['gen'] == None:
        self.raiseAnError(IOError,'node "gen" not found in FactorialSettings xml node!!!')
      if self.factOpt['options']['genMap'] == None:
        self.raiseAnError(IOError,'node "genMap" not found in FactorialSettings xml node!!!')
      self.factOpt['options']['gen'] = self.factOpt['options']['gen'].text.split(',')
      self.factOpt['options']['genMap'] = self.factOpt['options']['genMap'].text.split(',')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()):
        self.raiseAnError(IOError,'number of variable in genMap != number of variables !!!')
      if len(self.factOpt['options']['gen']) != len(self.gridInfo.keys()):
        self.raiseAnError(IOError,'number of variable in gen != number of variables !!!')
      rightOrder = [None]*len(self.gridInfo.keys())
      if len(self.factOpt['options']['genMap']) != len(self.factOpt['options']['gen']):
        self.raiseAnError(IOError,'gen and genMap different size!')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()):
        self.raiseAnError(IOError,'number of gen attributes and variables different!')
      for ii,var in enumerate(self.factOpt['options']['genMap']):
        if var not in self.gridInfo.keys():
          self.raiseAnError(IOError,' variable "'+var+'" defined in genMap block not among the inputted variables!')
        rightOrder[self.axisName.index(var)] = self.factOpt['options']['gen'][ii]
      self.factOpt['options']['orderedGen'] = rightOrder
    if self.factOpt['algorithmType'] != 'full':
      self.externalgGridCoord = True
      for varname in self.gridInfo.keys():
        if len(self.gridEntity.returnParameter("gridInfo")[varname][2]) != 2:
          self.raiseAnError(IOError,'The number of levels for type '+
                        self.factOpt['algorithmType'] +' must be 2! In variable '+varname+ ' got number of levels = ' +
                        str(len(self.gridEntity.returnParameter("gridInfo")[varname][2])))
    else:
      self.externalgGridCoord = False

  def localGetInitParams(self):
    """
      Appends a given dictionary with class specific member variables and their
      associated initialized values.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Grid.localGetInitParams(self)
    for key,value in self.factOpt.items():
      if key != 'options':
        paramDict['Factorial '+key] = value
      else:
        for kk,val in value.items():
          paramDict['Factorial options '+kk] = val
    return paramDict

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    Grid.localInitialize(self)
    if   self.factOpt['algorithmType'] == '2levelfract':
      self.designMatrix = doe.fracfact(' '.join(self.factOpt['options']['orderedGen'])).astype(int)
    elif self.factOpt['algorithmType'] == 'pb':
      self.designMatrix = doe.pbdesign(len(self.gridInfo.keys())).astype(int)
    if self.designMatrix is not None:
      self.designMatrix[self.designMatrix == -1] = 0 # convert all -1 in 0 => we can access to the grid info directly
      self.limit = self.designMatrix.shape[0]        # the limit is the number of rows

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
    if self.factOpt['algorithmType'] == 'full':
      Grid.localGenerateInput(self,model, myInput)
    else:
      self.gridCoordinate = self.designMatrix[self.counter - 1][:].tolist()
      Grid.localGenerateInput(self,model, myInput)
#
#
