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
  This module contains the Response Surface Design sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Grid import Grid
from ..contrib import pyDOE as doe
from ..utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class ResponseSurfaceDesign(Grid):
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
    inputSpecification = super(ResponseSurfaceDesign, cls).getInputSpecification()
    responseSurfaceDesignSettingsInput = InputData.parameterInputFactory("ResponseSurfaceDesignSettings")

    responseSurfaceDesignSettingsInput.addSub(InputData.parameterInputFactory("algorithmType", contentType=InputTypes.StringType))
    responseSurfaceDesignSettingsInput.addSub(InputData.parameterInputFactory("ncenters", contentType=InputTypes.IntegerType))
    responseSurfaceDesignSettingsInput.addSub(InputData.parameterInputFactory("centers", contentType=InputTypes.StringListType))
    responseSurfaceDesignSettingsInput.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.StringType))
    responseSurfaceDesignSettingsInput.addSub(InputData.parameterInputFactory("face", contentType=InputTypes.StringType))

    inputSpecification.addSub(responseSurfaceDesignSettingsInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Grid.__init__(self)
    self.limit    = 1
    self.printTag        = 'SAMPLER RESPONSE SURF DESIGN'
    self.respOpt         = {}                                    # response surface design options (type,etc)
    self.designMatrix    = None                                  # matrix container
    self.bounds          = {}                                    # dictionary of lower and upper
    self.mapping         = {}                                    # mapping between designMatrix coordinates and position in grid
    self.minNumbVars     = {'boxbehnken':3,'centralcomposite':2} # minimum number of variables
    # dictionary of accepted types and options (required True, optional False)
    self.acceptedOptions = {'boxbehnken':['ncenters'], 'centralcomposite':['centers','alpha','face']}

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    #TODO remove using xmlNode
    Grid.localInputAndChecks(self,xmlNode, paramInput)
    factsettings = xmlNode.find("ResponseSurfaceDesignSettings")
    if factsettings == None:
      self.raiseAnError(IOError,'ResponseSurfaceDesignSettings xml node not found!')
    facttype = factsettings.find("algorithmType")
    if facttype == None:
      self.raiseAnError(IOError,'node "algorithmType" not found in ResponseSurfaceDesignSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedOptions.keys():
      self.raiseAnError(IOError,'"type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedOptions.keys()))
    self.respOpt['algorithmType'] = facttype.text.lower()
    # set defaults
    if self.respOpt['algorithmType'] == 'boxbehnken':
      self.respOpt['options'] = {'ncenters':None}
    else:
      self.respOpt['options'] = {'centers':(4,4),'alpha':'orthogonal','face':'circumscribed'}
    for child in factsettings:
      if child.tag not in 'algorithmType':
        self.respOpt['options'][child.tag] = child.text.lower()
    # start checking
    for key,value in self.respOpt['options'].items():
      if key not in self.acceptedOptions[facttype.text.lower()]:
        self.raiseAnError(IOError,'node '+key+' unknown. Available are "'+' '.join(self.acceptedOptions[facttype.text.lower()])+'"!!')
      if self.respOpt['algorithmType'] == 'boxbehnken':
        if key == 'ncenters':
          if self.respOpt['options'][key] != None:
            try:
              self.respOpt['options'][key] = int(value)
            except:
              self.raiseAnError(IOError,'"'+key+'" is not an integer!')
      else:
        if key == 'centers':
          if len(value.split(',')) != 2:
            self.raiseAnError(IOError,'"'+key+'" must be a comma separated string of 2 values only!')
          try:
            self.respOpt['options'][key] = (int(value.split(',')[0]),int(value.split(',')[1]))
          except:
            self.raiseAnError(IOError,'"'+key+'" values must be integers!!')
        if key == 'alpha':
          if value not in ['orthogonal','rotatable']:
            self.raiseAnError(IOError,'Not recognized options for node ' +'"'+key+'". Available are "orthogonal","rotatable"!')
        if key == 'face':
          if value not in ['circumscribed','faced','inscribed']:
            self.raiseAnError(IOError,'Not recognized options for node ' +'"'+key+'". Available are "circumscribed","faced","inscribed"!')
    gridInfo = self.gridEntity.returnParameter('gridInfo')
    if len(self.toBeSampled.keys()) != len(gridInfo.keys()):
      self.raiseAnError(IOError,'inconsistency between number of variables and grid specification')
    for varName, values in gridInfo.items():
      if values[1] != "custom":
        self.raiseAnError(IOError,"The grid construct needs to be custom for variable "+varName)
      if len(values[2]) != 2:
        self.raiseAnError(IOError,"The number of values can be accepted are only 2 (lower and upper bound) for variable "+varName)
    self.gridCoordinate = [None]*len(self.axisName)
    if len(self.gridCoordinate) < self.minNumbVars[self.respOpt['algorithmType']]:
      self.raiseAnError(IOError,'minimum number of variables for type "'+ self.respOpt['type'] +'" is '+str(self.minNumbVars[self.respOpt['type']])+'!!')
    self.externalgGridCoord = True

  def localGetInitParams(self):
    """
      Appends a given dictionary with class specific member variables and their
      associated initialized values.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Grid.localGetInitParams(self)
    for key,value in self.respOpt.items():
      if key != 'options':
        paramDict['Response Design '+key] = value
      else:
        for kk,val in value.items():
          paramDict['Response Design options '+kk] = val
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
    if   self.respOpt['algorithmType'] == 'boxbehnken':
      self.designMatrix = doe.bbdesign(len(self.gridInfo.keys()),center=self.respOpt['options']['ncenters'])
    elif self.respOpt['algorithmType'] == 'centralcomposite':
      self.designMatrix = doe.ccdesign(len(self.gridInfo.keys()), center=self.respOpt['options']['centers'], alpha=self.respOpt['options']['alpha'], face=self.respOpt['options']['face'])
    gridInfo   = self.gridEntity.returnParameter('gridInfo')
    stepLength = {}
    for cnt, varName in enumerate(self.axisName):
      self.mapping[varName] = np.unique(self.designMatrix[:,cnt]).tolist()
      gridInfo[varName] = (gridInfo[varName][0],gridInfo[varName][1],InterpolatedUnivariateSpline(np.array([min(self.mapping[varName]), max(self.mapping[varName])]),
                           np.array([min(gridInfo[varName][2]), max(gridInfo[varName][2])]), k=1)(self.mapping[varName]).tolist())
      stepLength[varName] = [round(gridInfo[varName][-1][k+1] - gridInfo[varName][-1][k],14) for k in range(len(gridInfo[varName][-1])-1)]
    self.gridEntity.updateParameter("stepLength", stepLength, False)
    self.gridEntity.updateParameter("gridInfo", gridInfo)
    Grid.localInitialize(self)
    self.limit = self.designMatrix.shape[0]

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
    gridcoordinate = self.designMatrix[self.counter - 1][:].tolist()
    for cnt, varName in enumerate(self.axisName):
      self.gridCoordinate[cnt] = self.mapping[varName].index(gridcoordinate[cnt])
    Grid.localGenerateInput(self,model, myInput)
