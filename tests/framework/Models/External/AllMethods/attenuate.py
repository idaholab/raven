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
#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Simulates the attenuation of a beam through a purely-scattering medium with N distinct materials and unit length.
#     The uncertain inputs are the opacities.
#
import numpy as np

def evaluate(inp):
  if len(inp)>0:
    return np.exp(-sum(inp)/len(inp))
  else:
    return 1.0

##### RAVEN methods #####
def _readMoreXML(self,xmlNode):
  main = xmlNode.find('moreXMLInfo')
  node = main.find('valueForXML')
  self.fromReadMoreXML = float(node.text)

def initialize(self,runInfo,inputs):
  # check the existence of some entries
  frameworkDir = runInfo['FrameworkDir']
  workingDir = runInfo['WorkingDir']
  # check readMoreXML value is available
  self.fromInit = np.sqrt(self.fromReadMoreXML)

def createNewInput(self,inputs,samplerType,**kwargs):
  # check sampler type
  if samplerType != 'Grid':
    raise IOError('Received wrong sampler type in external model createNewInput!  Expected "Grid" but got '+samplerType)
  # set a variable through "self"
  self.fromCNISelf = self.fromReadMoreXML / 2.0
  toReturn = dict(kwargs['SampledVars'])
  toReturn['fromCNIDict'] = self.fromInit * 2.0
  toReturn['unwanted'] = 42
  return toReturn

def run(self,Input):
  # check everyone else is available here
  self.fromReadMoreXML
  self.fromInit
  self.fromCNISelf
  Input['fromCNIDict']
  # changed input
  self.y1 += 0.05
  self.ans  = evaluate([self.y1,self.y2])

#
#  This model has analytic mean and variance documented in raven/docs/tests
#
