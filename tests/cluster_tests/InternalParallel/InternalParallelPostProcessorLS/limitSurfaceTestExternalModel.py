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

import numpy
import copy
#import pylab as pyl
#import random
#import mpl_toolkits.mplot3d.axes3d as p3

def initialize(self,runInfoDict,inputFiles):
  print('There is snow in my memories ...there is always snow...and my brian becomes white if I do not stop remembering...')
  self.z               = 0
  return

#def createNewInput(self,myInput,samplerType,**Kwargs):
#  return Kwargs['SampledVars']

def run(self,Input):
  #self.z = Input['x0']+Input['y0']
  self.z = self.x0 + self.y0
