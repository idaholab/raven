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
import copy as cp
def run(self,Input):
  """
    Method require by RAVEN to run this as an external model
    @ In, self, object, object to store members on
    @ In Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  self.u1 = 1.11*cp.deepcopy(Input["u0"])
  self.v1 = 0.0222*cp.deepcopy(Input['v0'])
  self.w1 = 33.3*cp.deepcopy(Input["w0"])
  self.x1 = 444.*cp.deepcopy(Input['x0'])
  self.y1 = 0.555*cp.deepcopy(Input["y0"])
  self.z1 = 66.6*cp.deepcopy(Input['z0'])
