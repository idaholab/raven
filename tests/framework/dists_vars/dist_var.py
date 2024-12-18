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

def run(self,Input):

  self.t1 = Input['t1']
  self.t2 = Input['t2']
  self.t3 = Input['t3']
  self.t4 = Input['t4']
  self.x0 = Input['x0']
  self.y0 = Input['y0']
  self.z0 = Input['z0']
  self.x1 = Input['x1']
  self.y1 = Input['y1']
  self.z1 = Input['z1']

  if max(self.t1,self.t2) >= (30.0*24.0):
    self.out = 0.0 # OK
  else:
    temp = (30.0*24.0) - max(self.t1,self.t2) # Fail
    self.out = float(temp/(30.0*24.0))
