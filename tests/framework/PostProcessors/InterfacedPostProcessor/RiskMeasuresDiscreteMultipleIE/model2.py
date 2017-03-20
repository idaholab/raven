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
  """
    Method that implement a simple system with three components in a parallel configuration
    @ In, Input, dict, dictionary containing the data
    @ Out, outcome, float, logical status of the system given status of the components
  """
  Bstatus   = Input['Bstatus'][0]
  Cstatus   = Input['Cstatus'][0]
  Dstatus   = Input['Dstatus'][0]

  if (Bstatus == 1.0 and Cstatus == 1.0 and Dstatus==1.0):
    self.outcome = 1.0
  else:
    self.outcome = 0.0

  return self.outcome
