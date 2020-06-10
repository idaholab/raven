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

def run(self, Input):
  """
    Function to calculate the average of the sampled six variables. This is used to check distribution for large number of samples.
    @ In, Input, ParameterInput, RAVEN sampled params.
    @ Out, None
  """
  self.y = (self.x3 + self.x4 + self.x5 + self.x6 + self.x7 + self.x8) / 6.
