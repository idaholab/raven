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
import math

def run(self,Input):
  self.x1 = Input['x1']
  self.x2 = Input['x2']
  self.y4 = (self.x1**2.0) + (self.x2**2.0)

  if self.y4 <= 1.0:
    self.failure = 0.0
  else:
    self.failure = 1.0
