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


def initialize(self, runInfo, Input):
  self.a1 = 10.0
  self.a2 = 10.0
  self.b1 = 0.5
  self.b2 = 0.5

def run(self,Input):
  a1, a2, b1, b2 = self.a1, self.a2, self.b1, self.b2
  x1, x2 = self.x1, self.x2
  self.c = a1*(x1-b1)**2 + a2*(x2-b2)**2 #+ noise

