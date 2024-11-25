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

def run(raven,inputDict):
  """
    Simple mechanical operations to test inputting and outputting ND values.
  """
  raven.scalarOut = raven.scalarIn**2
  raven.vectorOut = []
  print('DEBUGG extmod input:')
  for k, v in inputDict.items():
    print(k)
  for _, value in enumerate(raven.y):
    raven.vectorOut.append(raven.ND_in[value,:].sum())
