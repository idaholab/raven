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
'''
Created on 01/04/19

@author: wangc
'''
import numpy as np

def run(self,Input):
  """
  """
  # define the raw dataset
  alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  # create mapping of characters to integers (0-25) and the reverse
  charToInt = dict((c, i) for i, c in enumerate(alphabet))
  intToChar = dict((i, c) for i, c in enumerate(alphabet))
  maxTime = 0.9
  tStep = maxTime/float(self.seqLength)

  self.index = int(Input['index']) # 0 ~ len(alphabet) - self.seqLength
  self.seqLength = int(Input['seqLength'])
  # prepare the dataset of input to output pairs encoded as integers
  self.x = np.zeros(self.seqLength)
  self.y = np.zeros(self.seqLength)
  self.time = np.zeros(self.seqLength)
  for t in range(self.seqLength):
    self.time[t] = tStep * t
    self.x[t] = charToInt[alphabet[self.index + t]]
    self.y[t] = charToInt[alphabet[self.index + self.seqLength]]

