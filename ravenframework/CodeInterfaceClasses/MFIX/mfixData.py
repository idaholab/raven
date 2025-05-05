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
"""
Created on April 09, 2025
@author: kimj5

comments: Interface for MFIX simulation 
"""

from mfix_raven_interface import MFIX
import pickle
import matplotlib.pyplot as plt

class mfixData:




workingDir = '/Users/wangc/projects/mfix/model_outputs'
command = ''
output = ''

mfix = MFIX()

output = mfix.finalizeCodeOutput(command, output, workingDir)

dataSet = mfix._dataSet

# dataSet.plot.scatter(x='time', y='height', z='void_frac', hue='void_frac')
# plt.show()

with open('dataset.pkl', 'wb') as f:
  pickle.dump(dataSet, f, protocol=-1)
