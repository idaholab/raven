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
Created on May 14 2024

@author: andrea

This python script/module mimics a code that  creates a solution vector (CSV in this case) that 
can be used by another code to restart a calculation. The solution vector is contained 
in a filename that is dumped as well.
The  'restart' file name can be retrieved by RAVEN and stored in a RAVEN variable.
"""
import xml.etree.ElementTree as ET
import sys
import os
import numpy as np

def readInput(filename):
  tree = ET.parse(filename)
  root = tree.getroot()
  input = {}
  for element in root:
    input[element.tag] = element.text
  return input

if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise Exception("No input file")
  inputFileName = sys.argv[1]
  initializationDict = readInput(inputFileName)
  rseed = int(float(initializationDict['randomSeed']))
  ts  = int(float(initializationDict['time_steps']))
  outputfile = initializationDict['outputFileName']
  variables = [var.strip() for var in initializationDict['randomVariableNames'].split(",")]
  # seed
  np.random.seed(rseed)
  values = {}
  for var in variables:
    values[var] = np.random.random(size=ts)
  with open(outputfile,"w") as fo:
    fo.write(",".join(variables+['restartFileName'])+'\n')
    for tts in range(ts):
      vv = []
      for var in variables:
        vv.append(str(values[var][tts]))
      vv.append(os.path.abspath(outputfile))
      fo.write(",".join(vv)+'\n')



