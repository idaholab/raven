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

@author: alfoa

This python script/module mimics a code that  can retrieve a restart file containing a solution vector
(CSV in this case) and simply moltiply the content by a factor 'multiplication_factor' retrieved from
the input file.
The  'restart' file name can be inputted by RAVEN as a Sampled Variable
"""
import xml.etree.ElementTree as ET
import sys
import numpy as np
import pandas as pd

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
  restartFileName = initializationDict['restartFileName']
  mult  = int(float(initializationDict['multiplication_factor']))
  outputfile = initializationDict['outputFileName']
  df = pd.read_csv(restartFileName)
  df = df.mul(mult)
  variables = df.keys()
  with open(outputfile,"w") as fo:
    fo.write(",".join(variables)+'\n')
    for tts in range(len(df)):
      vv = []
      for var in variables:
        vv.append(str(df[var][tts]))
      fo.write(",".join(vv)+'\n')



