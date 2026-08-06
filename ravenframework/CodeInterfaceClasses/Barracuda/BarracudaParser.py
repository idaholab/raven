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
Created on Mar 10, 2015

@author: talbpaul
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import os
import sys
import numpy as np
from ravenframework.utils import mathUtils
# numpy with version 1.14.0 and upper will change the floating point type and print
# https://docs.scipy.org/doc/numpy-1.14.0/release.html
if int(np.__version__.split('.')[1]) > 13:
  np.set_printoptions(**{'legacy':'1.13'})

def _reprIfFloat(value):
  """
    Uses repr if the value is a float
    @ In, value, any, the value to convert to a string
    @ Out, _reprIfFloat, string, a string conversion of this
  """
  if mathUtils.isAFloat(value):
    return repr(value)
  else:
    return str(value)

class BarracudaParser():
  """
    import the user-edited input file, build list of strings with replaceable parts
  """
  def __init__(self,inputFiles,**Kwargs):
    """
      Accept the input file and parse it by the prefix-postfix breaks. Someday might be able to change prefix,postfix,defaultDelim from input file, but not yet.
      @ In, inputFiles, list, string list of input filenames that might need parsing.
      @ In, perturbVarDict, dictionary of perturbed variables
      @ Out, None
    """
    self.inputFiles = inputFiles
    self.lines = self.inputFile.readlines()
    self.inputFiles.close()
    self.distributedPerturbedVars = Kwargs['SampledVars']

    # Iterate through self.distributedPerturbedVars and call updates to self.inputFiles
    for perturbedParam in self.distributedPerturbedVars.keys():
          if perturbedParam == 'temperature':
            self.updateTemperature(self.distributedPerturbedVars['temperature'])

    self.writeNewInput()


  def updateTemperature(self, newTemperature):
    """
      Edits self.lines to modify the temperature.
      @ In, newTemperature, float, value of the perturbed temperature.
      @ Out, None
    """
    # Perform math on perturbed variable

    # Iterate through file to find specific text
    for lineNo, line in enumerate(self.lines):
      if "isothermalT" in line:
        lineUpdateNo = lineNo
        break
    
    # Updaate lines with perturbed variable
    self.lines[lineUpdateNo] = "isothermalT     %8.6e\n" % (newTemperature)

    return


  def writeNewInput(self):
    """
      Generates a new input file with the existing parsed dictionary.
      @ In, None
      @ Out, None
    """
    outfile = self.inputFiles
    outfile.open('w')
    outfile.writelines(self.lines)
    outfile.close()
