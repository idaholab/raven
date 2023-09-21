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
Created on March 25, 2022

@author: aalfonsi
"""

class csasData:
  """
    Class that parses output of scale output (csas sequences) and collect a RAVEN compatible dict
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, str, file name to be parsed
      @ Out, None
    """
    self._data = {}
    self.filename = filen
    self.loadData()

  def loadData(self):
    """
      Method to load data from csas sqeuences
      @ In, None
      @ Out, None
    """
    with open(self.filename,"r+") as fobj:
      for line in fobj.readlines():
        if line.strip().startswith("***") and "best estimate system k-eff" in line:
          self._data['keff'] = [float(line.split("best estimate system k-eff")[1].strip().split()[0])]
        if line.strip().startswith("***") and "Energy of average lethargy of Fission (eV)" in line:
          self._data['AverageLethargyFission'] = [float(line.split("Energy of average lethargy of Fission (eV)")[1].strip().split()[0])]
        if line.strip().startswith("***") and "system nu bar" in line:
          self._data['nubar'] = [float(line.split("system nu bar")[1].strip().split()[0])]
        if line.strip().startswith("***") and "system mean free path (cm)" in line:
          self._data['meanFreePath'] = [float(line.split("system mean free path (cm)")[1].strip().split()[0])]

  def returnData(self):
    """
      Return Data into dict format
      @ In, None
      @ Out, self._data, dict, dictionary with data
    """
    return self._data
