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
import re
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
import sys
import os
"""
Created on March 25, 2018

@author: alfoa
"""

class tritonData:
  """
    Class that parses output of relap5 output file and reads in trip, minor block and write a csv file
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, string, file name to be parsed
      @ Out, None
    """
    self.lines            = open(os.path.abspath(os.path.expanduser(filen)),"r").readlines()
    # retrieve keff and kinf
    self.transportInfo    = self.retrieveKeff()
    # retrieve nuclide densities
    self.nuclideDensities = self.retrieveNuclideConcentrations()
    # retrieve mixture powers
    self.mixPowers        = self.retrieveMixtureInfo()
    # check if something has been found
    if all(v is None for v in [self.transportInfo,self.nuclideDensities,self.mixPowers]):
      raise IOError("No readable outputs have been found!")
      
  def retrieveKeff(self):
    """
      Retrieve Summary Keff info
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
    """
    indicesKeff = np.asarray([i for i, x in enumerate(self.lines) if x.strip().startswith("k-eff =")])
    indicesKinf = np.asarray([i for i, x in enumerate(self.lines) if x.strip().startswith("Infinite neutron multiplication")])
    if len(indicesKeff) == 0 and len(indicesKinf) == 0:
      return None
    outputDict = {'time':[],'info_ids':[], 'values':None}
    values = defaultdict(list)
    if len(indicesKeff) > 0:
      outputDict['info_ids'] = ['keff','iter_number','keff_delta','max_flux_delta',"kinf","kinf_epsilon","kinf_p","kinf_f","kinf_eta"]
      for cnt, index in enumerate(indicesKeff):
        #keff
        components = self.lines[index].split()
        time = float(components[4][:-1])
        outputDict['time'].append(time)
        values[time].append(float(components[2]))
        comps = self.lines[index-1].split()
        values[time].extend([int(comps[0]),float(comps[2]),float(comps[3])])
        # kinf
        kinf_index = indicesKinf[cnt]
        values[time].append(float(self.lines[kinf_index].split()[-1]))
        values[time].append(float(self.lines[kinf_index-2].split()[-1]))
        values[time].append(float(self.lines[kinf_index-2].split()[-1]))
        values[time].append(float(self.lines[kinf_index-3].split()[-1]))
        values[time].append(float(self.lines[kinf_index-4].split()[-1]))
        values[time].append(float(self.lines[kinf_index-5].split()[-1]))
      outputDict['values'] = values  
    return outputDict
         
  def retrieveMixtureInfo(self):
    """
      Retrieve Summary of Triton Mixture Info
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
    """
    indicesMatPowers = np.asarray([i+1 for i, x in enumerate(self.lines) if x.strip().startswith("--- Material powers for depletion pass no")])
    if len(indicesMatPowers) == 0:
      return None
    outputDict = {'time':[],'info_ids':[], 'values':None}
    values = defaultdict(list)
    outputDict['info_ids'] = ['bu']
    for cnt, index in enumerate(indicesMatPowers):
      #keff
      components = self.lines[index].split(",")
      time = float(components[0].split()[2])
      bu   = float(components[1].split()[2])
      values[time].append(bu)
      startIndex = index + 4
      mixName = ""
      while mixName.lower() != "total":
        startIndex+=1
        components = self.lines[startIndex].split()
        mixName = components[0].strip()
        if mixName.lower() != "total":
          if "tot_power_mix_"+mixName.strip() not in outputDict['info_ids']:
            outputDict['info_ids'].extend( ["tot_power_mix_"+mixName.strip(),"fract_power_mix_"+mixName.strip(),"th_flux_mix_"+mixName.strip(),"tot_flux_mix_"+mixName.strip()]  )
          values[time].extend( [float(components[1]),float(components[2]),float(components[4]),float(components[5])] )
    outputDict['values'] = values
    return outputDict      

  def retrieveNuclideConcentrations(self):
    """
      Retrieve Summary of Triton Nuclide Concentration
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
    """
    outputDict = None
    try:
      indexConc = next(i for i,x in enumerate(self.lines) if x.strip().startswith("| nuclide concentrations"))
    except StopIteration:
      return outputDict
   
    # retrieve time grid
    uomTime = self.lines[indexConc+1].split("|")[-1].split()[-1].strip()
    timeGrid = [float(elm.replace(uomTime[0],"")) for elm in self.lines[indexConc+2].split("|")[-1].split()]
    values = defaultdict(list)
    outputDict = {'time':timeGrid,'nuclide_ids':[], 'values':None}
    startIndex = indexConc + 3
    nuclideName = ""
    while nuclideName.lower() != "total":
      startIndex+=1
      components = self.lines[startIndex].split()
      nuclideName = components[0].strip()
      outputDict['nuclide_ids'].append(nuclideName)
      for i, val in enumerate(components[2:]):
        values[timeGrid[i]].append(val)
    
    outputDict['values'] = values
    return outputDict
  
  def pringCSV(self, fileout):
    """
      Print Data into CSV format
      @ In, fileout, str, the output file name
      @ Out, None
    """


if __name__ == '__main__':

  test = tritonData("~/Downloads/5_9pc_1200_numpar30.out")
  aa = test.retrieveNuclideConcentrations()
  bb = test.retrieveKeff()