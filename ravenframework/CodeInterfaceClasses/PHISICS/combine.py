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
Created on March 8th 2018
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import numpy as np

class combine():
  """
    Combines the PHISICS and RELAP  output into one.
  """
  def __init__(self,workingDir,relapData, phisicsData,depTimeDict,inpTimeDict):
    """
      Constructor.
      @ In, workingDir, string, absolute path to working directory
      @ In, relapData, dict, data from relap
      @ In, phisicsData, string, data from phisics
      @ In, depTimeDict, dictionary, information from the xml depletion file
      @ In, inpTimeDict, dictionary, information from the xml input file
      @ Out, None
    """
    self.timeStepSelected = []
    paramDict = {}
    paramDict['phisicsData'] =  phisicsData
    paramDict['relapData'] = relapData
    paramDict['depTimeDict'] = depTimeDict
    paramDict['inpTimeDict'] = inpTimeDict
    selectedTs = 0
    for i in paramDict['depTimeDict']['timeSteps'].split(' '):
      selectedTs = selectedTs + int(i)
      self.timeStepSelected.append(selectedTs)
    self.response = self.joinData(paramDict)

  def joinData(self, paramDict):
    """
      Joins the RELAP and PHISICS data based on the time lines selected from PHISICS.
      @ In, paramDict, dictionary, dictionary of parameters
      @ Out, response, dict, the output to be returned
    """
    phisicsVars = list(paramDict['phisicsData'].keys())
    relapVars = list(paramDict['relapData'].keys())
    headers =  phisicsVars + relapVars
    data = []
    data.append([0.0] * len(phisicsVars) + np.array(list(paramDict['relapData'].values())).T[0].tolist())

    thBurnStep = [float(val) for val in paramDict['inpTimeDict']['TH_between_BURN'].split(' ')]
    # check the thburn
    maxTime = max(thBurnStep)
    # check max Relap
    maxRelap = max(paramDict['relapData']['time'])
    if maxRelap < maxTime:
      thBurnStep[-1] = maxRelap

    lineNumber, THbetweenBurn, mrTau = 0, 0, 0

    while THbetweenBurn < len(thBurnStep):
      lineNumber = lineNumber + 1
      addedNow = False
      # if the time on a relap line is <= than the TH_between_burn selected
      if paramDict['relapData']['time'][lineNumber] <= thBurnStep[THbetweenBurn]:
        # print the relap line with the phisics line corresponding to last time step of a burnstep
        valuesPhisics = np.array(list(paramDict['phisicsData'].values())).T[self.timeStepSelected[mrTau]-1].tolist()
        valuesRelap = np.array(list(paramDict['relapData'].values())).T[lineNumber].tolist()
        data.append(valuesPhisics+valuesRelap)
        addedNow = True

      # if the relap time on a line is larger the TH_between_burn selected
      if paramDict['relapData']['time'][lineNumber] >= thBurnStep[THbetweenBurn]:
        # change the TH_between_burn selected
        THbetweenBurn = THbetweenBurn + 1
        # change the burn step in phisics
        mrTau = mrTau + 1
        # if this is the last TH_between_burn
        if THbetweenBurn == len(thBurnStep) and not addedNow:
          # print the last line of phisics and relap.
          valuesPhisics = np.array(list(paramDict['phisicsData'].values())).T[-1].tolist()
          valuesRelap = np.array(list(paramDict['relapData'].values())).T[-1].tolist()
          data.append(valuesPhisics+valuesRelap)
    data = np.asarray(data)
    response = {var: data[: , i] for i, var in enumerate(headers)}
    return response

  def returnData(self):
    """
      Method to return the data in a dictionary
      @ In, None
      @ Out, self.response, dict, the dictionary containing the data {var1:array,var2:array,etc}
    """
    return self.response
