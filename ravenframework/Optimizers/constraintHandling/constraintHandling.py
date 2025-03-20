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
  Implementation of constraint handline for Genetic Algorithms optimizer

  Created June,16,2020
  @authors: Junyung Kim, Mohammad Abdo
"""
import xarray as xr
import numpy as np
from ..GeneticAlgorithm import datasetToDataArray

def constraintHandling(self, info, rlz, offSprings, objectiveVal, multiObjective=False):
    """
    This function handles the constraints for both single and multi-objective optimization.
    @ In, info, dict, dictionary containing information about the run
    @ In, rlz, dict, dictionary containing the results of the run
    @ In, multiObjective, bool, indicates if it's a multi-objective optimization
    @ Out, None
    """
    traj = info['traj']

    # Collect parameters for constraint functions (excluding default params)
    constraintData = {}
    if self._constraintFunctions or self._impConstraintFunctions:
        params = []
        for y in (self._constraintFunctions + self._impConstraintFunctions):
            params += y.parameterNames()
        excludeParams = set(self._objectiveVar)
        excludeParams.update(list(self.toBeSampled.keys()))
        for p in list(set(params) - excludeParams):
            constraintData[p] = list(np.atleast_1d(rlz[p].data))

    # Compute constraint function g_j(x) for all constraints and population individuals
    g0 = np.zeros((np.shape(offSprings)[0], len(self._constraintFunctions) + len(self._impConstraintFunctions)))

    g = xr.DataArray(g0,
                     dims=['chromosome', 'Constraint'],
                     coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                             'Constraint': [y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]})

    for index, individual in enumerate(offSprings):
        newOpt = individual

        #note that objectiveVal is 2d in multiObjective and 1d in single
        if multiObjective:
            optDict = dict(zip(self._objectiveVar, [item[index] for item in objectiveVal]))
        else:
            optDict = {self._objectiveVar[0]: objectiveVal[index]}
        opt = {k: self._objMult[k] * optDict[k] for k in self._objectiveVar}

        for p, v in constraintData.items():
            opt[p] = v[index]

        for constIndex, constraint in enumerate(self._constraintFunctions + self._impConstraintFunctions):
            if constraint in self._constraintFunctions:
                g.data[index, constIndex] = self._handleExplicitConstraints(newOpt, constraint)
            else:
                g.data[index, constIndex] = self._handleImplicitConstraints(newOpt, opt, constraint)

    return g
