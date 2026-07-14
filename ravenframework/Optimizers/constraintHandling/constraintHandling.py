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

def constraintHandling(self, info, rlz, offspring, minObjVals, multiObjective=False):
    """
    This function handles the constraints for both single and multi-objective optimization.
    @ In, info, dict, dictionary containing information about the run
    @ In, rlz, dict, dictionary containing the results of the run
    @ In, offspring, xr.DataArray, offspring individuals
    @ In, minObjVals, list, RAVEN minimization-space objective values. Objectives declared as max have already been multiplied by -1.
    @ In, multiObjective, bool, indicates if it's a multi-objective optimization
    @ Out, constraintVals, xr.DataArray, constraint evaluations for each chromosome and constraint
    """
    allConstraintFunctions = self._constraintFunctions + self._impConstraintFunctions
    # Collect parameters for constraint functions (excluding default params)
    constraintData = {}
    if allConstraintFunctions:
        params = []
        for y in allConstraintFunctions:
            params += y.parameterNames()
        excludeParams = set(self._objectiveVar)
        excludeParams.update(list(self.toBeSampled.keys()))
        for p in list(set(params) - excludeParams):
            constraintData[p] = list(np.atleast_1d(rlz[p].data))

    # Compute constraint function g_j(x) for all constraints and population individuals
    constraintDataArray = np.zeros((np.shape(offspring)[0], len(allConstraintFunctions)))

    constraintVals = xr.DataArray(constraintDataArray,
                     dims=['chromosome', 'Constraint'],
                     coords={'chromosome': np.arange(np.shape(offspring)[0]),
                             'Constraint': [y.name for y in allConstraintFunctions]})

    for index, individual in enumerate(offspring):
        newOpt = individual

        minObjValsByName = dict(zip(self._objectiveVar, [item[index] for item in minObjVals]))

        # Implicit constraints expect user-facing objective signs, so convert minObjVals back.
        externalObjVals = {k: self._objMult[k] * minObjValsByName[k] for k in self._objectiveVar}

        for p, v in constraintData.items():
            externalObjVals[p] = v[index]

        for constIndex, constraint in enumerate(allConstraintFunctions):
            if constraint in self._constraintFunctions:
                constraintVals.data[index, constIndex] = self._handleExplicitConstraints(newOpt, constraint)
            else:
                constraintVals.data[index, constIndex] = self._handleImplicitConstraints(newOpt, externalObjVals, constraint)

    return constraintVals
