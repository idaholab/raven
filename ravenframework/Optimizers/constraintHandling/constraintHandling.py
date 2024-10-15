import xarray as xr
import numpy as np
from ..GeneticAlgorithm import datasetToDataArray

def constraintHandling(self, info, rlz, multiObjective=False):
    """
    This function handles the constraints for both single and multi-objective optimization.
    @ In, info, dict, dictionary containing information about the run
    @ In, rlz, dict, dictionary containing the results of the run
    @ In, multiObjective, bool, indicates if it's a multi-objective optimization
    @ Out, None
    """
    traj = info['traj']
    for t in self._activeTraj[1:]:
        self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory', 0)
    self.incrementIteration(traj)

    offSprings = datasetToDataArray(rlz, list(self.toBeSampled))

    # Handle objective values differently for single and multi-objective cases
    if multiObjective:
        objectiveVal = []
        for i in range(len(self._objectiveVar)):
            objectiveVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))
    else:
        objectiveVal = list(np.atleast_1d(rlz[self._objectiveVar[0]].data))

    # Collect parameters for constraint functions (excluding default params)
    constraintData = {}
    if self._constraintFunctions or self._impConstraintFunctions:
        params = []
        for y in (self._constraintFunctions + self._impConstraintFunctions):
            params += y.parameterNames()
        excludeParams = set(self._objectiveVar) if multiObjective else {self._objectiveVar[0]}
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

        # Handle optimization values differently for single and multi-objective
        if multiObjective:
            objOpt = dict(zip(self._objectiveVar, list(map(lambda x: -1 if x == "max" else 1, self._minMax))))
            opt = dict(zip(self._objectiveVar, [item[index] for item in objectiveVal]))
            opt = {k: objOpt[k] * opt[k] for k in opt}
        else:
            opt = {self._objectiveVar[0]: objectiveVal[index]}

        for p, v in constraintData.items():
            opt[p] = v[index]

        for constIndex, constraint in enumerate(self._constraintFunctions + self._impConstraintFunctions):
            if constraint in self._constraintFunctions:
                g.data[index, constIndex] = self._handleExplicitConstraints(newOpt, constraint)
            else:
                g.data[index, constIndex] = self._handleImplicitConstraints(newOpt, opt, constraint)

    # Compute fitness for the offspring
    offSpringFitness = self._fitnessInstance(rlz,
                                             objVar=self._objectiveVar,
                                             a=self._objCoeff,
                                             b=self._penaltyCoeff,
                                             penalty=None,
                                             constraintFunction=g,
                                             constraintNum=self._numOfConst,
                                             type=self._minMax)

    # Single-objective post-processing (if needed)
    if not multiObjective:
        self._collectOptPoint(rlz, offSpringFitness, objectiveVal, g)
        self._resolveNewGeneration(traj, rlz, objectiveVal, offSpringFitness, g, info)

    return traj, g, objectiveVal, offSprings, offSpringFitness