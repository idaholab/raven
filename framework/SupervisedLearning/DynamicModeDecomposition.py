
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
  Created on May 8, 2018

  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Base subclass definition for DynamicModeDecomposition ROM (transferred from alfoa in SupervisedLearning)
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils
from SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------


class DynamicModeDecomposition(supervisedLearning):
  """
    This surrogate is aimed to construct a "time-dep" surrogate based on
    Dynamic Mode Decomposition method.
    Ref. Kutz, Brunton, Brunton, Proctor. Dynamic Mode Decomposition:
        Data-Driven Modeling of Complex Systems. SIAM Other Titles in
        Applied Mathematics, 2016
  """
  def __init__(self,messageHandler,**kwargs):
    # print("__init__")
    """
      DMD constructor
      @ In, messageHandler, MessageHandler.MessageUser, a MessageHandler object in charge of raising errors,
                           and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
    """
    supervisedLearning.__init__(self,messageHandler,**kwargs)
    self.availDmdAlgorithms          = ['dmd','hodmd']                      # available dmd types: basic dmd and high order dmd
    self.dmdParams                   = {}                                   # dmd settings container
    self.printTag                    = 'DMD'                                # print tag
    self.pivotParameterID            = kwargs.get("pivotParameter","time")  # pivot parameter
    self._dynamicHandling            = True                                 # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self.dmdParams['rankSVD'       ] = kwargs.get('rankSVD',None)           # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    self.dmdParams['energyRankSVD' ] = kwargs.get('energyRankSVD',None)     #  0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by "energyRankSVD"
    self.dmdParams['rankTLSQ'      ] = kwargs.get('rankTLSQ',None)          # truncation rank for total least square
    self.dmdParams['exactModes'    ] = kwargs.get('exactModes',True)        # True if the exact modes need to be computed (eigs and eigvs), otherwise the projected ones (using the left-singular matrix)
    self.dmdParams['optimized'     ] = kwargs.get('optimized',False)        # amplitudes computed minimizing the error between the mods and all the timesteps (True) or 1st timestep only (False)
    self.dmdParams['dmdType'       ] = kwargs.get('dmdType','dmd')          # the dmd type to be applied. Currently we support dmd and hdmd (high order dmd)
    # variables filled up in the training stages
    self._amplitudes                 = {}                                   # {'target1': vector of amplitudes,'target2':vector of amplitudes, etc.}
    self._eigs                       = {}                                   # {'target1': vector of eigenvalues,'target2':vector of eigenvalues, etc.}
    self._modes                      = {}                                   # {'target1': matrix of dynamic modes,'target2':matrix of dynamic modes, etc.}
    self.__Atilde                    = {}                                   # {'target1': matrix of lowrank operator from the SVD,'target2':matrix of lowrank operator from the SVD, etc.}
    self.pivotValues                 = None                                 # pivot values (e.g. time)
    self.KDTreeFinder                = None                                 # kdtree weighting model
    self.timeScales                  = {}                                   # time-scales (training and dmd). {'training' and 'dmd':{t0:float,'dt':float,'intervals':int}}

    # some checks
    if self.dmdParams['rankSVD'] is not None and self.dmdParams['energyRankSVD'] is not None:
      self.raiseAWarning('Both "rankSVD" and "energyRankSVD" have been inputted. "energyRankSVD" is predominant and will be used!')
    if self.dmdParams['dmdType'] not in self.availDmdAlgorithms:
      self.raiseAnError(IOError,'dmdType(s) available are "'+', '.join(self.availDmdAlgorithms)+'"!')
    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")
    # print(self.__dict__)

  def __setstate__(self,state):
    # print("__setstate__")
    """
      Initializes the DMD with the data contained in state
      @ In, state, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(state)
    self.KDTreeFinder = spatial.KDTree(self.featureVals)

  def _localNormalizeData(self,values,names,feat):
    # print("_localNormalizeData")
    """
      Overwrites default normalization procedure.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)
    # print(self.muAndSigmaFeatures)

  #######
  def __getTimeScale(self,dmd=True):
    # print("__getTimeScale")
    """
      Get the ts of the dmd (if dmd = True) or training (if dmd = False) reconstructed time scale.
      @ In, dmd, bool, optional, True if dmd time scale needs to be returned, othewise training one
      @ Out, timeScale, numpy.array, the dmd or training reconstructed time scale
    """
    timeScaleInfo = self.timeScales['dmd'] if dmd else self.timeScales['training']
    timeScale = np.arange(timeScaleInfo['t0'], timeScaleInfo['intervals'] + timeScaleInfo['dt'], timeScaleInfo['dt'])
    return timeScale

  def __getTimeEvolution(self, target):
    # print("__getTimeEvolution")
    """
      Get the time evolution of each mode
      @ In, target, str, the target for which mode evolution needs to be retrieved for
      @ Out, timeEvol, numpy.ndarray, the matrix that contains all the time evolution (by row)
    """
    omega = np.log(self._eigs[target]) / self.timeScales['training']['dt']
    # print(omega)

    van = np.exp(np.multiply(*np.meshgrid(omega, self.__getTimeScale())))
    # print(*np.meshgrid(omega, self.__getTimeScale()))
    # print(np.multiply(*np.meshgrid(omega, self.__getTimeScale())))
    # print(np.exp(np.multiply(*np.meshgrid(omega, self.__getTimeScale()))))
    # print('\n')
    timeEvol = (van * self._amplitudes[target]).T
    # print(timeEvol)

    return timeEvol

  def _reconstructData(self, target):
    # print("_reconstructData")
    """
      Retrieve the reconstructed data
      @ In, target, str, the target for which the data needs to be reconstructed
      @ Out, data, numpy.ndarray, the matrix (nsamples,n_time_steps) containing the reconstructed data
    """
    data = self._modes[target].dot(self.__getTimeEvolution(target))
    return data

  def __trainLocal__(self,featureVals,targetVals):
    # print("__trainLocal__")
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, numpy.ndarray, shape = [n_timeStep, n_dimensions], an array of time series data
    """
    # print(featureVals) # 9*2 array, the grid of enrichment and burnup [[5.9 26.402] [5.9 18.858] ... [9.9 16.501]]
    # print(targetVals) # 9*8*3 array, each layer is the [t, decay_heat, decay_heat_pu] corresponding to the featureVals

    # targetVals, The first layer are:
    # [    [0.00000000e+00   2.46120300e+03   8.20401000e-03]
    #      [1.00000000e-02   7.86907000e+02   2.62302300e-03]
    #      [2.00000000e-02   6.53188000e+02   2.17729300e-03]
    #      [5.00000000e-02   4.87992000e+02   1.62664000e-03]
    #      [7.50000000e-02   4.28547000e+02   1.42849000e-03]
    #      [1.00000000e-01   3.92058000e+02   1.30686000e-03]
    #      [2.00000000e-01   3.20266000e+02   1.06755300e-03]
    #      [3.00000000e-01   2.86340000e+02   9.54467000e-04] ]
    self.featureVals  = featureVals
    self.KDTreeFinder = spatial.KDTree(featureVals) # nearest neighbour method
    pivotParamIndex   = self.target.index(self.pivotParameterID)
    # print(self.pivotParameterID) # 't'
    # print(pivotParamIndex) # 0
    # print(self.target) # ['t', 'decay_heat', 'decay_heat_pu']
    self.pivotValues  = targetVals[0,:,pivotParamIndex]
    ts                = len(self.pivotValues)
    # print(self.pivotValues) # [ 0.     0.01   0.02   0.05   0.075  0.1    0.2    0.3  ]
    # print(ts) # 8
    for target in list(set(self.target) - set([self.pivotParameterID])): # target == decay_heat or decay_heat_pu
      # print(list(set(self.target) - set([self.pivotParameterID]))) # ['decay_heat', 'decay_heat_pu']
      targetParamIndex  = self.target.index(target)
      # print(target, targetParamIndex) # decay_heat 1,  or decay_heat_pu 2

      snaps = targetVals[:,:,targetParamIndex]
      # print(snaps) # 9*8 array, 9 combinations of [enrichment, burnup] and 8 time steps
      # if number of features (i.e. samples) > number of snapshots, we apply the high order DMD or HODMD has been requested
      imposedHODMD = False
      if self.dmdParams['dmdType'] == 'hodmd' or snaps.shape[0] < snaps.shape[1]:
        v = max(snaps.shape[1] - snaps.shape[0],2)
        imposedHODMD = True
        snaps = np.concatenate([snaps[:, i:snaps.shape[1] - v  + i + 1] for i in range(v) ], axis=0)
      # overlap snaps
      X, Y = snaps[:, :-1], snaps[:, 1:] # X: 9*7 array, t=k; Y: 9*7 array, t=k+1
      # print(X)
      if self.dmdParams['rankTLSQ'] is not None:
        X, Y = mathUtils.computeTruncatedTotalLeastSquare(X, Y, self.dmdParams['rankTLSQ'])
        # print('aaa')
      # print(X)
      rank = self.dmdParams['energyRankSVD'] if self.dmdParams['energyRankSVD'] is not None else (self.dmdParams['rankSVD'] if self.dmdParams['rankSVD'] is not None else -1)
      # print(rank) # rank == 0
      U, s, V = mathUtils.computeTruncatedSingularValueDecomposition(X, rank)
      # print(U, s, V) # U: 9*3 array; s: 1*3 array; V: 7*3 array.
      # lowrank operator from the SVD of matrices X and Y
      self.__Atilde[target] = U.T.conj().dot(Y).dot(V) * np.reciprocal(s)
      print(self.__Atilde[target]) # 3*3 array
      self._eigs[target], self._modes[target] = mathUtils.computeEigenvaluesAndVectorsFromLowRankOperator(self.__Atilde[target],
                                                                                                          Y, U, s, V,
                                                                                                          self.dmdParams['exactModes'])
      if imposedHODMD:
        self._modes[target] = self._modes[target][:targetVals[:,:,targetParamIndex].shape[0],:]
      self._amplitudes[target] = mathUtils.computeAmplitudeCoefficients(self._modes[target],
                                                                        targetVals[:,:,targetParamIndex],
                                                                        self._eigs[target],
                                                                        self.dmdParams['optimized'])
      # print(self._modes[target])
    # Default timesteps (even if the time history is not equally spaced in time, we "trick" the dmd to think it).
    self.timeScales = dict.fromkeys( ['training','dmd'],{'t0': 0, 'intervals': ts - 1, 'dt': 1})
    # print(self.timeScales) # {'training': {'t0': 0, 'intervals': 7, 'dt': 1}, 'dmd': {'t0': 0, 'intervals': 7, 'dt': 1}}

  def __evaluateLocal__(self,featureVals):
    # print("__evaluateLocal__")
    """
      This method is used to inquire the DMD to evaluate (after normalization that in
      this case is not performed)  a set of points contained in featureVals.
      a KDTree algorithm is used to construct a weighting function for the reconstructed space
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    returnEvaluation = {self.pivotParameterID:self.pivotValues}
    # print(returnEvaluation)
    for target in list(set(self.target) - set([self.pivotParameterID])):
      reconstructData = self._reconstructData(target).real
      # print(reconstructData)
      # find the nearest data and compute weights
      if len(reconstructData) > 1:
        weights, indexes = self.KDTreeFinder.query(featureVals, k=min(2**len(self.features),len(reconstructData)))
        # if 0 (perfect match), assign minimum possible distance
        weights[weights == 0] = sys.float_info.min
        weights =1./weights
        # normalize to 1
        weights = weights/weights.sum()
        for point in range(len(weights)):
          returnEvaluation[target] =  np.sum ((weights[point,:]*reconstructData[indexes[point,:]].T) , axis=1)
      else:
        returnEvaluation[target] = reconstructData[0]

    return returnEvaluation

  def writeXMLPreamble(self, writeTo, targets = None):
    # print("writeXMLPreamble")
    """
      Specific local method for printing anything desired to xml file at the begin of the print.
      @ In, writeTo, xmlUtils.StaticXmlElement instance, element to write to
      @ In, targets, list, list of targets for whom information should be written.
      @ Out, None
    """
    # add description
    supervisedLearning.writeXMLPreamble(self, writeTo, targets)
    description  = ' This XML file contains the main information of the DMD ROM.'
    description += ' If "modes" (dynamic modes), "eigs" (eigenvalues), "amplitudes" (mode amplitudes)'
    description += ' and "dmdTimeScale" (internal dmd time scale) are dumped, the method'
    description += ' is explained in P.J. Schmid, Dynamic mode decomposition'
    description += ' of numerical and experimental data, Journal of Fluid Mechanics 656.1 (2010), 5-28'
    writeTo.addScalar('ROM',"description",description)

  def writeXML(self, writeTo, targets = None, skip = None):
    # print("writeXML")
    """
      Adds requested entries to XML node.
      @ In, writeTo, xmlTuils.StaticXmlElement, element to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    if skip is None:
      skip = []

    # check what
    what = ['exactModes','optimized','dmdType','features','timeScale','eigs','amplitudes','modes','dmdTimeScale']
    if self.dmdParams['rankTLSQ'] is not None:
      what.append('rankTLSQ')
    what.append('energyRankSVD' if self.dmdParams['energyRankSVD'] is not None else 'rankSVD')
    if targets is None:
      readWhat = what
    else:
      readWhat = targets
    for s in skip:
      if s in readWhat:
        readWhat.remove(s)
    if not set(readWhat) <= set(what):
      self.raiseAnError(IOError, "The following variables specified in <what> node are not recognized: "+ ",".join(np.setdiff1d(readWhat, what).tolist()) )
    else:
      what = readWhat

    target = self.target[-1]
    toAdd = ['exactModes','optimized','dmdType']
    if self.dmdParams['rankTLSQ'] is not None:
      toAdd.append('rankTLSQ')
    toAdd.append('energyRankSVD' if self.dmdParams['energyRankSVD'] is not None else 'rankSVD')
    self.dmdParams['rankSVD'] = self.dmdParams['rankSVD'] if self.dmdParams['rankSVD'] is not None else -1

    for add in toAdd:
      if add in what :
        writeTo.addScalar(target,add,self.dmdParams[add])
    targNode = writeTo._findTarget(writeTo.getRoot(), target)
    if "features" in what:
      writeTo.addScalar(target,"features",' '.join(self.features))
    if "timeScale" in what:
      writeTo.addScalar(target,"timeScale",' '.join(['%.6e' % elm for elm in self.pivotValues.ravel()]))
    if "dmdTimeScale" in what:
      writeTo.addScalar(target,"dmdTimeScale",' '.join(['%.6e' % elm for elm in self.__getTimeScale()]))
    if "eigs" in what:
      eigsReal = " ".join(['%.6e' % self._eigs[target][indx].real for indx in
                       range(len(self._eigs[target]))])
      writeTo.addScalar("eigs","real", eigsReal, root=targNode)
      eigsImag = " ".join(['%.6e' % self._eigs[target][indx].imag for indx in
                               range(len(self._eigs[target]))])
      writeTo.addScalar("eigs","imaginary", eigsImag, root=targNode)
    if "amplitudes" in what:
      ampsReal = " ".join(['%.6e' % self._amplitudes[target][indx].real for indx in
                       range(len(self._amplitudes[target]))])
      writeTo.addScalar("amplitudes","real", ampsReal, root=targNode)
      ampsImag = " ".join(['%.6e' % self._amplitudes[target][indx].imag for indx in
                               range(len(self._amplitudes[target]))])
      writeTo.addScalar("amplitudes","imaginary", ampsImag, root=targNode)
    if "modes" in what:
      for smp in range(len(self._modes[target])):
        valDict = {'real': ' '.join([ '%.6e' % elm for elm in self._modes[target][smp,:].real]),
                   'imaginary':' '.join([ '%.6e' % elm for elm in self._modes[target][smp,:].imag])}
        attributeDict = {self.features[index]:'%.6e' % self.featureVals[smp,index] for index in range(len(self.features))}
        writeTo.addVector("modes","realization",valDict, root=targNode, attrs=attributeDict)

  def __confidenceLocal__(self,featureVals):
    # print("__confidenceLocal__")
    """
      The confidence associate with a set of requested evaluations
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, None
    """
    pass

  def __resetLocal__(self,featureVals):
    # print("__resetLocal__")
    """
      After this method the ROM should be described only by the initial parameter settings
      @ In, featureVals, numpy.ndarray, shape= (n_samples, n_dimensions), an array of input data (training data)
      @ Out, None
    """
    self.amITrained   = False
    self._amplitudes  = {}
    self._eigs        = {}
    self._modes       = {}
    self.__Atilde     = {}
    self.pivotValues  = None
    self.KDTreeFinder = None
    self.featureVals  = None

  def __returnInitialParametersLocal__(self):
    # print("__returnInitialParametersLocal__")
    """
      This method returns the initial parameters of the SM
      @ In, None
      @ Out, self.dmdParams, dict, the dict of the SM settings
    """
    return self.dmdParams

  def __returnCurrentSettingLocal__(self):
    # print("__returnCurrentSettingLocal__")
    """
      This method is used to pass the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, self.dmdParams, dict, the dict of the SM settings
    """
    return self.dmdParams


class DynamicModeDecompositionControl(supervisedLearning):
  """
    This surrogate is aimed to construct a "time-dep" surrogate based on
    Dynamic Mode Decomposition with conmethod.
    Ref. Proctor, Joshua L., Steven L. Brunton, and J. Nathan Kutz.
        "Dynamic mode decomposition with control."
        SIAM Journal on Applied Dynamical Systems 15, no. 1 (2016): 142-161.

    Updated on July 1st, 2020

    @author: Haoyu Wang, Argonne National Laboratory

  """
  def __init__(self,messageHandler,**kwargs):
    # print("__init__")
    """
      DMD constructor
      @ In, messageHandler, MessageHandler.MessageUser, a MessageHandler object in charge of raising errors,
                           and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
    """
    supervisedLearning.__init__(self,messageHandler,**kwargs)
    self.availDmdAlgorithms          = ['dmd','hodmd']                      # available dmd types: basic dmd and high order dmd
    self.dmdParams                   = {}                                   # dmd settings container
    self.printTag                    = 'DMDC'                               # print tag
    self.pivotParameterID = kwargs.get("pivotParameter", "time")  # pivot parameter
    self.pivotParameterID = self.pivotParameterID.split(',')  # pivot parameter

    self._dynamicHandling            = True                                 # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self.dmdParams['rankSVD'       ] = kwargs.get('rankSVD',-1)           # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    self.dmdParams['dmdType'       ] = kwargs.get('dmdType','dmd')          # the dmd type to be applied. Currently we support dmd and hdmd (high order dmd)
    # variables filled up in the training stages
    # self._amplitudes                 = {}                                   # {'target1': vector of amplitudes,'target2':vector of amplitudes, etc.}
    # self._eigs                       = {}                                   # {'target1': vector of eigenvalues,'target2':vector of eigenvalues, etc.}
    # self._modes                      = {}                                   # {'target1': matrix of dynamic modes,'target2':matrix of dynamic modes, etc.}
    self.__Atilde                    = {}                                   # {'target1': matrix of lowrank operator from the SVD,'target2':matrix of lowrank operator from the SVD, etc.}
    self.__Btilde                    = {}
    self.pivotVals                   = []                                   # pivot values (e.g. U), the variable names are in self.pivotParameterID
    self.stateID                     = []                                   # state variables names (e.g. X)
    self.stateVals                   = []                                   # state values (e.g. X)
    self.timeScales                  = {}                                   # time-scales (training and dmd). {'training' and 'dmd':{t0:float,'dt':float,'intervals':int}}

    # some checks
    if self.dmdParams['dmdType'] not in self.availDmdAlgorithms:
      self.raiseAnError(IOError,'dmdType(s) available are "'+', '.join(self.availDmdAlgorithms)+'"!')

  def __returnInitialParametersLocal__(self):
    # print("__returnInitialParametersLocal__")
    """
      This method returns the initial parameters of the SM
      @ In, None
      @ Out, self.dmdParams, dict, the dict of the SM settings
    """
    return self.dmdParams

  def _localNormalizeData(self,values,names,feat):
    # print("_localNormalizeData")
    """
      Overwrites default normalization procedure.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)
    # print(self.muAndSigmaFeatures)

  def __trainLocal__(self,featureVals,targetVals):
    # print("__trainLocal__")
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, numpy.ndarray, shape = [n_timeStep, n_dimensions], an array of time series data
    """
    ### Extract the Pivot Values (Actuator, U) ###
    for pivotParameter in self.pivotParameterID: # self.pivotParameterID = ['u1', 'u2']
      pivotParamIndex   = self.features.index(pivotParameter)
      self.pivotVals.append(featureVals[:,pivotParamIndex])
    self.pivotVals = np.asarray(self.pivotVals).T # self.pivotVals is Num_Entries*2 array, the snapshots of [u1, u2]

    ### Extract the time scale (discrete, in time step marks) ###
    (ts,_) = np.shape(self.pivotVals)
    self.timeScales = np.arange(0,ts,1)

    ### Extract the Target Values (Virtual Target, all 0) ###
    # print(self.target) # ['VirtualTarget']
    # print(np.shape(targetVals)) # 500*1 array, the snapshots of ['VirtualTarget']

    ### Extract the State Values (State, X) ###
    for VarID in self.features: # features = ['u1', 'u2', 'x1', 'x2', ..., 'x19']
      if VarID not in self.pivotParameterID:
        self.stateID.append(VarID)
    # print(self.stateID) # ['x1', ..., 'x19']

    for VarID in self.stateID:
      VarIndex = self.features.index(VarID)
      # print(VarID, VarIndex)
      self.stateVals.append(featureVals[:, VarIndex])
    self.stateVals = np.asarray(self.stateVals).T  # self.stateVals is Num_Entries*19 array, the snapshots of [x1, x2, ..., x19]

    X1 = self.stateVals[:-1,:].T   # 19*(Num_Entries-1) array, snapshot of X[0:Num_Entries-1]
    X2 = self.stateVals[1: ,:].T   # 19*(Num_Entries-1) array, snapshot of X[1:Num_Entries]
    U  = self.pivotVals[:-1,:].T   # 2* (Num_Entries-1) array, snapshot of U[0:Num_Entries-1]

    self.__Atilde, self.__Btilde = self.fun_DMDc(X1, X2, U, self.dmdParams['rankSVD'])

    # Default timesteps (even if the time history is not equally spaced in time, we "trick" the dmd to think it).
    self.timeScales = dict.fromkeys( ['training','dmd'],{'t0': 0, 'intervals': ts - 1, 'dt': 1})
    # print(self.timeScales) # {'training': {'t0': 0, 'intervals': 7, 'dt': 1}, 'dmd': {'t0': 0, 'intervals': 7, 'dt': 1}}

  def writeXMLPreamble(self, writeTo, targets = None):
    # print("writeXMLPreamble")
    """
      Specific local method for printing anything desired to xml file at the begin of the print.
      @ In, writeTo, xmlUtils.StaticXmlElement instance, element to write to
      @ In, targets, list, list of targets for whom information should be written.
      @ Out, None
    """
    # add description
    supervisedLearning.writeXMLPreamble(self, writeTo, targets)
    description  = ' This XML file contains the main information of the DMDC ROM.'
    description += ' The method is explained in:'
    description += ' Proctor, Joshua L., Steven L. Brunton, and J. Nathan Kutz. '
    description += ' "Dynamic mode decomposition with control." '
    description += ' SIAM Journal on Applied Dynamical Systems 15, no. 1 (2016): 142-161.'
    writeTo.addScalar('ROM',"description",description)

  def writeXML(self, writeTo, targets = None, skip = None):
    # print("writeXML")
    """
      Adds requested entries to XML node.
      @ In, writeTo, xmlTuils.StaticXmlElement, element to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    if not self.amITrained: # self.amITrained = true
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    if skip is None: # skip =  None
      skip = []

    what = ['dmdType','rankSVD','acturators','acturatorsCount','states','statesCount',
            'Atilde','Btilde','dmdTimeScale']
    # print(what) # ['dmdType', 'rankSVD', 'features', 'acturators', 'Atilde', 'Btilde', 'Ctilde', 'dmdTimeScale']

    if targets is None: # targets = None
      readWhat = what
    else:
      readWhat = targets

    for s in skip: # skip is empty, so nothing to skip
      if s in readWhat:
        readWhat.remove(s) # "readWhat" remains unchanged.

    if not set(readWhat) <= set(what): # when "readWhat" contains something outside of "what", raise error
      self.raiseAnError(IOError, "The following variables specified in <what> node are not recognized: "+ ",".join(np.setdiff1d(readWhat, what).tolist()) )
    else:
      what = readWhat

    target = 'DMDcModel'

    toAdd = ['dmdType','rankSVD']
    self.dmdParams['rankSVD'] = self.dmdParams['rankSVD'] if self.dmdParams['rankSVD'] is not None else -1

    for add in toAdd: # toAdd = ['dmdType','rankSVD']
      if add in what :
        writeTo.addScalar(target,add,self.dmdParams[add])

    targNode = writeTo._findTarget(writeTo.getRoot(), target)
    if "acturators" in what:
      writeTo.addScalar(target, "acturators", ' '.join(self.pivotParameterID))
    if "acturatorsCount" in what:
      writeTo.addScalar(target, "acturatorsCount", len(self.pivotParameterID))
    if "states" in what:
      writeTo.addScalar(target, "states", ' '.join(self.stateID))
    if "statesCount" in what:
      writeTo.addScalar(target, "statesCount", len(self.stateID))
    if "dmdTimeScale" in what:
      writeTo.addScalar(target,"dmdTimeScale",' '.join(['%.3d' % elm for elm in self.__getTimeScale()]))

    if "Atilde" in what:
      # print(self.__Atilde)
      # write the real part of Atilde
      AtildeReal = "; ".join(
        " ".join('%.6e' % self.__Atilde[row,col].real for col in range(len(self.__Atilde[0])))
        for row in range(len(self.__Atilde)))
      writeTo.addScalar("Atilde","real",AtildeReal,root=targNode)
      # write the imaginary part of Atilde
      AtildeImage = "; ".join(
        " ".join('%.6e' % self.__Atilde[row,col].imag for col in range(len(self.__Atilde[0])))
        for row in range(len(self.__Atilde)))
      writeTo.addScalar("Atilde","imaginary",AtildeImage,root=targNode)
      writeTo.addScalar("Atilde","matrixShape",",".join(str(x) for x in np.shape(self.__Atilde)),root=targNode)
      writeTo.addScalar("Atilde","formatNote","Matrix rows are separated by cartrige return ';'",root=targNode)


    if "Btilde" in what:
      # print(self.__Btilde)
      # write the real part of Btilde
      BtildeReal = "; ".join(
        " ".join('%.6e' % self.__Btilde[row,col].real for col in range(len(self.__Btilde[0])))
        for row in range(len(self.__Btilde)))
      writeTo.addScalar("Btilde","real",BtildeReal,root=targNode)
      # write the imaginary part of Btilde
      BtildeImage = "; ".join(
        " ".join('%.6e' % self.__Btilde[row,col].imag for col in range(len(self.__Btilde[0])))
        for row in range(len(self.__Btilde)))
      writeTo.addScalar("Btilde","imaginary",BtildeImage,root=targNode)
      writeTo.addScalar("Btilde","matrixShape",",".join(str(x) for x in np.shape(self.__Btilde)),root=targNode)
      writeTo.addScalar("Btilde","formatNote","Matrix rows are separated by semicolon ';'",root=targNode)

  def __getTimeScale(self,dmd=True):
    # print("__getTimeScale")
    """
      Get the ts of the dmd (if dmd = True) or training (if dmd = False) reconstructed time scale.
      @ In, dmd, bool, optional, True if dmd time scale needs to be returned, othewise training one
      @ Out, timeScale, numpy.array, the dmd or training reconstructed time scale
    """
    timeScaleInfo = self.timeScales['dmd'] if dmd else self.timeScales['training']
    timeScale = np.arange(timeScaleInfo['t0'], timeScaleInfo['intervals'] + timeScaleInfo['dt'], timeScaleInfo['dt'])
    return timeScale

  #######
  def __evaluateLocal__(self,featureVals):
    # print("__evaluateLocal__")
    """
      This method is used to inquire the DMD to evaluate (after normalization that in
      this case is not performed)  a set of points contained in featureVals.
      a KDTree algorithm is used to construct a weighting function for the reconstructed space
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    # print(np.shape(featureVals)) # (100, 21)

    ### Initialize the final return value ###
    returnEvaluation = {}

    ### Extract the Actuator signal U ###
    Eval_U = []
    for VarID in self.pivotParameterID:
      VarIndex = self.features.index(VarID)
      Eval_U.append(featureVals[:, VarIndex])
      returnEvaluation.update({VarID: featureVals[:, VarIndex]})
    Eval_U = np.asarray(Eval_U)
    _,ts_Eval = np.shape(Eval_U) # ts_Eval = 100

    ### Extract the initial state vector ###
    Eval_X = [[]]
    for VarID in self.stateID:
      VarIndex = self.features.index(VarID)
      Eval_X[0].append(featureVals[0, VarIndex])
    Eval_X = np.asarray(Eval_X).T

    ### perform the self-propagation of X, X[k+1] = A*X[k] + B*U[k] ###
    for i in range(0,ts_Eval-1):
      X_pred = np.reshape(self.__Atilde.dot(Eval_X[:,i]) + self.__Btilde.dot(Eval_U[:,i]),(-1,1))
      Eval_X = np.hstack((Eval_X,X_pred))

    ### Store the results to the dictionary "returnEvaluation"
    for VarID in self.stateID:
      VarIndex = self.stateID.index(VarID)
      returnEvaluation.update({VarID: Eval_X[VarIndex,:]})

    return returnEvaluation


  #######
  def __setstate__(self,state):
    # print("__setstate__")
    """
      Initializes the DMD with the data contained in state
      @ In, state, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(state)
    self.KDTreeFinder = spatial.KDTree(self.featureVals)

  def __confidenceLocal__(self,featureVals):
    # print("__confidenceLocal__")
    """
      The confidence associate with a set of requested evaluations
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, None
    """
    pass

  def __resetLocal__(self,featureVals):
    # print("__resetLocal__")
    """
      After this method the ROM should be described only by the initial parameter settings
      @ In, featureVals, numpy.ndarray, shape= (n_samples, n_dimensions), an array of input data (training data)
      @ Out, None
    """
    self.amITrained   = False
    self._amplitudes  = {}
    self._eigs        = {}
    self._modes       = {}
    self.__Atilde     = {}
    self.pivotValues  = None
    self.KDTreeFinder = None
    self.featureVals  = None

  def __returnCurrentSettingLocal__(self):
    # print("__returnCurrentSettingLocal__")
    """
      This method is used to pass the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, self.dmdParams, dict, the dict of the SM settings
    """
    return self.dmdParams

  def fun_DMDc(self, X1, X2, U, rankSVD):
    # Input dimensions:
    # X1, X2: both [n*L] matrices, n-dimension state vectors by L entries
    # U: [m*L] matrix, m-dimension control vector by L entries
    # A_0[n*n], B_0[n*m]: initial guess of A and B, <class 'numpy.ndarray'>
    # k: int
    # nr: int
    n = len(X2)  # Dimension of State Vector
    Omega = np.concatenate((X1, U), axis=0)  # Omega Matrix, stack X1 and U
    U, S, V = np.linalg.svd(Omega, full_matrices=True)  # Singular Value Decomp. U*S*V'=Omega
    S = np.diag(S);    V = V.T
    # print("U=",U); print("S=",S); print("V=",V)
    if rankSVD is -1:
      p = len(S)  # p is the number of non-zero element in S
    elif rankSVD is 0: # optimal rank
      omeg = lambda x: 0.56 * x ** 3 - 0.95 * x ** 2 + 1.82 * x + 1.43
      p = np.sum(S > np.median(S) * omeg(np.divide(*sorted(Omega.shape))))
    else:
      p = rankSVD

    Ut = U[:, 0:p];    St = S[0:p, 0:p];    Vt = V[:, 0:p]  # truncation for the first p elements
    # print("Ut=", Ut); print("St=", St); print("Vt=", Vt)
    U1 = Ut[0:n, :];    U2 = Ut[n:, :]  # Cut Ut into U1 (for x) and U2 (for v)
    # print(U1)
    # print(U2)
    Q, R = np.linalg.qr(St)  # QR decomp. St=Q*R, Q unitary, R upper triangular
    r = 1 / np.linalg.cond(R)  # inverse of condition number of invention: smallest eigenvalue/biggest eigenvalue
    # print("St=",St);    print("Q=",Q);    print("R=",R);    print("r=",r)

    if np.linalg.det(R) == 0:  # if R is singular matrix, raise an error
      self.raiseAnError(IOError, "The R matrix is singlular, Please check the singularity of [X1;U]!")
    else:
      beta = X2.dot(Vt).dot(np.linalg.inv(R)).dot(Q.T)
      A_id = beta.dot(U1.T)
      B_id = beta.dot(U2.T)

    # outputs:
    # A_id: [n*n]. Estimated A matrix <class 'numpy.ndarray'>
    # B_id: [n*m]. Estimated B matrix <class 'numpy.ndarray'>
    return A_id, B_id