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
# @ Authors: Mohammad Abdo  (@Jimmy-INL)
#            Congjian Wang  (@wangcj05)
#            Andrea Alfonsi (@aalfonsi)
#            Aaron Epiney   (@AaronEpiney)

#External Modules------------------------------------------------------------------------------------
import numpy as np
import xarray as xr
import scipy as sp
from scipy.linalg import sqrtm
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ravenframework.utils import InputData, InputTypes
from .. import ValidationBase
#Internal Modules End--------------------------------------------------------------------------------

class Representativity(ValidationBase):
  """
    Representativity is a base class for validation problems
    It represents the base class for most validation problems
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(Representativity, cls).getInputSpecification()
    prototypeParameters = InputData.parameterInputFactory("prototypeParameters", contentType=InputTypes.StringListType,
                      descr=r"""mock model parameters/inputs""")
    prototypeParameters.addParam("type", InputTypes.StringType)
    specs.addSub(prototypeParameters)
    targetParameters = InputData.parameterInputFactory("targetParameters", contentType=InputTypes.StringListType,
                            descr=r"""Target model parameters/inputs""")
    specs.addSub(targetParameters)
    targetPivotParameterInput = InputData.parameterInputFactory("targetPivotParameter", contentType=InputTypes.StringType,
                                descr=r"""ID of the temporal variable of the target model. Default is ``time''.
        \nb Used just in case the  \xmlNode{pivotValue}-based operation  is requested (i.e., time dependent validation).""")
    specs.addSub(targetPivotParameterInput)
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR Representativity'
    self.dynamicType = ['static'] #  for now only static is available
    self.name = 'Representativity'
    self.stat = [None, None]
    self.featureDataObject = None
    self.targetDataObject = None
    self.senPrefix = 'sen'

  def getBasicStat(self):
    """
      Get Basic Statistic PostProcessor
      @ In, None
      @ Out, stat, object, Basic Statistic PostProcessor Object
    """
    from .. import factory as ppFactory # delay import to allow definition
    stat = ppFactory.returnInstance('BasicStatistics')
    stat.what = ['NormalizedSensitivities'] # expected value calculation
    return stat

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs) != 2:
      self.raiseAnError(IOError, "PostProcessor", self.name, "can only accept two DataObjects, but got {}!".format(str(len(inputs))))
    params = self.prototypeOutputs+self.targetOutputs+self.prototypeParameters+self.targetParameters
    validParams = [True if "|" in x  else False for x in params]
    if not all(validParams):
      notValid = list(np.asarray(params)[np.where(np.asarray(validParams)==False)[0]])
      self.raiseAnError(IOError, "'prototypeParameters', 'targetParameters', 'prototypeOutputs', and 'targetOutputs' should use 'DataObjectName|Input or Output|variable' format, but variables {} do not follow this rule.".format(','.join(notValid)))
    # Assume features and targets are in the format of: DataObjectName|Variables
    names = set([x.split("|")[0] for x in self.prototypeOutputs] + [x.split("|")[0] for x in self.prototypeParameters])
    if len(names) != 1:
      self.raiseAnError(IOError, "'prototypeOutputs' and 'prototypeParameters' should come from the same DataObjects, but they present in differet DataObjects:{}".fortmat(','.join(names)))
    featDataObject = list(names)[0]
    names = set([x.split("|")[0] for x in self.targetOutputs] + [x.split("|")[0] for x in self.targetParameters])
    if len(names) != 1:
      self.raiseAnError(IOError, "'targetOutputs' and 'targetParameters' should come from the same DataObjects, but they present in differet DataObjects:{}".fortmat(','.join(names)))
    targetDataObject = list(names)[0]
    featVars = [x.split("|")[-1] for x in self.prototypeOutputs] + [x.split("|")[-1] for x in self.prototypeParameters]
    targVars = [x.split("|")[-1] for x in self.targetOutputs] + [x.split("|")[-1] for x in self.targetParameters]

    for i, inp in enumerate(inputs):
      if inp.name == featDataObject:
        self.featureDataObject = (inp, i)
      else:
        self.targetDataObject = (inp, i)

    vars = self.featureDataObject[0].vars + self.featureDataObject[0].indexes
    if not set(featVars).issubset(set(vars)):
      missing = featVars - set(vars)
      self.raiseAnError(IOError, "Variables {} are missing from DataObject {}".format(','.join(missing), self.featureDataObject[0].name))
    vars = self.targetDataObject[0].vars + self.targetDataObject[0].indexes
    if not set(targVars).issubset(set(vars)):
      missing = targVars - set(vars)
      self.raiseAnError(IOError, "Variables {} are missing from DataObject {}".format(','.join(missing), self.targetDataObject[0].name))

    featStat = self.getBasicStat()
    featStat.toDo = {'sensitivity':[{'targets':set([x.split("|")[-1] for x in self.prototypeOutputs]), 'features':set([x.split("|")[-1] for x in self.prototypeParameters]),'prefix':self.senPrefix}]}
    featStat.initialize(runInfo, [self.featureDataObject[0]], initDict)
    self.stat[self.featureDataObject[-1]] = featStat
    tartStat = self.getBasicStat()
    tartStat.toDo = {'sensitivity':[{'targets':set([x.split("|")[-1] for x in self.targetOutputs]), 'features':set([x.split("|")[-1] for x in self.targetParameters]),'prefix':self.senPrefix}]}
    tartStat.initialize(runInfo, [self.targetDataObject[0]], initDict)
    self.stat[self.targetDataObject[-1]] = tartStat


  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'prototypeParameters':
        self.prototypeParameters = child.value
      elif child.getName() == 'targetParameters':
        self.targetParameters = child.value
      elif child.getName() == 'targetPivotParameter':
        self.targetPivotParameter = child.value
    _, notFound = paramInput.findNodesAndExtractValues(['prototypeParameters',
                                                               'targetParameters'])
    # notFound must be empty
    assert(not notFound)

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it computes representativity/bias factors, corrected data, etc.

      @ In, inputIn, dictionary of data to process
      @ Out, evaluation, dict, dictionary containing the post-processed results
    """
    dataSets = [data for _, _, data in inputIn['Data']]
    pivotParameter = self.pivotParameter
    names=[]
    if isinstance(inputIn['Data'][0][-1], xr.Dataset):
      names = [self.getDataSetName(inp[-1]) for inp in inputIn['Data']]
      if len(inputIn['Data'][0][-1].indexes) > 1 and self.pivotParameter is None:
        if 'dynamic' not in self.dynamicType: #self.model.dataType:
          self.raiseAnError(IOError, "The validation algorithm '{}' is not a dynamic model but time-dependent data has been inputted in object {}".format(self._type, inputIn['Data'][0][-1].name))
        else:
          pivotParameter = self.pivotParameter
    evaluation ={k: np.atleast_1d(val) for k, val in  self._evaluate(dataSets, **{'dataobjectNames': names}).items()}

    ## TODO: This is a placeholder to remember the time dependent case
    # if pivotParameter:
    #   # Uncomment this to cause crash: print(dataSets[0], pivotParameter)
    #   if len(dataSets[0][pivotParameter]) != len(list(evaluation.values())[0]):
    #     self.raiseAnError(RuntimeError, "The pivotParameter value '{}' has size '{}' and validation output has size '{}'".format( len(dataSets[0][self.pivotParameter]), len(evaluation.values()[0])))
    #   if pivotParameter not in evaluation:
    #     evaluation[pivotParameter] = dataSets[0][pivotParameter]
    return evaluation

  def _evaluate(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    # # ## Analysis:
    # # 1. Compute mean and variance:
    # For mock model
    self._computeMoments(datasets[0], self.prototypeParameters, self.prototypeOutputs)
    measurableNames = [s.split("|")[-1] for s in self.prototypeOutputs]
    measurables = [datasets[0][var].meanValue for var in measurableNames]
    # For target model
    self._computeMoments(datasets[1], self.targetParameters, self.targetOutputs)
    FOMNames = [s.split("|")[-1] for s in self.targetOutputs]
    FOMs = np.atleast_2d([datasets[1][var].meanValue for var in FOMNames]).reshape(-1,1)
    # # 2. Propagate error from parameters to experiment and target outputs.
    # For mock model
    self._computeErrors(datasets[0],self.prototypeParameters, self.prototypeOutputs)
    measurableErrorNames = ['err_' + s.split("|")[-1] for s in self.prototypeOutputs]
    FOMErrorNames = ['err_' + s.split("|")[-1] for s in self.targetOutputs]
    self._computeMoments(datasets[0], measurableErrorNames, measurableErrorNames)
    UMeasurables = np.atleast_2d([datasets[0][var].meanValue for var in measurableErrorNames]).reshape(-1,1)
    # For target model
    self._computeErrors(datasets[1],self.targetParameters, self.targetOutputs)
    self._computeMoments(datasets[1], FOMErrorNames, FOMErrorNames)
    UFOMs = np.atleast_2d([datasets[1][var].meanValue for var in FOMErrorNames]).reshape(-1,1)
    # # 3. Compute mean and variance in the error space:
    self._computeMoments(datasets[0],['err_' + s.split("|")[-1] for s in self.prototypeParameters],['err_' + s2.split("|")[-1] for s2 in self.prototypeOutputs])
    self._computeMoments(datasets[1],['err_' + s.split("|")[-1] for s in self.targetParameters],['err_' + s2.split("|")[-1] for s2 in self.targetOutputs])
    # # 4. Compute Uncertainties in parameters
    UparVar = self._computeUncertaintyMatrixInErrors(datasets[0],['err_' + s.split("|")[-1] for s in self.prototypeParameters])
    # # 5. Compute Uncertainties in outputs
    # Outputs of Mock model (Measurables F_i)
    UMeasurablesVar = self._computeUncertaintyMatrixInErrors(datasets[0],['err_' + s.split("|")[-1] for s in self.prototypeOutputs])
    # Outputs of Target model (Targets FOM_i)
    UFOMsVar = self._computeUncertaintyMatrixInErrors(datasets[1],['err_' + s.split("|")[-1] for s in self.targetOutputs])
    # # 6. Compute Normalized Uncertainties
    # In mock experiment outputs (measurables)
    sens = self.stat[self.featureDataObject[-1]].run({"Data":[[None, None, datasets[self.featureDataObject[-1]]]]})
    # normalize sensitivities
    senMeasurables = self._generateSensitivityMatrix(self.prototypeOutputs, self.prototypeParameters, sens, datasets[0])
    # In target outputs (FOMs)
    sens = self.stat[self.targetDataObject[-1]].run({"Data":[[None, None, datasets[self.targetDataObject[-1]]]]})
    # normalize sensitivities
    senFOMs = self._generateSensitivityMatrix(self.targetOutputs, self.targetParameters, sens, datasets[1])
    # # 7. Compute representativities
    r,rExact = self._calculateBiasFactor(senMeasurables, senFOMs, UparVar, UMeasurablesVar)
    # # 8. Compute corrected Uncertainties
    UtarVarTilde = self._calculateCovofTargetErrorsfromBiasFactor(senFOMs,UparVar,r)
    UtarVarTildeExact = self._calculateCovofTargetErrorsfromBiasFactor(senFOMs,UparVar,rExact)
    # # 9 Compute Corrected Targets,
    # for var in self.targetOutputs:
    #   self._getDataFromDatasets(datasets, var, names=None)
    parametersNames = [s.split("|")[-1] for s in self.prototypeParameters]
    par = np.atleast_2d([datasets[0][var].meanValue for var in parametersNames]).reshape(-1,1)
    correctedTargets, correctedTargetCovariance, correctedTargetErrorCov, UtarVarTilde_no_Umes_var, Inner1 = self._targetCorrection(FOMs, UparVar, UMeasurables, UMeasurablesVar, senFOMs, senMeasurables)
    correctedParameters, correctedParametersCovariance = self._parameterCorrection(par, UparVar, UMeasurables, UMeasurablesVar, senMeasurables)

    # # 9. Create outputs
    """
      Assuming the number of parameters is P,
      number of measurables in the mock/prototype experiment is M,
      and the number of figure of merits (FOMS) is F, then the representativity outcomes to be reported are:

      BiasFactor: $R \in \mathbb{R}^{M \times F}$ reported element by element as BiasFactor_MockFi_TarFj
      ExactBiasFactor: same as the bias factor but assuming measureables are also uncertain.
      CorrectedParameters: best parameters to perform the measurements at parTilde \in \mathbb{R}^{P}
      UncertaintyinCorrectedParameters: $parTildeVar \in \mathbb{R}^{P \times P}$
      CorrectedTargets: $TarTilde \in \mathbb{R}^{F}$
      UncertaintyinCorrectedTargets:$TarTildeVar \in \mathbb{R}^{F \times F}$
      ExactUncertaintyinCorrectedTargets:$TarTildeVar \in \mathbb{R}^{F \times F}$
    """
    outs = {}
    for i,param in enumerate(self.prototypeParameters):
      name4 = "CorrectedParameters_{}".format(param.split("|")[-1])
      outs[name4] = correctedParameters[i]
      for j, param2 in enumerate(self.prototypeParameters):
        if param == param2:
          name5 = "VarianceInCorrectedParameters_{}".format(param.split("|")[-1])
          outs[name5] = correctedParametersCovariance[i,i]
        else:
          name6 = "CovarianceInCorrectedParameters_{}_{}".format(param.split("|")[-1],param2.split("|")[-1])
          outs[name6] = correctedParametersCovariance[i,j]

    for i,targ in enumerate(self.targetOutputs):
      name3 = "CorrectedTargets_{}".format(targ.split("|")[-1])
      outs[name3] = correctedTargets[i]
      for j,feat in enumerate(self.prototypeOutputs):
        name1 = "BiasFactor_Mock{}_Tar{}".format(feat.split("|")[-1], targ.split("|")[-1])
        name2 = "ExactBiasFactor_Mock{}_Tar{}".format(feat.split("|")[-1], targ.split("|")[-1])
        outs[name1] = r[i,j]
        outs[name2] = rExact[i,j]
      for k,tar in enumerate(self.targetOutputs):
        if k == i:
          name3 = "CorrectedVar_Tar{}".format(tar.split("|")[-1])
          name4 = "ExactCorrectedVar_Tar{}".format(tar.split("|")[-1])
        else:
          name3 = "CorrectedCov_Tar{}_Tar{}".format(targ.split("|")[-1], tar.split("|")[-1])
          name4 = "ExactCorrectedCov_Tar{}_Tar{}".format(targ.split("|")[-1], tar.split("|")[-1])
        outs[name3] = UtarVarTilde[i,k]
        outs[name4] = UtarVarTildeExact[i,k]
    return outs

  def _generateSensitivityMatrix(self, outputs, inputs, sensDict, datasets, normalize=True):
    """
      Reconstruct sensitivity matrix from the Basic Statistic calculation
      @ In, inputs, list, list of input variables
      @ In, outputs, list, list of output variables
      @ In, sensDict, dict, dictionary contains the sensitivities
      @ Out, sensMatr, numpy.array, 2-D array of the reconstructed sensitivity matrix
    """
    sensMatr = np.zeros((len(outputs), len(inputs)))
    inputVars = [x.split("|")[-1] for x in inputs]
    outputVars = [x.split("|")[-1] for x in outputs]
    for i, outVar in enumerate(outputVars):
      for j, inpVar in enumerate(inputVars):
        senName = "{}_{}_{}".format(self.senPrefix, outVar, inpVar)
        # Assume static data (PointSets are provided as input)
        if not normalize:
          sensMatr[i, j] = sensDict[senName][0]
        else:
          sensMatr[i, j] = sensDict[senName][0]* datasets[inpVar].meanValue / datasets[outVar].meanValue
    return sensMatr

  def _getDataFromDatasets(self, datasets, var, names=None):
    """
      Utility function to retrieve the data from datasets
      @ In, datasets, list, list of datasets (data1,data2,etc.) to search from.
      @ In, names, list, optional, list of datasets names (data1,data2,etc.). If not present, the search will be done on the full list.
      @ In, var, str, the variable to find (either in fromat dataobject|var or simply var)
      @ Out, data, tuple(numpy.ndarray, xarray.DataArray or None), the retrived data (data, probability weights (None if not present))
    """
    data = None
    pw = None
    dat = None
    if "|" in var and names is not None:
      do, feat =  var.split("|")
      doindex = names.index(do)
      dat = datasets[doindex][feat]
    else:
      for doindex, ds in enumerate(datasets):
        if var in ds:
          dat = ds[var]
          break
    if 'ProbabilityWeight-{}'.format(feat) in datasets[names.index(do)]:
      pw = datasets[doindex]['ProbabilityWeight-{}'.format(feat)].values
    elif 'ProbabilityWeight' in datasets[names.index(do)]:
      pw = datasets[doindex]['ProbabilityWeight'].values
    dim = len(dat.shape)
    dat = dat.values
    if dim == 1:
      #  the following reshaping does not require a copy
      dat.shape = (dat.shape[0], 1)
    data = dat, pw
    return data

  def _computeMoments(self, datasets, features, targets):
    """
      A utility function to compute moments, mean value, variance and covariance
        @ In, datasets, xarray datasets, datasets containing prototype (mock) data and target data
        @ In, features, names of feature variables: measurables
        @ In, targets, names of target variables: figures of merit (FOMs)
        @ out, datasets, xarray datasets, datasets after adding moments
    """
    for var in [x.split("|")[-1] for x in features + targets]: #datasets.data_vars
      datasets[var].attrs['meanValue'] = np.mean(datasets[var].values)
      for var2 in [x.split("|")[-1] for x in features + targets]:
        if var == var2:
          datasets[var2].attrs['var'] = np.var(datasets[var].values)
        else:
          datasets[var2].attrs['cov_'+str(var)] = np.cov(datasets[var2].values,datasets[var].values)
    return datasets

  def _computeErrors(self,datasets,features,targets):
    """
      A utility function to transform variables to the relative error of these variable
      @ In, datasets, xarray datasets, datasets containing prototype (mock) data and target data
      @ In, features, names of feature variables: measurables
      @ In, targets, names of target variables: figures of merit (FOMs)
      @ out, datasets, xarray datasets, datasets after computing errors in each variable
    """
    for var in [x.split("|")[-1] for x in features + targets]:
      datasets['err_'+str(var)] = (datasets[var].values - datasets[var].attrs['meanValue'])/datasets[var].attrs['meanValue']

  def _computeUncertaintyMatrixInErrors(self, data, parameters):
    """
      A utility function to variance and covariance of variables in the error space
      @ In, data, xarray dataset, data containing either prototype (mock) data or target data
      @ In, parameters, names of parameters/inputs to each model
      @ out, uncertMatr, np.array, The variance covariance matrix of errors
    """
    uncertMatr = np.zeros((len(parameters), len(parameters)))
    for i, var1 in enumerate(parameters):
      for j, var2 in enumerate(parameters):
        if var1 == var2:
          uncertMatr[i, j] = data[var1].attrs['var']
        else:
          uncertMatr[i, j] = data[var1].attrs['cov_'+var2][0,1]
    return uncertMatr

  def _calculateBiasFactor(self, normalizedSenExp, normalizedSenTar, UparVar, UmesVar=None):
    """
      A utility function to compute the bias factor (i.e., representativity factor)
      @ In, normalizedSenExp, np.array, the normalized sensitivities of the mock/prototype measurables
      @ In, normalizedSenTar, np.array, the normalized sensitivities of the target variables/Figures of merit (FOMs) with respect to the parameters
      @ In, UparVar, np.array, variance covariance matrix of the parameters error
      @ In, UmesVar, np.array, variance covariance matrix of the measurables error, default is None
      @ Out, r, np.array, the representativity (bias factor) matrix neglecting uncertainties in measurables
      @ Out, rExact, np.array, the representativity (bias factor) matrix considering uncertainties in measurables
    """


    UmesVar_r = np.array([[ 0.03603028, 0.01503859, 0.00722245], [ 0.01503859, 0.00766866, 0.00507709], [ 0.00722245, 0.00507709, 0.00424897]])
    normalizedSenTar_r = np.array([[-0.86540007, 1.86540007], [ 0.08002442, 0.91997558], [ 0.41033666, 0.58966334]])
    normalizedSenExp_r = np.array([[-0.80794013, 1.80794013], [ 0.07731272, 0.92268728], [ 0.40131466, 0.59868534]])
    UparVar_r=np.array([[ 0.00774389, -0.00049797], [-0.00049797, 0.00903588]])
    r_r = np.array([[ 0.95693156, 0.16791481, -0.12086468], [ 0.15894451, 0.37959475, 0.46035267], [-0.11985607, 0.45355145, 0.66341829]])
    rE_r = np.array([[ 0.67739381, 0.11855779, -0.08918473], [ 0.1120806 , 0.2671181 , 0.3267284 ], [-0.08542844, 0.31902209, 0.47204009]])
    if UmesVar is None:
      UmesVar = np.zeros((len(normalizedSenExp), len(normalizedSenExp)))
    # Compute representativity (#eq 79)
    r = (sp.linalg.pinv(sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T)) @ (normalizedSenTar @ UparVar @ normalizedSenExp.T) @ sp.linalg.pinv(sqrtm(normalizedSenExp @ UparVar @ normalizedSenExp.T))).real
    rExact = (sp.linalg.pinv(sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T)) @ (normalizedSenTar @ UparVar @ normalizedSenExp.T) @ sp.linalg.pinv(sqrtm(normalizedSenExp @ UparVar @ normalizedSenExp.T + UmesVar))).real
    print('UmesVar', UmesVar)
    print('normalizedSenExp',normalizedSenExp)
    print("normalizedSenTar",normalizedSenTar)
    print("UparVar", UparVar)
    print('r',r )
    print('rExact', rExact)
    exp = sqrtm(normalizedSenExp @ UparVar @ normalizedSenExp.T)
    tar = sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T)
    print('Exp cond: sqrtm(normalizedSenExp @ UparVar @ normalizedSenExp.T)', np.linalg.cond(exp))
    print('Tar cond: sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T)', np.linalg.cond(tar))
    print('UmesVar Error', UmesVar-UmesVar_r)
    print('normalizedSenExp Error',normalizedSenExp-normalizedSenExp_r)
    print("normalizedSenTar Error",normalizedSenTar-normalizedSenTar_r)
    print("UparVar Error", UparVar-UparVar_r)
    print('r Error',r -r_r)
    print('rExact Error', rExact-rE_r)

    return r, rExact

  def _calculateCovofTargetErrorsfromBiasFactor(self, normalizedSenTar, UparVar, r):
    """
      A utility function to compute variance covariance matrix of the taget errors from the bias factors
      @ In, normalizedSenTar, np.array, the normalized sensitivities of the targets
      @ In, UparVar, np.array, the variance covariance matrix of the parameters in the error space
      @ In, r, np.array, the bias factor matrix
      @ Out, UtarVarTilde, np.array, the variance convariance matrix of error in the corrected targets
    """
    # re-compute Utar_var_tilde from r (#eq 80)
    chol = sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T).real
    UtarVarTilde =  chol @ (np.eye(np.shape(r)[0]) - r @ r.T) @ chol
    return UtarVarTilde

  def _parameterCorrection(self, par, UparVar, Umes, UmesVar, normalizedSen): #eq 48 and eq 67
    """
      A utility function that computes the correction in parameters
      @ In, par, np.array, the parameters (inputs) of the mock experiment
      @ In, UparVar, np.array, variance covariance matrix of the parameters in the error space
      @ In, Umes, np.array, the error in measurements
      @ In, UmesVar, np.array, variance covariance matrix of the measurables in the error space
      @ In, normalizedSen, np.array, the normalized sensitivity matrix
      @ Out, parTilde, np.array, the corrected parameters
      @ Out, parTildeVar, np.array, the variance covariance matrix of the corrected parameters (uncertainty in the corrected parameters)
    """
    # Compute adjusted par #eq 48
    UparTilde = UparVar @ normalizedSen.T @ np.linalg.pinv(normalizedSen @ UparVar @ normalizedSen.T + UmesVar) @ Umes

    # back transform to parameters
    parTilde = UparTilde * par + par

    # Compute adjusted par_var #eq 67
    UparVarTilde = UparVar - UparVar @ normalizedSen.T @ np.linalg.pinv(normalizedSen @ UparVar @ normalizedSen.T + UmesVar) @ normalizedSen @ UparVar

    # back transform the variance
    UparVarTildeDiag = np.diagonal(UparVarTilde)
    for ind,c in enumerate(UparVarTildeDiag):
        if c<0:
            UparVarTilde[ind,ind] = 0
    UparVarTildeDiag2 = np.sqrt(UparVarTildeDiag)
    UparVarTildeDiag3 = UparVarTildeDiag2 * np.squeeze(par)
    parVarTilde = np.square(UparVarTildeDiag3)
    parVarTilde = np.diag(parVarTilde)
    return parTilde, parVarTilde

  def _targetCorrection(self, FOMs, UparVar, Umes, UmesVar, normalizedSenTar, normalizedSenExp):
    """
      A utility function to compute corrections in targets based on the representativity analysis
      @ In, FOMs, np.array, target out puts (Figures of merit)
      @ In, UparVar, np.array, np.array, variance covariance matrix of the parameters in the error space
      @ In, Umes, np.array, the error in measurements
      @ In, UmesVar, np.array, variance covariance matrix of the measurables in the error space
      @ In, normalizedSenTar, np.array, normalized sensitivities of the target outputs w.r.t. parameterts
      @ In, normalizedSenExp, np.array, normalized sensitivities of the mock prototype/experiment outputs (measurements) w.r.t. parameterts
      @ Out, tarTilde, np.array, corrected targets (FOMs)
      @ Out, tarVarTilde, np.array, variance covariance matrix for the corrected targets
      @ Out, UtarVarTilde,  np.array, variance covariance matrix for the corrected targets in error space
      @ Out, UtarVartilde_no_UmesVar, np.array, variance covariance matrix for the corrected targets in error space assuming no uncer
      @ Out, propagetedExpUncert, np.array, propagated variance covariance matrix of experiments due to parameter uncertainties
    """
    # Compute adjusted target #eq 71
    UtarTilde = normalizedSenTar @ UparVar @ normalizedSenExp.T @ np.linalg.pinv(normalizedSenExp @ UparVar @ normalizedSenTar.T + UmesVar) @ Umes
    # back transform to parameters
    tarTilde = UtarTilde * FOMs + FOMs

    # Compute adjusted par_var #eq 74
    UtarVarTilde = normalizedSenTar @ UparVar @ normalizedSenTar.T - normalizedSenTar @ UparVar @ normalizedSenExp.T @ np.linalg.pinv(normalizedSenExp @ UparVar @ normalizedSenTar.T + UmesVar) @ normalizedSenExp @ UparVar @ normalizedSenTar.T

    # back transform the variance
    UtarVarTildeDiag = np.diagonal(UtarVarTilde)
    for ind,c in enumerate(UtarVarTildeDiag):
        if c<0:
            UtarVarTilde[ind,ind] = 0
    UtarVarTildeDiag2 = np.sqrt(UtarVarTildeDiag)
    UtarVarTildeDiag3 = UtarVarTildeDiag2 * np.squeeze(FOMs)
    tarVarTilde = np.square(UtarVarTildeDiag3)
    tarVarTilde = np.diag(tarVarTilde)

    # Compute adjusted par_var neglecting UmesVar (to compare to representativity)
    # The representativity (#eq 79 negelcts UmesVar)
    propagetedExpUncert = (normalizedSenExp @ UparVar) @ normalizedSenExp.T
    UtarVarztilde_no_UmesVar = (normalizedSenTar @ UparVar @ normalizedSenTar.T)\
                                - (normalizedSenTar @ UparVar @ normalizedSenExp.T)\
                                @ np.linalg.pinv(normalizedSenExp @ UparVar @ normalizedSenExp.T)\
                                @ (normalizedSenExp @ UparVar @ normalizedSenTar.T)
    return tarTilde, tarVarTilde, UtarVarTilde, UtarVarztilde_no_UmesVar, propagetedExpUncert
