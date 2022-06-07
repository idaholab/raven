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
  Created on April 29, 2021

  @author: Mohammad Abdo (@Jimmy-INL)

  This class represents a base class for the validation algorithms
  It inherits from the PostProcessor directly
  ##TODO: Recast it once the new PostProcesso API gets in place
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import xarray as xr
import scipy as sp
from scipy.linalg import sqrtm
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ravenframework.utils import InputData, InputTypes
from ravenframework.utils import utils
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
    parametersInput = InputData.parameterInputFactory("featureParameters", contentType=InputTypes.StringListType,
                      descr=r"""mock model parameters/inputs""")
    parametersInput.addParam("type", InputTypes.StringType)
    specs.addSub(parametersInput)
    targetParametersInput = InputData.parameterInputFactory("targetParameters", contentType=InputTypes.StringListType,
                            descr=r"""Target model parameters/inputs""")
    targetParametersInput.addParam("type", InputTypes.StringType)
    specs.addSub(targetParametersInput)
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
    params = self.features+self.targets+self.featureParameters+self.targetParameters
    validParams = [True if "|" in x  else False for x in params]
    if not all(validParams):
      notValid = list(np.asarray(params)[np.where(np.asarray(validParams)==False)[0]])
      self.raiseAnError(IOError, "'Features', 'Targets', 'featureParameters', and 'targetParameters' should use 'DataObjectName|variable' format, but variables {} do not follow this rule.".format(','.join(notValid)))
    # Assume features and targets are in the format of: DataObjectName|Variables
    names = set([x.split("|")[0] for x in self.features] + [x.split("|")[0] for x in self.featureParameters])
    if len(names) != 1:
      self.raiseAnError(IOError, "'Features' and 'featureParameters' should come from the same DataObjects, but they present in differet DataObjects:{}".fortmat(','.join(names)))
    featDataObject = list(names)[0]
    names = set([x.split("|")[0] for x in self.targets] + [x.split("|")[0] for x in self.targetParameters])
    if len(names) != 1:
      self.raiseAnError(IOError, "'Targets' and 'targetParameters' should come from the same DataObjects, but they present in differet DataObjects:{}".fortmat(','.join(names)))
    targetDataObject = list(names)[0]
    featVars = [x.split("|")[-1] for x in self.features] + [x.split("|")[1] for x in self.featureParameters]
    targVars = [x.split("|")[-1] for x in self.targets] + [x.split("|")[1] for x in self.targetParameters]

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
    featStat.toDo = {'sensitivity':[{'targets':set([x.split("|")[-1] for x in self.features]), 'features':set([x.split("|")[-1] for x in self.featureParameters]),'prefix':self.senPrefix}]}
    featStat.initialize(runInfo, [self.featureDataObject[0]], initDict)
    self.stat[self.featureDataObject[-1]] = featStat
    tartStat = self.getBasicStat()
    tartStat.toDo = {'sensitivity':[{'targets':set([x.split("|")[-1] for x in self.targets]), 'features':set([x.split("|")[-1] for x in self.targetParameters]),'prefix':self.senPrefix}]}
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
      if child.getName() == 'featureParameters':
        self.featureParameters = child.value
      elif child.getName() == 'targetParameters':
        self.targetParameters = child.value
      elif child.getName() == 'targetPivotParameter':
        self.targetPivotParameter = child.value

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    dataSets = [data for _, _, data in inputIn['Data']]
    pivotParameter = self.pivotParameter
    names=[]
    if isinstance(inputIn['Data'][0][-1], xr.Dataset):
      names = [self.getDataSetName(inp[-1]) for inp in inputIn['Data']]
      if len(inputIn['Data'][0][-1].indexes) and self.pivotParameter is None:
        if 'dynamic' not in self.dynamicType: #self.model.dataType:
          self.raiseAnError(IOError, "The validation algorithm '{}' is not a dynamic model but time-dependent data has been inputted in object {}".format(self._type, inputIn['Data'][0][-1].name))
        else:
          pivotParameter = self.pivotParameter
    evaluation ={k: np.atleast_1d(val) for k, val in  self._evaluate(dataSets, **{'dataobjectNames': names}).items()}#inputIn
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
    self._computeMoments(datasets[0], self.featureParameters, self.features)
    # For target model
    self._computeMoments(datasets[1], self.targetParameters, self.targets)
    # # 2. Propagate error from parameters to experiment and target outputs.
    # For mock model
    self._computeErrors(datasets[0],self.featureParameters, self.features)
    # For target model
    self._computeErrors(datasets[1],self.targetParameters, self.targets)
    # # 3. Compute mean and variance in the error space:
    self._computeMoments(datasets[0],['err_' + s.split("|")[-1] for s in self.featureParameters],['err_' + s2.split("|")[-1] for s2 in self.features])
    self._computeMoments(datasets[1],['err_' + s.split("|")[-1] for s in self.targetParameters],['err_' + s2.split("|")[-1] for s2 in self.targets])
    # # 4. Compute Uncertainties in parameters
    UParVar = self._computeUncertaintyinParametersErrorMatrix(datasets[0],['err_' + s.split("|")[-1] for s in self.featureParameters])
    # # 5. Compute Uncertainties in outputs
    # Outputs of Mock model (Measurables F_i)
    UMeasurablesVar = self._computeUncertaintyinParametersErrorMatrix(datasets[0],['err_' + s.split("|")[-1] for s in self.features])
    # Outputs of Target model (Targets FOM_i)
    UFOMsVar = self._computeUncertaintyinParametersErrorMatrix(datasets[1],['err_' + s.split("|")[-1] for s in self.targets])
    # # 6. Compute Normalized Uncertainties
    # In mock experiment outputs (measurables)
    sens = self.stat[self.featureDataObject[-1]].run({"Data":[[None, None, datasets[self.featureDataObject[-1]]]]})
    senMeasurables = self._generateSensitivityMatrix(self.features, self.featureParameters, sens, datasets)
    # In target outputs (FOMs)
    sens = self.stat[self.targetDataObject[-1]].run({"Data":[[None, None, datasets[self.targetDataObject[-1]]]]})
    senFOMs = self._generateSensitivityMatrix(self.targets, self.targetParameters, sens, datasets)
    # # 7. Compute representativities
    r,rExact = self._calculateBiasFactor(senMeasurables, senFOMs, UParVar, UMeasurablesVar)
    # # 8. Compute corrected Uncertainties
    UtarVarTilde = self._calculateCovofTargetErrorsfromBiasFactor(senFOMs,UParVar,r)
    UtarVarTildeExact = self._calculateCovofTargetErrorsfromBiasFactor(senFOMs,UParVar,rExact)
    # # 9. Create outputs
    outs = {}
    for i,targ in enumerate(self.targets):
      for j,feat in enumerate(self.features):
        name1 = "BiasFactor_Mock{}_Tar{}".format(feat.split("|")[-1], targ.split("|")[-1])
        name2 = "ExactBiasFactor_Mock{}_Tar{}".format(feat.split("|")[-1], targ.split("|")[-1])
        outs[name1] = r[i,j]
        outs[name2] = rExact[i,j]
      for k,tar in enumerate(self.targets):
        if k == i:
          name3 = "CorrectedVar_Tar{}".format(tar.split("|")[-1])
          name4 = "ExactCorrectedVar_Tar{}".format(tar.split("|")[-1])
        else:
          name3 = "CorrectedCov_Tar{}_Tar{}".format(targ.split("|")[-1], tar.split("|")[-1])
          name4 = "ExactCorrectedCov_Tar{}_Tar{}".format(targ.split("|")[-1], tar.split("|")[-1])
        outs[name3] = UtarVarTilde[i,k]
        outs[name4] = UtarVarTildeExact[i,k]
    ## TODO:
    # # 5. Compute correction in parameters
    # par_tilde, par_var_tilde = Parameter_correction_theory(par, Upar, Upar_var, Umes, Umes_var, expNormalizedSen)
    # pm_tilde, Cpm_tilde = Parameter_correction_theory(par, Upar, Upar_var,UF, UF_var,mesParametersNormalizedSen)
    # # 6. Compute correction in targets
    # FOMsim_tilde_theory, FOMsim_var_tilde_theory, UFOMsim_var_tilde_theory, Umes_var, UFOM_var_tilde_no_Umes_var, Inner1  = Target_correction_theory(par, FOM, Upar, Upar_var, Umes, Umes_var, mesParametersNormalizedSen, expNormalizedSen)
    # # 7. Computer representativity factor
    # r,r_exact, UFOMsim_var_tilde_rep,UFOMsim_var_tilde_rep_exact   = Representativity(par, Upar, Upar_var, F, nSF, nSFOM, Umes_var)
    # print('==== Representativity ====')
    # print('r')
    # print(r)
    # print('UFOMsim_var_tilde_rep')
    # print(UFOMsim_var_tilde_rep)
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
          sensMatr[i, j] = sensDict[senName][0]* datasets[0][inpVar].meanValue / datasets[0][outVar].meanValue
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
    for var in [x.split("|")[-1] for x in features + targets]: #datasets.data_vars
      datasets[var].attrs['meanValue'] = np.mean(datasets[var].values)
      for var2 in [x.split("|")[-1] for x in features + targets]:
        if var == var2:
          datasets[var2].attrs['var'] = np.var(datasets[var].values)
        else:
          datasets[var2].attrs['cov_'+str(var)] = np.cov(datasets[var2].values,datasets[var].values)
    return datasets

  def _computeErrors(self,datasets,features,targets):
    for var in [x.split("|")[-1] for x in features + targets]:
      datasets['err_'+str(var)] = (datasets[var].values - datasets[var].attrs['meanValue'])/datasets[var].attrs['meanValue']
      # for var2 in [x.split("|")[-1] for x in features + targets]:
      #   datasets[var].attrs['err_cov_'+str(var)] = np.cov((datasets[var2].values - datasets[var2].attrs['meanValue'])/datasets[var2].attrs['meanValue'],(datasets[var2].values - datasets[var2].attrs['meanValue'])/datasets[var].attrs['meanValue'])

  # def _propagateErrors(self,data):
  #   # par = [data[var.split("|")[1]] for var in self.featureParameters]
  #   # par_var = xr.cov(par[0],par[1])
  #   # Trans_samples = np.zeros((np.shape(data)[0],np.shape(A)[0]))
  #   for ind,samp in enumerate(data):
  #     Trans_samples[ind,:] = linModel(A,samp,b)
  #   Avg = np.average(Trans_samples, axis=0)
  #   C = np.cov(Trans_samples.T)
  #   return Avg, C, Trans_samples

  def _computeUncertaintyinParametersErrorMatrix(self, data, parameters):

    uncertMatr = np.zeros((len(parameters), len(parameters)))
    # inputVars = [x.split("|")[-1] for x in parameters]
    # outputVars = [x.split("|")[-1] for x in outputs]
    for i, var1 in enumerate(parameters):
      for j, var2 in enumerate(parameters):
        if var1 == var2:
          uncertMatr[i, j] = data[var1].attrs['var']
        else:
          uncertMatr[i, j] = data[var1].attrs['cov_'+var2][0,1]
    return uncertMatr

  def _ParameterCorrectionTheory(par, Upar, UparVar, Umes, UmesVar, normalizedSen):
    pass

  def _calculateBiasFactor(self, normalizedSenExp, normalizedSenTar, UparVar, UmesVar=None):
    # Compute representativity (#eq 79)
    r = (sp.linalg.pinv(sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T)) @ sqrtm(normalizedSenTar @ UparVar @ normalizedSenExp.T) @ sqrtm(normalizedSenTar @ UparVar @ normalizedSenExp.T) @ sp.linalg.pinv(sqrtm(normalizedSenExp @ UparVar @ normalizedSenExp.T))).real
    rExact = (sp.linalg.pinv(sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T)) @ sqrtm(normalizedSenTar @ UparVar @ normalizedSenExp.T) @ sqrtm(normalizedSenTar @ UparVar @ normalizedSenExp.T) @ sp.linalg.pinv(sqrtm(normalizedSenExp @ UparVar @ normalizedSenExp.T + UmesVar))).real
    return r, rExact

  def _calculateCovofTargetErrorsfromBiasFactor(self, normalizedSenTar, UparVar, r):
    # re-compute Utar_var_tilde from r (#eq 80)
    chol = sqrtm(normalizedSenTar @ UparVar @ normalizedSenTar.T).real
    UtarVarTilde =  chol @ (np.eye(np.shape(r)[0]) - r @ r.T) @ chol
    return UtarVarTilde