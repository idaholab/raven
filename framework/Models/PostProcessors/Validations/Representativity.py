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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import InputData, InputTypes
from utils import utils
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
    from Models.PostProcessors import factory as ppFactory # delay import to allow definition
    self.printTag = 'POSTPROCESSOR Representativity'
    self.dynamicType = ['static','dynamic'] #  for now only static is available
    self.acceptableMetrics = ["RepresentativityFactors"] #  acceptable metrics
    self.name = 'Representativity'
    self.stat = ppFactory.returnInstance('BasicStatistics')
    self.stat.what = ['NormalizedSensitivities'] # expected value calculation

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    self.stat.toDo = {'NormalizedSensitivity':[{'targets':set([x.split("|")[1] for x in self.features]), 'features':set([x.split("|")[1] for x in self.featureParameters]),'prefix':'nsen'}]}
    self.stat.initialize(runInfo, inputs, initDict)

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
    senMeasurables = self.stat.run({"Data":[[None, None, datasets[0]]]})
    senFOMs = self.stat.run({"Data":[[None, None, datasets[1]]]})

    names = kwargs.get('dataobjectNames')
    outs = {}
    for feat, targ, param, targParam in zip(self.features, self.targets, self.featureParameters, self.targetParameters):
      featData = self._getDataFromDatasets(datasets, feat, names)
      targData = self._getDataFromDatasets(datasets, targ, names)
      parameters = self._getDataFromDatasets(datasets, param, names)
      targetParameters = self._getDataFromDatasets(datasets, targParam, names)
      covParameters = senFOMs @ senMeasurables.T
      for metric in self.metrics:
        name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.name)
        outs[name] = metric.evaluate((featData, targData), senFOMs = senFOMs, senMeasurables=senMeasurables, covParameters=covParameters)
    return outs

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