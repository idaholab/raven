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
Created on Aug 4, 2020

@author: ZHOUJ2
"""
#External Modules---------------------------------------------------------------
import numpy as np
import xarray as xr
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from .BasicStatistics import BasicStatistics
from utils import utils
from utils import InputData, InputTypes
#Internal Modules End-----------------------------------------------------------

class EconomicRatio(BasicStatistics):
  """
    EconomicRatio filter class. It computes economic metrics
  """

  # values from BasicStatistics
  scalarVals =   BasicStatistics.scalarVals
  vectorVals =   BasicStatistics.vectorVals
  steVals    =   BasicStatistics.steVals

  # economic/financial metrics
  tealVals   = ['sharpeRatio',             #financial metric
                'sortinoRatio',            #financial metric
                'gainLossRatio',           #financial metric
                'valueAtRisk',             # Value at risk (alpha)
                'expectedShortfall'        # conditional value at risk (gammma)
                ]

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """

    # get input specification from BasicStatistics (scalarVals and vectorVals)
    inputSpecification = super(EconomicRatio, cls).getInputSpecification()

    # add tealVals
    for teal in cls.tealVals:
      tealSpecification = InputData.parameterInputFactory(teal,
                                                          contentType=InputTypes.StringListType)
      if teal in ["sortinoRatio", "gainLossRatio"]:
        tealSpecification.addParam("threshold", InputTypes.StringType)
      elif teal in ["expectedShortfall", "valueAtRisk"]:
        tealSpecification.addParam("threshold", InputTypes.FloatType)
      tealSpecification.addParam("prefix", InputTypes.StringType)
      inputSpecification.addSub(tealSpecification)

    return inputSpecification

  def __init__(self):
    super().__init__()
    self.printTag = "PostProcessor ECONOMIC RATIO"

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the EconomicRatio pp. In here the working dir is
      grepped.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """

    # construct a list of all the parameters that have requested values from scalarVals
    # and vectorVals into self.allUsedParams
    super().initialize(runInfo, inputs, initDict)

    # add a list of all the parameters that have requested values from tealVals into
    # self.allUsedParams
    for metricName in self.tealVals:
      if metricName in self.toDo.keys():
        for entry in self.toDo[metricName]:
          self.allUsedParams.update(entry["targets"])
          try:
            self.allUsedParams.update(entry["features"])
          except KeyError:
            pass

    # for backward compatibility, compile the full list of parameters used in EconomicRatio calculations
    self.parameters["targets"] = list(self.allUsedParams)
    inputObj = inputs[-1]  if type(inputs) == list else inputs
    if inputObj.type == "HistorySet":
      self.dynamic = True
    inputMetaKeys = []
    outputMetaKeys = []
    for metric, infos in self.toDo.items():
      steMetric = metric + "_ste"
      if steMetric in self.steVals:
        for info in infos:
          prefix = info["prefix"]
          for target in info["targets"]:
            metaVar = prefix + "_ste_" + target if not self.outputDataset else metric + "_ste"
            metaDim = inputObj.getDimensions(target)
            if len(metaDim[target]) == 0:
              inputMetaKeys.append(metaVar)
            else:
              outputMetaKeys.append(metaVar)
    metaParams = {}
    if not self.outputDataset:
      if len(outputMetaKeys) > 0:
        metaParams = {key:[self.pivotParameter] for key in outputMetaKeys}
    else:
      if len(outputMetaKeys) > 0:
        params = {key:[self.pivotParameter, self.steMetaIndex] for key in outputMetaKeys + inputMetaKeys}
        metaParams.update(params)
      elif len(inputMetaKeys) > 0:
        params = {key:[self.steMetaIndex] for key in inputMetaKeys}
        metaParams.update(params)
    metaKeys = inputMetaKeys + outputMetaKeys
    self.addMetaKeys(metaKeys, metaParams)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """

    # handle scalarVals and vectorVals from BasicStatistics
    super()._handleInput(paramInput, childVals=self.tealVals)

    # now handle tealVals
    for child in paramInput.subparts:
      tag = child.getName()
      if tag in self.tealVals:
        if "prefix" not in child.parameterValues:
          self.raiseAnError(IOError, "No prefix is provided for node: ", tag)
        # get the prefix
        prefix = child.parameterValues["prefix"]
        if tag in ["sortinoRatio", "gainLossRatio"]:
          # get targets
          targets = set(child.value)
          # if targets are not specified by user
          if len(targets) < 1:
            self.raiseAWarning("No targets were specified in text of <"+tag+">!  Skipping metric...")
            continue
          if tag not in self.toDo.keys():
            self.toDo[tag] = [] # list of {"targets": (), "prefix": str, "threshold": str}
          if "threshold" not in child.parameterValues:
            threshold = "zero"
          else:
            threshold = child.parameterValues["threshold"].lower()
            if threshold not in ["zero", "median"]:
              self.raiseAWarning("Unrecognized threshold in {}, prefix '{}' use zero instead!".format(tag, prefix))
              threshold = "zero"

          if "expectedValue" not in self.toDo.keys():
            self.toDo["expectedValue"] = []
          if "median" not in self.toDo.keys():
            self.toDo["median"] = []
          # add any expectedValue targets that are missing
          expectedValueToAdd = self._additionalTargetsToAdd("expectedValue", child.value)
          if len(expectedValueToAdd) > 0:
            self.toDo["expectedValue"].append({"targets": expectedValueToAdd, "prefix": "BSMean"})
          # add any median targets that are missing
          medianValueToAdd = self._additionalTargetsToAdd("median", child.value)
          if len(medianValueToAdd) > 0:
            self.toDo["median"].append({"targets": medianValueToAdd, "prefix": "BSMED"})
          self.toDo[tag].append({"targets": set(targets), "prefix": prefix, "threshold": threshold})
        elif tag in ["expectedShortfall", "valueAtRisk"]:
          # get targets
          targets = set(child.value)
          if tag not in self.toDo.keys():
            self.toDo[tag] = [] # list of {"targets": (), "prefix": str, "threshold": str}
            if "threshold" not in child.parameterValues:
              threshold = 0.05
            else:
              threshold = child.parameterValues["threshold"]
              if threshold > 1 or threshold < 0:
                self.raiseAnError("Threshold in {}, prefix '{}' out of range, please use a float in range (0, 1)!".format(tag, prefix))
            self.toDo[tag].append({"targets": set(targets), "prefix": prefix, "threshold": threshold})
        else:
          if tag not in self.toDo.keys():
            self.toDo[tag] = [] # list of {"targets": (), "prefix": str}
          if "expectedValue" not in self.toDo.keys():
            self.toDo["expectedValue"] = []
          if "sigma" not in self.toDo.keys():
            self.toDo["sigma"] = []
          # add any expectedValue targets that are missing
          expectedValueToAdd = self._additionalTargetsToAdd("expectedValue", child.value)
          if len(expectedValueToAdd) > 0:
            self.toDo["expectedValue"].append({"targets": expectedValueToAdd, "prefix": "BSMean"})
          # add any sigma targets that are missing
          sigmaToAdd = self._additionalTargetsToAdd("sigma", child.value)
          if len(sigmaToAdd) > 0:
            self.toDo["sigma"].append({"targets": sigmaToAdd, "prefix": "BSSigma"})
          self.toDo[tag].append({"targets": set(child.value), "prefix": prefix})
      else:
        if tag not in self.scalarVals + self.vectorVals:
          self.raiseAWarning("Unrecognized node in EconomicRatio '" + tag + "' has been ignored!")

    assert(len(self.toDo) > 0), self.raiseAnError(IOError, "EconomicRatio needs parameters to work on! please check input for PP: " + self.name)

  def _additionalTargetsToAdd(self, metric, childValue):
    """
      Some EconomicRatio metrics require additional metrics from BasicStatistics, find necessary additions
      @ In, metric, str, name of metric
      @ In, childValue, set, set of targets
      @ Out, valsToAdd, set, set of targets to add to toDo
    """

    if len(self.toDo[metric]) > 0:
      # check if metric should be added
      vals = set() # currently collected
      for tmp_dict in self.toDo[metric]:
        if len(vals) == 0:
          vals = set(tmp_dict["targets"])
        else:
          vals = vals.union(tmp_dict["targets"])
      valsToAdd = set()
      for val in childValue:
        if val not in vals:
          valsToAdd.add(val)
    else:
      valsToAdd = set(childValue)

    return valsToAdd

  # TODO: update if necessary
  def __computePower(self, p, dataset):
    """
      Compute the p-th power of weights
      @ In, p, int, the power
      @ In, dataset, xarray.Dataset, probability weights of all input variables
      @ Out, pw, xarray.Dataset, the p-th power of weights
    """
    pw = {}
    coords = dataset.coords
    for target, targValue in dataset.variables.items():
      ##remove index variable
      if target in coords:
        continue
      pw[target] = np.power(targValue,p)
    pw = xr.Dataset(data_vars=pw,coords=coords)
    return pw

  # TODO: update if necessary
  def _computeSortedWeightsAndPoints(self,arrayIn,pbWeight,percent):
    """
      Method to compute the sorted weights and points
      @ In, arrayIn, list/numpy.array, the array of values from which the percentile needs to be estimated
      @ In, pbWeight, list/numpy.array, the reliability weights that correspond to the values in 'array'
      @ In, percent, float, the percentile that needs to be computed (between 0.01 and 1.0)
      @ Out, sortedWeightsAndPoints, list/numpy.array, with [:,0] as the value of the probability density function at the bin, normalized, and [:,1] is the coresonding edge of the probability density function.
      @ Out, indexL, index of the lower quantile
    """

    idxs                   = np.argsort(np.asarray(list(zip(pbWeight,arrayIn)))[:,1])
    sortedWeightsAndPoints = np.asarray(list(zip(pbWeight[idxs],arrayIn[idxs])))
    weightsCDF             = np.cumsum(sortedWeightsAndPoints[:,0])
    indexL = utils.first(np.asarray(weightsCDF >= percent).nonzero())[0]
    return sortedWeightsAndPoints, indexL

  def __runLocal(self, inputData):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In, inputData, tuple,  (inputDataset, pbWeights), tuple, the dataset of inputs and the corresponding
        variable probability weight
      @ Out, outputSet or outputDict, xarray.Dataset or dict, dataset or dictionary containing the results
    """
    inputDataset, pbWeights = inputData[0], inputData[1]
    #storage dictionary for skipped metrics
    self.skipped = {}
    #construct a dict of required computations
    needed = dict((metric,{'targets':set()}) for metric in self.scalarVals + self.tealVals)
    needed.update(dict((metric,{'targets':set(),'threshold':{}}) for metric in ['sortinoRatio','gainLossRatio']))
    needed.update(dict((metric,{'targets':set(),'threshold':[]}) for metric in ['valueAtRisk', 'expectedShortfall']))
    for metric, params in self.toDo.items():
      if metric in self.vectorVals:
        # vectorVals handled in BasicStatistics, not here
        continue
      for entry in params:
        needed[metric]['targets'].update(entry['targets'])
        if 'threshold' in entry.keys() :
          if metric in ['sortinoRatio','gainLossRatio']:
            threshold = entry['threshold']
            for k in entry['targets']:
              if k in needed[metric]['threshold'].keys():
                needed[metric]['threshold'][k].append(entry['threshold'])
              else:
                needed[metric]['threshold'][k] = []
                needed[metric]['threshold'][k].append(entry['threshold'])
          else:
            thd = entry['threshold']
            if thd not in needed[metric]['threshold']:
              needed[metric]['threshold'].append(thd)
          pass
    # variable                     | needs                  | needed for
    # --------------------------------------------------------------------
    # median needs                 |                        | sortinoRatio, gainLossRatio
    # sigma needs                  | variance               | sharpeRatio
    # variance                     | expectedValue          | sigma
    # expectedValue                |                        | sharpeRatio, sortinoRatio, gainLossRatio
    # sharpeRatio needs            | expectedValue,sigma    |
    # sortinoRatio needs           | expectedValue,median   |
    # gainLossRatio needs          | expectedValue,median   |

    # update needed dictionary when standard errors are requested
    needed['expectedValue']['targets'].update(needed['sigma']['targets'])
    needed['expectedValue']['targets'].update(needed['variance']['targets'])
    needed['expectedValue']['targets'].update(needed['median']['targets'])
    needed['expectedValue']['targets'].update(needed['sharpeRatio']['targets'])
    needed['expectedValue']['targets'].update(needed['sortinoRatio']['targets'])
    needed['expectedValue']['targets'].update(needed['gainLossRatio']['targets'])
    needed['sigma']['targets'].update(needed['sharpeRatio']['targets'])
    needed['variance']['targets'].update(needed['sigma']['targets'])
    needed['median']['targets'].update(needed['sortinoRatio']['targets'])
    needed['median']['targets'].update(needed['gainLossRatio']['targets'])

    for metric, params in needed.items():
      needed[metric]['targets'] = list(params['targets'])

    #
    # BEGIN actual calculations
    #

    calculations = self.calculations

    #################
    # TEAL VALUES   #
    #################
    #
    # SharpeRatio
    #
    metric = 'sharpeRatio'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      meanSet = calculations['expectedValue'][list(needed[metric]['targets'])]
      sigmaSet = calculations['sigma'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      calculations[metric] = meanSet/sigmaSet
    #
    # ValueAtRisk
    #
    metric = 'valueAtRisk'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      threshold = needed[metric]['threshold']
      VaRSet = xr.Dataset()
      relWeight = pbWeights[list(needed[metric]['targets'])]
      targWarn = "" # targets that return negative VaR for warning
      for target in needed[metric]['targets']:
        targWeight = relWeight[target].values
        targDa = dataSet[target]
        VaRList = []
        for thd in threshold:
          if self.pivotParameter in targDa.sizes.keys():
            VaR = [self._computeWeightedPercentile(group.values,targWeight,percent=thd) for label,group in targDa.groupby(self.pivotParameter)]
          else:
            VaR = self._computeWeightedPercentile(targDa.values,targWeight,percent=thd)
          VaRList.append(-VaR)
        if any(np.array(VaRList) < 0):
          targWarn += target + ", "
        if self.pivotParameter in targDa.sizes.keys():
          da = xr.DataArray(VaRList,dims=('threshold',self.pivotParameter),coords={'threshold':threshold,self.pivotParameter:self.pivotValue})
        else:
          da = xr.DataArray(VaRList,dims=('threshold'),coords={'threshold':threshold})
        VaRSet[target] = da
      # write warning for negative VaR values
      if len(targWarn) > 0:
        self.raiseAWarning("At least one negative VaR value calculated for target(s) {}. Negative VaR implies high probability of profit.".format(targWarn[:-2]))
      calculations[metric] = VaRSet
    #
    # ExpectedShortfall
    #
    metric = 'expectedShortfall'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      threshold = needed[metric]['threshold']
      CVaRSet = xr.Dataset()
      relWeight = pbWeights[list(needed[metric]['targets'])]
      for target in needed[metric]['targets']:
        targWeight = relWeight[target].values
        targDa = dataSet[target]
        CVaRList = []
        for thd in threshold:
          if self.pivotParameter in targDa.sizes.keys():
            sortedWeightsAndPoints, indexL = [self._computeSortedWeightsAndPoints(group.values,targWeight,thd) for label,group in targDa.groupby(self.pivotParameter)]
            quantile = [self._computeWeightedPercentile(group.values,targWeight,percent=thd) for label,group in targDa.groupby(self.pivotParameter)]
          else:
            sortedWeightsAndPoints, indexL = self._computeSortedWeightsAndPoints(targDa.values,targWeight,thd)
          quantile = self._computeWeightedPercentile(targDa.values,targWeight,percent=thd)
          lowerPartialE = np.sum(sortedWeightsAndPoints[:indexL,0]*sortedWeightsAndPoints[:indexL,1])
          lowerPartialP = np.sum(sortedWeightsAndPoints[:indexL,0])
          Es = lowerPartialE + quantile*(thd -lowerPartialP)
          CVaRList.append(-Es/(thd))

        if self.pivotParameter in targDa.sizes.keys():
          da = xr.DataArray(CVaRList,dims=('threshold',self.pivotParameter),coords={'threshold':threshold,self.pivotParameter:self.pivotValue})
        else:
          da = xr.DataArray(CVaRList,dims=('threshold'),coords={'threshold':threshold})
        CVaRSet[target] = da
      calculations[metric] = CVaRSet
    #
    # sortinoRatio
    #
    metric = 'sortinoRatio'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      meanSet = calculations['expectedValue'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])]
      orgZeroSet = xr.full_like(meanSet, 0)
      orgMedSet = calculations['median']
      zeroTarget = []
      daZero = xr.Dataset()
      medTarget = []
      daMed = xr.Dataset()
      for entry in self.toDo[metric]:
        if entry['threshold'] == 'zero':
          zeroTarget = entry['targets']
          zeroSet = orgZeroSet[list(zeroTarget)]
          dataSet = inputDataset[list(zeroTarget)]
          lowerPartialVarianceDS = self._computeLowerPartialVariance(dataSet,zeroSet,pbWeight=relWeight,dim=self.sampleTag)
          lpsDS = self.__computePower(0.5,lowerPartialVarianceDS)
          incapableZeroTarget = [x for x in zeroTarget if not lpsDS[x].values != 0]
          for target in incapableZeroTarget:
            needed[metric]['threshold'][target].remove('zero')
          zeroTarget = [x for x in zeroTarget if not lpsDS[x].values == 0]
          if incapableZeroTarget:
            self.raiseAWarning("For metric {} target {}, no lower part data can be found for threshold zero!  Skipping target".format(metric, incapableZeroTarget))
          daZero = meanSet[zeroTarget]/lpsDS[zeroTarget]
          daZero = daZero.assign_coords(threshold ='zero')
          daZero = daZero.expand_dims('threshold')
        elif entry['threshold'] == 'median':
          medTarget = entry['targets']
          medSet = orgMedSet[list(medTarget)]
          dataSet = inputDataset[list(medTarget)]
          lowerPartialVarianceDS = self._computeLowerPartialVariance(dataSet,medSet,pbWeight=relWeight,dim=self.sampleTag)
          lpsDS = self.__computePower(0.5,lowerPartialVarianceDS)
          incapableMedTarget = [x for x in medTarget if not lpsDS[x].values != 0]
          medTarget = [x for x in medTarget if not lpsDS[x].values == 0]
          if incapableMedTarget:
            self.raiseAWarning("For metric {} target {}, no lower part data can be found for threshold median!  Skipping target".format(metric, incapableMedTarget))

          daMed = meanSet[medTarget]/lpsDS[medTarget]
          daMed = daMed.assign_coords(threshold ='median')
          daMed = daMed.expand_dims('threshold')
      calculations[metric] = xr.merge([daMed, daZero])
    #
    # gainLossRatio
    #
    metric = 'gainLossRatio'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      relWeight = pbWeights[list(needed[metric]['targets'])]
      orgMedSet = calculations['median']
      orgZeroSet = xr.full_like(orgMedSet, 0)
      zeroTarget = []
      daZero = xr.Dataset()
      medTarget = []
      daMed = xr.Dataset()

      for entry in self.toDo[metric]:
        if entry['threshold'] == 'zero':
          zeroTarget = entry['targets']
          zeroSet = orgZeroSet[list(zeroTarget)]
          dataSet = inputDataset[list(zeroTarget)]


          higherSet = (dataSet-zeroSet).clip(min=0)
          lowerSet = (zeroSet-dataSet).clip(min=0)
          relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
          higherMeanSet = (higherSet * relWeight).sum(dim = self.sampleTag)
          lowerMeanSet = (lowerSet * relWeight).sum(dim = self.sampleTag)

          incapableZeroTarget = [x for x in zeroTarget if not lowerMeanSet[x].values != 0]
          for target in incapableZeroTarget:
            needed[metric]['threshold'][target].remove('zero')
          zeroTarget = [x for x in zeroTarget if not lowerMeanSet[x].values == 0]
          if incapableZeroTarget:
            self.raiseAWarning("For metric {} target {}, no lower part data can be found for threshold zero!  Skipping target".format(metric,incapableZeroTarget))
          daZero = higherMeanSet[zeroTarget]/lowerMeanSet[zeroTarget]
          daZero = daZero.assign_coords(threshold ='zero')
          daZero = daZero.expand_dims('threshold')

        elif entry['threshold'] == 'median':
          medTarget = entry['targets']
          medSet = orgMedSet[list(medTarget)]
          dataSet = inputDataset[list(medTarget)]


          higherSet = (dataSet-medSet).clip(min=0)
          lowerSet = (medSet-dataSet).clip(min=0)
          relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
          higherMeanSet = (higherSet * relWeight).sum(dim = self.sampleTag)
          lowerMeanSet = (lowerSet * relWeight).sum(dim = self.sampleTag)


          incapableMedTarget = [x for x in medTarget if not lowerMeanSet[x].values != 0]
          medTarget = [x for x in medTarget if not lowerMeanSet[x].values == 0]
          if incapableMedTarget:
            self.raiseAWarning("For metric {} target {}, lower part mean is zero for threshold median!  Skipping target".format(metric, incapableMedTarget))

          daMed = higherMeanSet[medTarget]/lowerMeanSet[medTarget]
          daMed = daMed.assign_coords(threshold ='median')
          daMed = daMed.expand_dims('threshold')
      calculations[metric] = xr.merge([daMed, daZero])


    for metric, ds in calculations.items():
      if metric in self.scalarVals + self.tealVals +['equivalentSamples'] and metric !='samples':
        calculations[metric] = ds.to_array().rename({'variable':'targets'})
    outputSet = xr.Dataset(data_vars=calculations)
    outputDict = {}
    for metric, requestList  in self.toDo.items():
      for targetDict in requestList:
        prefix = targetDict['prefix'].strip()
        for target in targetDict['targets']:
          if metric in self.tealVals:
            varName = prefix + '_' + target
            if 'threshold' in targetDict.keys():
              try:
                outputDict[varName] = np.atleast_1d(outputSet[metric].sel(**{'targets':target,'threshold':targetDict['threshold']}))
              except KeyError:
                outputDict[varName] = np.nan
            else:
              outputDict[varName] = np.atleast_1d(outputSet[metric].sel(**{'targets':target}))
            steMetric = metric + '_ste'
          else:
            #check if it was skipped for some reason
            skip = self.skipped.get(metric, None)
            if skip is not None:
              self.raiseADebug('Metric',metric,'was skipped for parameters',targetDict,'!  See warnings for details.  Ignoring...')
              continue
    if self.pivotParameter in outputSet.sizes.keys():
      outputDict[self.pivotParameter] = np.atleast_1d(self.pivotValue)
    return outputDict

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputSet, xarray.Dataset or dictionary, dataset or dictionary containing the results
    """
    # get metrics from BasicStatistics
    outputSetBasicStatistics = BasicStatistics.run(self, inputIn)

    # get metrics from EconomicRatio
    inputData = self.inputToInternal(inputIn)
    outputSet = self.__runLocal(inputData)

    # combine results
    if isinstance(outputSet, dict):
      # returned dictionary
      outputSet.update(outputSetBasicStatistics)
    else:
      # returned xarray.Dataset
      outputSet = xr.merge([outputSet, outputSetBasicStatistics])

    return outputSet
