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
  scalarVals =   BasicStatistics.scalarVals
  vectorVals =   BasicStatistics.vectorVals

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
    inputSpecification = super(EconomicRatio, cls).getInputSpecification()

    for scalar in cls.scalarVals:
      scalarSpecification = InputData.parameterInputFactory(scalar, contentType=InputTypes.StringListType)
      scalarSpecification.addParam("prefix", InputTypes.StringType)
      inputSpecification.addSub(scalarSpecification)

    for teal in cls.tealVals:
      tealSpecification = InputData.parameterInputFactory(teal, contentType=InputTypes.StringListType)
      if teal in['sortinoRatio','gainLossRatio']:
        tealSpecification.addParam("threshold", InputTypes.StringType)
      elif teal in['expectedShortfall','valueAtRisk']:
        tealSpecification.addParam("threshold", InputTypes.FloatType)
      tealSpecification.addParam("prefix", InputTypes.StringType)
      inputSpecification.addSub(tealSpecification)

    pivotParameterInput = InputData.parameterInputFactory('pivotParameter', contentType=InputTypes.StringType)
    inputSpecification.addSub(pivotParameterInput)


    return inputSpecification

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, (inputDataset, pbWeights), tuple, the dataset of inputs and the corresponding variable probability weight
    """
    # The EconomicRatio postprocessor only accept DataObjects
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0:
      self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")
    pbWeights = None

    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'EconomicRatio postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)
    # extract all required data from input DataObjects, an input dataset is constructed
    dataSet = currentInput.asDataset()
    inputDataset = dataSet[self.parameters['targets']]
    self.sampleTag = currentInput.sampleTag

    if currentInput.type == 'HistorySet':
      dims = inputDataset.sizes.keys()
      if self.pivotParameter is None:
        if len(dims) > 1:
          self.raiseAnError(IOError, self, 'Time-dependent statistics is requested (HistorySet) but no pivotParameter \
                got inputted!')
      elif self.pivotParameter not in dims:
        self.raiseAnError(IOError, self, 'Pivot parameter', self.pivotParameter, 'is not the associated index for \
                requested variables', ','.join(self.parameters['targets']))
      else:
        if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
          self.raiseAnError(IOError, "The data provided by the data objects", currentInput.name, "is not synchronized!")
        self.pivotValue = inputDataset[self.pivotParameter].values
        if self.pivotValue.size != len(inputDataset.groupby(self.pivotParameter)):
          msg = "Duplicated values were identified in pivot parameter, please use the 'HistorySetSync'" + \
          " PostProcessor to syncronize your data before running 'EconomicRatio' PostProcessor."
          self.raiseAnError(IOError, msg)
    # extract all required meta data
    metaVars = currentInput.getVars('meta')
    self.pbPresent = True if 'ProbabilityWeight' in metaVars else False

    if self.pbPresent:
      pbWeights = xr.Dataset()
      self.realizationWeight = dataSet[['ProbabilityWeight']]/dataSet[['ProbabilityWeight']].sum()
      for target in self.parameters['targets']:
        pbName = 'ProbabilityWeight-' + target
        if pbName in metaVars:
          pbWeights[target] = dataSet[pbName]/dataSet[pbName].sum()
        elif self.pbPresent:
          pbWeights[target] = self.realizationWeight['ProbabilityWeight']
    else:
      self.raiseAWarning('EconomicRatio postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
    return inputDataset, pbWeights

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the EconomicRatio pp. In here the working dir is
      grepped.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    #construct a list of all the parameters that have requested values into self.allUsedParams
    self.allUsedParams = set()
    for metricName in self.scalarVals + self.tealVals:
      if metricName in self.toDo.keys():
        for entry in self.toDo[metricName]:
          self.allUsedParams.update(entry['targets'])

    #for backward compatibility, compile the full list of parameters used in Economic Ratio calculations
    self.parameters['targets'] = list(self.allUsedParams)
    PostProcessorInterface.initialize(self, runInfo, inputs, initDict)
    inputObj = inputs[-1] if type(inputs) == list else inputs
    inputMetaKeys = []
    outputMetaKeys = []
    metaParams = {}
    if len(outputMetaKeys) > 0:
      metaParams = {key:[self.pivotParameter] for key in outputMetaKeys}

    metaKeys = inputMetaKeys + outputMetaKeys
    self.addMetaKeys(metaKeys,metaParams)

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
    self.toDo = {}
    for child in paramInput.subparts:
      tag = child.getName()
      if tag in self.scalarVals + self.tealVals:
        if 'prefix' not in child.parameterValues:
          self.raiseAnError(IOError, "No prefix is provided for node: ", tag)
        #get the prefix
        prefix = child.parameterValues['prefix']

      if tag in self.tealVals:
        if tag in ['sortinoRatio', 'gainLossRatio']:
          #get targets
          targets = set(child.value)
          if tag not in self.toDo.keys():
            self.toDo[tag] = [] # list of {'targets':(), 'prefix':str, 'threshold':str}
          if 'threshold' not in child.parameterValues:
            threshold = 'zero'
          else:
            threshold = child.parameterValues['threshold'].lower()
            if threshold not in ['zero','median']:
              self.raiseAWarning('Unrecognized threshold in {}, prefix \'{}\' use zero instead!'.format(tag, prefix))
              threshold = 'zero'

          if 'expectedValue' not in self.toDo.keys():
            self.toDo['expectedValue'] = []
          if 'median' not in self.toDo.keys():
            self.toDo['median'] = []
          self.toDo['expectedValue'].append({'targets':set(child.value),
                            'prefix':'BSMean'})
          self.toDo['median'].append({'targets':set(child.value),
                            'prefix':'BSMED'})
          self.toDo[tag].append({'targets':set(targets),
                                'prefix':prefix,
                                'threshold':threshold})

        elif tag in ['expectedShortfall', 'valueAtRisk']:
          #get targets
          targets = set(child.value)
          if tag not in self.toDo.keys():
            self.toDo[tag] = [] # list of {'targets':(), 'prefix':str, 'threshold':str}
          if 'threshold' not in child.parameterValues:
            threshold = 0.05
          else:
            threshold = child.parameterValues['threshold']
            if threshold >1 or threshold <0:
              self.raiseAnError('Threshold in {}, prefix \'{}\' out of range, please use a float in range (0, 1)!'.format(tag, prefix))

          self.toDo[tag].append({'targets':set(targets),
                                'prefix':prefix,
                                'threshold':threshold})
        else:
          if tag not in self.toDo.keys():
            self.toDo[tag] = [] # list of {'targets':(), 'prefix':str}
          if 'expectedValue' not in self.toDo.keys():
            self.toDo['expectedValue'] = []
          if 'sigma' not in self.toDo.keys():
            self.toDo['sigma'] = []
          self.toDo['expectedValue'].append({'targets':set(child.value),
                               'prefix':'BSMean'})
          self.toDo['sigma'].append({'targets':set(child.value),
                               'prefix':'BSSigma'})
          self.toDo[tag].append({'targets':set(child.value),
                               'prefix':prefix})
      elif tag in self.scalarVals:
        if tag not in self.toDo.keys():
          self.toDo[tag] = [] # list of {'targets':(), 'prefix':str}
        self.toDo[tag].append({'targets':set(child.value),
                               'prefix':prefix})
      elif tag == "pivotParameter":
        self.pivotParameter = child.value

      else:
        self.raiseAWarning('Unrecognized node in EconomicRatio "',tag,'" has been ignored!')

    assert (len(self.toDo)>0), self.raiseAnError(IOError, 'EconomicRatio needs parameters to work on! Please check input for PP: ' + self.name)

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

  def _computeWeightedPercentile(self,arrayIn,pbWeight,percent=0.5):
    """
      Method to compute the weighted percentile in a array of data
      @ In, arrayIn, list/numpy.array, the array of values from which the percentile needs to be estimated
      @ In, pbWeight, list/numpy.array, the reliability weights that correspond to the values in 'array'
      @ In, percent, float, the percentile that needs to be computed (between 0.01 and 1.0)
      @ Out, result, float, the percentile
    """

    idxs                   = np.argsort(np.asarray(list(zip(pbWeight,arrayIn)))[:,1])
    # Inserting [0.0,arrayIn[idxs[0]]] is needed when few samples are generated and
    # a percentile that is < that the first pb weight is requested. Otherwise the median
    # is returned.
    sortedWeightsAndPoints = np.insert(np.asarray(list(zip(pbWeight[idxs],arrayIn[idxs]))),0,[0.0,arrayIn[idxs[0]]],axis=0)
    weightsCDF             = np.cumsum(sortedWeightsAndPoints[:,0])
    # This step returns the index of the array which is < than the percentile, because
    # the insertion create another entry, this index should shift to the bigger side
    indexL = utils.first(np.asarray(weightsCDF >= percent).nonzero())[0]
    # This step returns the indices (list of index) of the array which is > than the percentile
    indexH = utils.first(np.asarray(weightsCDF > percent).nonzero())
    try:
      # if the indices exists that means the desired percentile lies between two data points
      # with index as indexL and indexH[0]. Calculate the midpoint of these two points
      result = 0.5*(sortedWeightsAndPoints[indexL,1]+sortedWeightsAndPoints[indexH[0],1])
    except IndexError:
      result = sortedWeightsAndPoints[indexL,1]
    return result

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
      for target in needed[metric]['targets']:
        targWeight = relWeight[target].values
        targDa = dataSet[target]
        VaRList = []
        for thd in threshold:
          if self.pivotParameter in targDa.sizes.keys():
            VaR = [self._computeWeightedPercentile(group.values,targWeight,percent=thd) for label,group in targDa.groupby(self.pivotParameter)]
          else:
            VaR = self._computeWeightedPercentile(targDa.values,targWeight,percent=thd)
          VaRList.append(abs(VaR))
        if self.pivotParameter in targDa.sizes.keys():
          da = xr.DataArray(VaRList,dims=('threshold',self.pivotParameter),coords={'threshold':threshold,self.pivotParameter:self.pivotValue})
        else:
          da = xr.DataArray(VaRList,dims=('threshold'),coords={'threshold':threshold})
        VaRSet[target] = da
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
      daMed = xr.Dataset
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
            self.raiseAWarning("For metric {} target {}, lower part mean is zero for threshold median!  Skipping target".format(matric, incapableMedTarget))

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
          if metric in self.scalarVals:
            varName = prefix + '_' + target
            outputDict[varName] = np.atleast_1d(outputSet[metric].sel(**{'targets':target}))
            steMetric = metric + '_ste'
          elif metric in self.tealVals:
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
    BasicStatistics.run(self,inputIn)
    inputData = self.inputToInternal(inputIn)
    outputSet = self.__runLocal(inputData)
    return outputSet

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputRealization = evaluation[1]
    if output.type in ['PointSet','HistorySet']:
      self.raiseADebug('Dumping output in data object named ' + output.name)
      output.addRealization(outputRealization)
    elif output.type in ['DataSet']:
      self.raiseADebug('Dumping output in DataSet named ' + output.name)
      output.load(outputRealization,style='dataset')
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')
