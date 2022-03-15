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
Created on July 10, 2013

@author: alfoa
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import math
import copy
from scipy import integrate
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import utils
from ...utils import mathUtils
from ...utils import InputData, InputTypes
from ... import Files
from ... import Distributions
#Internal Modules End--------------------------------------------------------------------------------

# global number of integration points
integrationSegments = int(1e5)


def _getGraphs(functions, fZStats = False):
  """
    Returns the graphs of the functions.
    The functions are a list of (dataStats, cdf_function, pdf_function,name)
    It returns a dictionary with the graphs and other statistics calculated.
    @ In, functions, list, list of functions (data_stats_dict, cdf_function, pdf_function,name)
    @ In, fZStats, bool, optional, true if the F(z) (cdf) needs to be computed
    @ Out, retDict, dict, the return dictionary
  """
  retDict = {}
  dataStats = [x[0] for x in functions]
  means = [x["mean"] for x in dataStats]
  stdDevs = [x["stdev"] for x in dataStats]
  cdfs = [x[1] for x in functions]
  pdfs = [x[2] for x in functions]
  names = [x[3] for x in functions]
  low = min([m - 3.0*s for m,s in zip(means,stdDevs)])
  high = max([m + 3.0*s for m,s in zip(means,stdDevs)])
  lowLow = min([m - 5.0*s for m,s in zip(means,stdDevs)])
  highHigh = max([m + 5.0*s for m,s in zip(means,stdDevs)])
  minBinSize = min([x["minBinSize"] for x in dataStats])
  n = int(math.ceil((high-low)/minBinSize))
  interval = (high - low)/n

  #Print the cdfs and pdfs of the data to be compared.
  origCdfAndPdfArray = []
  origCdfAndPdfArray.append(["x"])
  for name in names:
    origCdfAndPdfArray.append([name+'_cdf'])
    origCdfAndPdfArray.append([name+'_pdf'])

  for i in range(n):
    x = low+interval*i
    origCdfAndPdfArray[0].append(x)
    k = 1
    for stats, cdf, pdf, name in functions:
      origCdfAndPdfArray[k].append(cdf(x))
      origCdfAndPdfArray[k+1].append(pdf(x))
      k += 2
  retDict["cdf_and_pdf_arrays"] = origCdfAndPdfArray

  if len(means) < 2:
    return

  cdfAreaDifference = integrate.quad(lambda x:abs(cdfs[1](x)-cdfs[0](x)),lowLow,highHigh,limit=1000)[0]

  #print a bunch of comparison statistics
  pdfCommonArea = integrate.quad(lambda x:min(pdfs[0](x),pdfs[1](x)),
                            lowLow,highHigh,limit=1000)[0]
  for i in range(len(pdfs)):
    pdfArea = integrate.quad(pdfs[i],lowLow,highHigh,limit=1000)[0]
    retDict['pdf_area_'+names[i]] = pdfArea
    dataStats[i]["pdf_area"] = pdfArea
  retDict['cdf_area_difference'] = cdfAreaDifference
  retDict['pdf_common_area'] = pdfCommonArea
  dataStats[0]["cdf_area_difference"] = cdfAreaDifference
  dataStats[0]["pdf_common_area"] = pdfCommonArea
  if fZStats:
    def fZ(z):
      """
        Compute f(z) with a quad rule
        @ In, z, float, the coordinate
        @ Out, fZ, the f(z)
      """
      return integrate.quad(lambda x: pdfs[0](x)*pdfs[1](x-z), lowLow, highHigh,limit=1000)[0]

    midZ = means[0]-means[1]
    lowZ = midZ - 3.0*max(stdDevs[0],stdDevs[1])
    highZ = midZ + 3.0*max(stdDevs[0],stdDevs[1])
    #print the difference function table.
    fZTable = [["z"],["f_z(z)"]]
    zN = 20
    intervalZ = (highZ - lowZ)/zN
    for i in range(zN):
      z = lowZ + intervalZ*i
      fZTable[0].append(z)
      fZTable[1].append(fZ(z))
    retDict["f_z_table"] = fZTable
    sumFunctionDiff = integrate.quad(fZ, lowZ, highZ,limit=1000)[0]
    firstMomentFunctionDiff = integrate.quad(lambda x:x*fZ(x), lowZ, highZ,limit=1000)[0]
    varianceFunctionDiff = integrate.quad(lambda x:((x-firstMomentFunctionDiff)**2)*fZ(x),lowZ,highZ,limit=1000)[0]
    retDict['sum_function_diff'] = sumFunctionDiff
    retDict['first_moment_function_diff'] = firstMomentFunctionDiff
    retDict['variance_function_diff'] = varianceFunctionDiff
  return retDict

def __processData(data, methodInfo):
  """
    Method to process the computed data
    @ In, data, np.array, the data to process
    @ In, methodInfo, dict, the info about which processing method needs to be used
    @ Out, ret, dict, the processed data including the counts of the bins
  """
  ret = {}
  if hasattr(data,'tolist'):
    sortedData = data.tolist()
  else:
    sortedData = list(data)
  sortedData.sort()
  low = sortedData[0]
  high = sortedData[-1]
  dataRange = high - low
  ret['low'] = low
  ret['high'] = high
  if not 'binMethod' in methodInfo:
    numBins = methodInfo.get("numBins", 10)
  else:
    binMethod = methodInfo['binMethod']
    dataN = len(sortedData)
    if binMethod == 'square-root':
      numBins = int(math.ceil(math.sqrt(dataN)))
    elif binMethod == 'sturges':
      numBins = int(math.ceil(mathUtils.log2(dataN) + 1))
    else:
      self.raiseADebug("Unknown binMethod " + binMethod, 'ExceptedError')
      numBins = 5
  ret['numBins'] = numBins
  kind = methodInfo.get("kind", "uniformBins")
  if kind == "uniformBins":
    bins = [low + x * dataRange / numBins for x in range(1, numBins)]
    ret['minBinSize'] = dataRange / numBins
  elif kind == "equalProbability":
    stride = len(sortedData) // numBins
    bins = [sortedData[x] for x in range(stride - 1, len(sortedData) - stride + 1, stride)]
    if len(bins) > 1:
      ret['minBinSize'] = min(map(lambda x, y: x - y, bins[1:], bins[:-1]))
    else:
      ret['minBinSize'] = dataRange
  counts = mathUtils.countBins(sortedData, bins)
  ret['bins'] = bins
  ret['counts'] = counts
  ret.update(mathUtils.calculateStats(sortedData))
  skewness = ret["skewness"]
  delta_func = lambda skewness: math.sqrt((math.pi / 2.0) * (abs(skewness) ** (2.0 / 3.0)) /
                                (abs(skewness) ** (2.0 / 3.0) + ((4.0 - math.pi) / 2.0) ** (2.0 / 3.0)))
  # see https://en.wikipedia.org/wiki/Skew_normal_distribution (Estimation)
  if skewness > 0.9952717:
    print("The population skewness > 0.9952717 => alpha, omega and xi parameters are not in convergence!")
  delta_alpha = delta_func(min(0.9952717,skewness))
  delta_alpha = math.copysign(delta_alpha, skewness)
  alpha       = delta_alpha / math.sqrt(1.0 - delta_alpha ** 2)
  variance = ret["sampleVariance"]
  omega = math.sqrt(variance / (1.0 - 2 * delta_alpha ** 2 / math.pi))
  mean = ret['mean']
  xi = mean - omega * delta_alpha * math.sqrt(2.0 / math.pi)
  ret['alpha'] = alpha
  ret['omega'] = omega
  ret['xi'] = xi
  return ret

def _getPDFandCDFfromData(dataName, data, csv, methodInfo, interpolation,
                         generateCSV):
  """
    This method is used to convert some data into a PDF and CDF function.
    Note, it might be better done by scipy.stats.gaussian_kde
    @ In, dataName, str, The name of the data.
    @ In, data, np.array, one dimentional array of the data to process
    @ In, csv, File, file to write out information on data.
    @ In, methodInfo, dict, the info about which processing method needs to be used
    @ In, interpolation, str, "linear" or "quadratic", depending on which interpolation is used
    @ In, generateCSV, bool, True if the csv should be written
    @ Out, (dataStats, cdfFunc, pdfFunc), tuple, dataStats is dictionary with things like "mean" and "stdev", cdfFunction is a function that returns the CDF value and pdfFunc is a function that returns the PDF value.
  """
  #Convert data to pdf and cdf.
  dataStats = __processData( data, methodInfo)
  dataKeys = set(dataStats.keys())
  counts = dataStats['counts']
  bins = dataStats['bins']
  countSum = sum(counts)
  binBoundaries = [dataStats['low']] + bins + [dataStats['high']]
  if generateCSV:
    utils.printCsv(csv, '"' + dataName + '"')
    utils.printCsv(csv, '"numBins"', dataStats['numBins'])
    utils.printCsv(csv, '"binBoundary"', '"binMidpoint"', '"binCount"', '"normalizedBinCount"', '"f_prime"', '"cdf"')
  cdf = [0.0] * len(counts)
  midpoints = [0.0] * len(counts)
  cdfSum = 0.0
  for i in range(len(counts)):
    f0 = counts[i] / countSum
    cdfSum += f0
    cdf[i] = cdfSum
    midpoints[i] = (binBoundaries[i] + binBoundaries[i + 1]) / 2.0
  cdfFunc = mathUtils.createInterp(midpoints, cdf, 0.0, 1.0, interpolation)
  fPrimeData = [0.0] * len(counts)
  for i in range(len(counts)):
    h = binBoundaries[i + 1] - binBoundaries[i]
    nCount = counts[i] / countSum  # normalized count
    f0 = cdf[i]
    if i + 1 < len(counts):
      f1 = cdf[i + 1]
    else:
      f1 = 1.0
    if i + 2 < len(counts):
      f2 = cdf[i + 2]
    else:
      f2 = 1.0
    if interpolation == 'linear':
      fPrime = (f1 - f0) / h
    else:
      fPrime = (-1.5 * f0 + 2.0 * f1 + -0.5 * f2) / h
    fPrimeData[i] = fPrime
    if generateCSV:
      utils.printCsv(csv, binBoundaries[i + 1], midpoints[i], counts[i], nCount, fPrime, cdf[i])
  pdfFunc = mathUtils.createInterp(midpoints, fPrimeData, 0.0, 0.0, interpolation)
  dataKeys -= set({'numBins', 'counts', 'bins'})
  if generateCSV:
    for key in dataKeys:
      utils.printCsv(csv, '"' + key + '"', dataStats[key])
  return dataStats, cdfFunc, pdfFunc


class ComparisonStatistics(PostProcessorInterface):
  """
    ComparisonStatistics is to calculate statistics that compare
    two different codes or code to experimental data.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(ComparisonStatistics, cls).getInputSpecification()
    KindInputEnumType = InputTypes.makeEnumType("kind", "kindType", ["uniformBins", "equalProbability"])
    KindInput = InputData.parameterInputFactory("kind", contentType=KindInputEnumType)
    KindInput.addParam("numBins", InputTypes.IntegerType, False)
    KindInput.addParam("binMethod", InputTypes.StringType, False)
    inputSpecification.addSub(KindInput)

    ## FIXME: Is this class necessary?
    class CSCompareInput(InputData.ParameterInput):
      """
        class for reading in the compare block in comparison statistics
      """

    CSCompareInput.createClass("compare", False)
    CSDataInput = InputData.parameterInputFactory("data", contentType=InputTypes.StringType)
    CSCompareInput.addSub(CSDataInput)
    CSReferenceInput = InputData.parameterInputFactory("reference")
    CSReferenceInput.addParam("name", InputTypes.StringType, True)
    CSCompareInput.addSub(CSReferenceInput)
    inputSpecification.addSub(CSCompareInput)

    FZInput = InputData.parameterInputFactory("fz", contentType=InputTypes.StringType) #bool
    inputSpecification.addSub(FZInput)

    CSInterpolationEnumType = InputTypes.makeEnumType("csinterpolation","csinterpolationType",["linear","quadratic"])
    CSInterpolationInput = InputData.parameterInputFactory("interpolation",contentType=CSInterpolationEnumType)
    inputSpecification.addSub(CSInterpolationInput)

    DistributionInput = InputData.parameterInputFactory("Distribution", contentType=InputTypes.StringType)
    DistributionInput.addParam("class", InputTypes.StringType)
    DistributionInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(DistributionInput)

    return inputSpecification

  class CompareGroup:
    """
      Class aimed to compare two group of data
    """
    def __init__(self):
      """
        Constructor
        @ In, None
        @ Out, None
      """
      self.dataPulls = []
      self.referenceData = {}

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dataDict = {}  # Dictionary of all the input data, keyed by the name
    self.compareGroups = []  # List of each of the groups that will be compared
    # self.dataPulls = [] #List of data references that will be used
    # self.referenceData = [] #List of reference (experimental) data
    self.methodInfo = {}  # Information on what stuff to do.
    self.fZStats = False
    self.interpolation = "quadratic"
    # assembler objects to be requested
    self.addAssemblerObject('Distribution', InputData.Quantity.zero_to_infinity)

  def inputToInternal(self, currentInput):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputToInternal, list, the resulting converted object
    """
    return [(currentInput)]

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the ComparisonStatistics pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for outer in paramInput.subparts:
      if outer.getName() == 'compare':
        compareGroup = ComparisonStatistics.CompareGroup()
        for child in outer.subparts:
          if child.getName() == 'data':
            dataName = child.value
            splitName = dataName.split("|")
            name, kind = splitName[:2]
            rest = splitName[2:]
            compareGroup.dataPulls.append([name, kind, rest])
          elif child.getName() == 'reference':
            # This has name=distribution
            compareGroup.referenceData = dict(child.parameterValues)
            if "name" not in compareGroup.referenceData:
              self.raiseAnError(IOError, 'Did not find name in reference block')

        self.compareGroups.append(compareGroup)
      if outer.getName() == 'kind':
        self.methodInfo['kind'] = outer.value
        if 'numBins' in outer.parameterValues:
          self.methodInfo['numBins'] = outer.parameterValues['numBins']
        if 'binMethod' in outer.parameterValues:
          self.methodInfo['binMethod'] = outer.parameterValues['binMethod'].lower()
      if outer.getName() == 'fz':
        self.fZStats = utils.stringIsTrue(outer.value.lower())
      if outer.getName() == 'interpolation':
        interpolation = outer.value.lower()
        if interpolation == 'linear':
          self.interpolation = 'linear'
        elif interpolation == 'quadratic':
          self.interpolation = 'quadratic'
        else:
          self.raiseADebug('unexpected interpolation method ' + interpolation)
          self.interpolation = interpolation

  def run(self, input):  # inObj,workingDir=None):
    """
      This method executes the postprocessor action. In this case, it just returns the inputs
      @ In,  input, object, object contained the data to process. (inputToInternal output)
      @ Out, dataDict, dict, Dictionary containing the inputs
    """
    dataDict = {}
    for aInput in input:
      dataDict[aInput.name] = aInput
    return dataDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    self.raiseADebug("finishedJob: " + str(finishedJob) + ", output " + str(output))
    evaluation = finishedJob.getEvaluation()

    outputDictionary = evaluation[1]
    self.dataDict.update(outputDictionary)

    dataToProcess = []
    for compareGroup in self.compareGroups:
      dataPulls = compareGroup.dataPulls
      reference = compareGroup.referenceData
      foundDataObjects = []
      for name, kind, rest in dataPulls:
        dataSet = self.dataDict[name].asDataset()
        if len(rest) == 1:
          foundDataObjects.append(copy.copy(dataSet[rest[0]].values))
      dataToProcess.append((dataPulls, foundDataObjects, reference))
    if not isinstance(output,Files.File):
      self.raiseAnError(IOError, 'unsupported type ' + str(type(output)))
    for dataPulls, datas, reference in dataToProcess:
      graphData = []
      if "name" in reference:
        distributionName = reference["name"]
        distribution = self.retrieveObjectFromAssemblerDict('Distribution', distributionName)
        if distribution is None:
          self.raiseAnError(IOError, 'Did not find Distribution with name ' + distributionName)
        refDataStats = {"mean":distribution.untruncatedMean(),
                        "stdev":distribution.untruncatedStdDev()}
        refDataStats["minBinSize"] = refDataStats["stdev"] / 2.0
        refPdf = lambda x:distribution.pdf(x)
        refCdf = lambda x:distribution.cdf(x)
        graphData.append((refDataStats, refCdf, refPdf, "ref_" + distributionName))
      for dataPull, data in zip(dataPulls, datas):
        dataStats, cdfFunc, pdfFunc = _getPDFandCDFfromData(str(dataPull),
                                                            data,
                                                            output,
                                                            self.methodInfo,
                                                            self.interpolation,
                                                            True)
        self.raiseADebug("dataStats: " + str(dataStats))
        graphData.append((dataStats, cdfFunc, pdfFunc, str(dataPull)))
      graphDataDict = _getGraphs(graphData, self.fZStats)
      for key in graphDataDict:
        value = graphDataDict[key]
        if type(value).__name__ == 'list':
          utils.printCsv(output, *(['"' + l[0] + '"' for l in value]))
          for i in range(1, len(value[0])):
            utils.printCsv(output, *([l[i] for l in value]))
        else:
          utils.printCsv(output, '"' + key + '"', value)
      for i in range(len(graphData)):
        dataStat = graphData[i][0]
        def delist(l):
          """
            Method to create a string out of a list l
            @ In, l, list, the list to be 'stringed' out
            @ Out, delist, string, the string representing the list
          """
          if type(l).__name__ == 'list':
            return '_'.join([delist(x) for x in l])
          else:
            return str(l)
        newFileName = output.getBase() + "_" + delist(dataPulls) + "_" + str(i) + ".csv"
        if type(dataStat).__name__ != 'dict':
          assert(False)
          continue
        dataPairs = []
        for key in sorted(dataStat.keys()):
          value = dataStat[key]
          if np.isscalar(value):
            dataPairs.append((key, value))
        extraCsv = Files.factory.returnInstance('CSV')
        extraCsv.initialize(newFileName)
        extraCsv.open("w")
        extraCsv.write(",".join(['"' + str(x[0]) + '"' for x in dataPairs]))
        extraCsv.write("\n")
        extraCsv.write(",".join([str(x[1]) for x in dataPairs]))
        extraCsv.write("\n")
        extraCsv.close()
      utils.printCsv(output)
