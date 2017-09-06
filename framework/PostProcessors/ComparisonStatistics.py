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
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import numpy as np
import math
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import mathUtils
from utils import InputData
import Files
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class ComparisonStatistics(PostProcessor):
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
    KindInputEnumType = InputData.makeEnumType("kind","kindType",["uniformBins","equalProbability"])
    KindInput = InputData.parameterInputFactory("kind", contentType=KindInputEnumType)
    KindInput.addParam("numBins",InputData.IntegerType, False)
    KindInput.addParam("binMethod", InputData.StringType, False)
    inputSpecification.addSub(KindInput)

    ## FIXME: Is this class necessary?
    class CSCompareInput(InputData.ParameterInput):
      """
        class for reading in the compare block in comparison statistics
      """

    CSCompareInput.createClass("compare", False)
    CSDataInput = InputData.parameterInputFactory("data", contentType=InputData.StringType)
    CSCompareInput.addSub(CSDataInput)
    CSReferenceInput = InputData.parameterInputFactory("reference")
    CSReferenceInput.addParam("name", InputData.StringType, True)
    CSCompareInput.addSub(CSReferenceInput)
    inputSpecification.addSub(CSCompareInput)

    FZInput = InputData.parameterInputFactory("fz", contentType=InputData.StringType) #bool
    inputSpecification.addSub(FZInput)

    CSInterpolationEnumType = InputData.makeEnumType("csinterpolation","csinterpolationType",["linear","quadratic"])
    CSInterpolationInput = InputData.parameterInputFactory("interpolation",contentType=CSInterpolationEnumType)
    inputSpecification.addSub(CSInterpolationInput)

    DistributionInput = InputData.parameterInputFactory("Distribution", contentType=InputData.StringType)
    DistributionInput.addParam("class", InputData.StringType)
    DistributionInput.addParam("type", InputData.StringType)
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

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.dataDict = {}  # Dictionary of all the input data, keyed by the name
    self.compareGroups = []  # List of each of the groups that will be compared
    # self.dataPulls = [] #List of data references that will be used
    # self.referenceData = [] #List of reference (experimental) data
    self.methodInfo = {}  # Information on what stuff to do.
    self.fZStats = False
    self.interpolation = "quadratic"
    self.requiredAssObject = (True, (['Distribution'], ['-n']))
    self.distributions = {}

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
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = ComparisonStatistics.getInputSpecification()()
    paramInput.parseNode(xmlNode)
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
        self.fZStats = (outer.value.lower() in utils.stringsThatMeanTrue())
      if outer.getName() == 'interpolation':
        interpolation = outer.value.lower()
        if interpolation == 'linear':
          self.interpolation = 'linear'
        elif interpolation == 'quadratic':
          self.interpolation = 'quadratic'
        else:
          self.raiseADebug('unexpected interpolation method ' + interpolation)
          self.interpolation = interpolation


  def _localGenerateAssembler(self, initDict):
    """
      This method  is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    self.distributions = initDict.get('Distributions', {})

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
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")

    outputDictionary = evaluation[1]
    self.dataDict.update(outputDictionary)

    dataToProcess = []
    for compareGroup in self.compareGroups:
      dataPulls = compareGroup.dataPulls
      reference = compareGroup.referenceData
      foundDataObjects = []
      for name, kind, rest in dataPulls:
        data = self.dataDict[name].getParametersValues(kind)
        if len(rest) == 1:
          foundDataObjects.append(data[rest[0]])
      dataToProcess.append((dataPulls, foundDataObjects, reference))
    generateCSV = False
    generatePointSet = False
    if isinstance(output,Files.File):
      generateCSV = True
    elif output.type == 'PointSet':
      generatePointSet = True
    else:
      self.raiseAnError(IOError, 'unsupported type ' + str(type(output)))
    if generateCSV:
      csv = output
    for dataPulls, datas, reference in dataToProcess:
      graphData = []
      if "name" in reference:
        distributionName = reference["name"]
        if not distributionName in self.distributions:
          self.raiseAnError(IOError, 'Did not find ' + distributionName +
                             ' in ' + str(self.distributions.keys()))
        else:
          distribution = self.distributions[distributionName]
        refDataStats = {"mean":distribution.untruncatedMean(),
                        "stdev":distribution.untruncatedStdDev()}
        refDataStats["minBinSize"] = refDataStats["stdev"] / 2.0
        refPdf = lambda x:distribution.pdf(x)
        refCdf = lambda x:distribution.cdf(x)
        graphData.append((refDataStats, refCdf, refPdf, "ref_" + distributionName))
      for dataPull, data in zip(dataPulls, datas):
        #Convert data to pdf and cdf.
        dataStats = self.__processData( data, self.methodInfo)
        dataKeys = set(dataStats.keys())
        counts = dataStats['counts']
        bins = dataStats['bins']
        countSum = sum(counts)
        binBoundaries = [dataStats['low']] + bins + [dataStats['high']]
        if generateCSV:
          utils.printCsv(csv, '"' + str(dataPull) + '"')
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
        cdfFunc = mathUtils.createInterp(midpoints, cdf, 0.0, 1.0, self.interpolation)
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
          if self.interpolation == 'linear':
            fPrime = (f1 - f0) / h
          else:
            fPrime = (-1.5 * f0 + 2.0 * f1 + -0.5 * f2) / h
          fPrimeData[i] = fPrime
          if generateCSV:
            utils.printCsv(csv, binBoundaries[i + 1], midpoints[i], counts[i], nCount, fPrime, cdf[i])
        pdfFunc = mathUtils.createInterp(midpoints, fPrimeData, 0.0, 0.0, self.interpolation)
        dataKeys -= set({'numBins', 'counts', 'bins'})
        if generateCSV:
          for key in dataKeys:
            utils.printCsv(csv, '"' + key + '"', dataStats[key])
        self.raiseADebug("dataStats: " + str(dataStats))
        graphData.append((dataStats, cdfFunc, pdfFunc, str(dataPull)))
      graphDataDict = mathUtils.getGraphs(graphData, self.fZStats)
      if generateCSV:
        for key in graphDataDict:
          value = graphDataDict[key]
          if type(value).__name__ == 'list':
            utils.printCsv(csv, *(['"' + l[0] + '"' for l in value]))
            for i in range(1, len(value[0])):
              utils.printCsv(csv, *([l[i] for l in value]))
          else:
            utils.printCsv(csv, '"' + key + '"', value)
      if generatePointSet:
        for key in graphDataDict:
          value = graphDataDict[key]
          if type(value).__name__ == 'list':
            for i in range(len(value)):
              subvalue = value[i]
              name = subvalue[0]
              subdata = subvalue[1:]
              if i == 0:
                output.updateInputValue(name, subdata)
              else:
                output.updateOutputValue(name, subdata)
            break  # XXX Need to figure out way to specify which data to return
      if generateCSV:
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
          extraCsv = Files.returnInstance('CSV',self)
          extraCsv.initialize(newFileName,self.messageHandler)
          extraCsv.open("w")
          extraCsv.write(",".join(['"' + str(x[0]) + '"' for x in dataPairs]))
          extraCsv.write("\n")
          extraCsv.write(",".join([str(x[1]) for x in dataPairs]))
          extraCsv.write("\n")
          extraCsv.close()
        utils.printCsv(csv)

  def __processData(self, data, methodInfo):
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
    delta = math.sqrt((math.pi / 2.0) * (abs(skewness) ** (2.0 / 3.0)) /
                      (abs(skewness) ** (2.0 / 3.0) + ((4.0 - math.pi) / 2.0) ** (2.0 / 3.0)))
    delta = math.copysign(delta, skewness)
    alpha = delta / math.sqrt(1.0 - delta ** 2)
    variance = ret["sampleVariance"]
    omega = variance / (1.0 - 2 * delta ** 2 / math.pi)
    mean = ret['mean']
    xi = mean - omega * delta * math.sqrt(2.0 / math.pi)
    ret['alpha'] = alpha
    ret['omega'] = omega
    ret['xi'] = xi
    return ret
