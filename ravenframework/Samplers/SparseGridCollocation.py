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
  This module contains the Stochastic Collocation sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from talbpw
"""
# External Modules----------------------------------------------------------------------------------
from operator import mul
from functools import reduce
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .Grid import Grid
from ..utils import utils, InputData, InputTypes
from .. import Distributions
from .. import Quadratures
from .. import OrthoPolynomials
from .. import IndexSets
# Internal Modules End------------------------------------------------------------------------------

class SparseGridCollocation(Grid):
  """
    Sparse Grid Collocation sampling strategy
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
    inputSpecification = super(SparseGridCollocation, cls).getInputSpecification()
    inputSpecification.addParam("parallel", InputTypes.StringType)
    inputSpecification.addParam("outfile", InputTypes.StringType)

    romInput = InputData.parameterInputFactory("ROM", contentType=InputTypes.StringType)
    romInput.addParam("type", InputTypes.StringType)
    romInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(romInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type           = 'SparseGridCollocationSampler'
    self.printTag       = 'SAMPLER '+self.type.upper()
    self.maxPolyOrder   = None  #L, the relative maximum polynomial order to use in any dimension
    self.indexSetType   = None  #TP, TD, or HC; the type of index set to use
    self.polyDict       = {}    #varName-indexed dict of polynomial types
    self.quadDict       = {}    #varName-indexed dict of quadrature types
    self.importanceDict = {}    #varName-indexed dict of importance weights
    self.maxPolyOrder   = None  #integer, relative maximum polynomial order to be used in any one dimension
    self.lastOutput     = None  #pointer to output dataObjects object
    self.ROM            = None  #pointer to ROM
    self.jobHandler     = None  #pointer to job handler for parallel runs
    self.doInParallel   = True  #compute sparse grid in parallel flag, recommended True
    self.dists          = {}    #Contains the instance of the distribution to be used. keys are the variable names
    self.writeOut       = None
    self.indexSet       = None
    self.sparseGrid     = None
    self.features       = None
    self.sparseGridType = None
    self.addAssemblerObject('ROM', InputData.Quantity.one)

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In, None
      @ Out, gridDict, dict, dictionary of objects needed
    """
    gridDict = Grid._localWhatDoINeed(self)
    gridDict['internal'] = [(None,'jobHandler')]

    return gridDict

  def _localGenerateAssembler(self,initDict):
    """
      Generates the assembler.
      @ In, initDict, dict, init objects
      @ Out, None
    """
    Grid._localGenerateAssembler(self, initDict)
    self.jobHandler = initDict['internal']['jobHandler']
    self.dists = self.transformDistDict()
    # Do a distributions check for ND
    # This sampler only accept ND distributions with variable transformation defined in this sampler
    for dist in self.dists.values():
      if isinstance(dist, Distributions.NDimensionalDistributions):
        self.raiseAnError(IOError, 'ND Dists contain the variables in the original input space are  not supported for this sampler!')

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    # TODO remove using xmlNode
    self.doInParallel = xmlNode.attrib['parallel'].lower() in ['1','t','true','y','yes'] if 'parallel' in xmlNode.attrib else True
    self.writeOut = xmlNode.attrib['outfile'] if 'outfile' in xmlNode.attrib else None
    for child in xmlNode:
      if child.tag == 'Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
        if varName not in self.dependentSample:
          self.axisName.append(varName)

  def transformDistDict(self):
    """
      Performs distribution transformation
      If the method 'pca' is used in the variables transformation (i.e. latentVariables to manifestVariables), the correlated variables
      will be transformed into uncorrelated variables with standard normal distributions. Thus, the dictionary of distributions will
      be also transformed.
      @ In, None
      @ Out, distDicts, dict, distribution dictionary {varName:DistributionObject}
    """
    # Generate a standard normal distribution, this is used to generate the sparse grid points and weights for multivariate normal
    # distribution if PCA is used.
    standardNormal = Distributions.Normal()
    standardNormal.mean = 0.0
    standardNormal.sigma = 1.0
    standardNormal.initializeDistribution()
    distDicts = {}
    for varName in self.variables2distributionsMapping:
      distDicts[varName] = self.distDict[varName]
    if self.variablesTransformationDict:
      for key, varsDict in self.variablesTransformationDict.items():
        if self.transformationMethod[key] == 'pca':
          listVars = varsDict['latentVariables']
          for var in listVars:
            distDicts[var] = standardNormal

    return distDicts

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    SVL = self.readFromROM()
    self._generateQuadsAndPolys(SVL)
    #print out the setup for each variable.
    msg = self.printTag+' INTERPOLATION INFO:\n'
    msg += '    Variable | Distribution | Quadrature | Polynomials\n'
    for v in self.quadDict:
      msg += '   '+' | '.join([v,self.distDict[v].type,self.quadDict[v].type,self.polyDict[v].type])+'\n'
    msg += '    Polynomial Set Degree: '+str(self.maxPolyOrder)+'\n'
    msg += '    Polynomial Set Type  : '+str(SVL.indexSetType)+'\n'
    self.raiseADebug(msg)

    self.raiseADebug('Starting index set generation...')
    self.indexSet = IndexSets.factory.returnInstance(SVL.indexSetType)
    self.indexSet.initialize(self.features, self.importanceDict, self.maxPolyOrder)
    if self.indexSet.type=='Custom':
      self.indexSet.setPoints(SVL.indexSetVals)

    self.sparseGrid = Quadratures.factory.returnInstance(self.sparseGridType)
    self.raiseADebug(f'Starting {self.sparseGridType} sparse grid generation...')
    self.sparseGrid.initialize(self.features, self.indexSet, self.dists, self.quadDict, self.jobHandler)

    if self.writeOut is not None:
      msg = self.sparseGrid.__csv__()
      outFile = open(self.writeOut,'w')
      outFile.writelines(msg)
      outFile.close()

    self.limit=len(self.sparseGrid)
    self.raiseADebug(f'Size of Sparse Grid: {self.limit}')
    self.raiseADebug('Finished sampler generation.')

    self.raiseADebug('indexset:',self.indexSet)
    for SVL in self.ROM.supervisedContainer:
      SVL.initialize({'SG': self.sparseGrid,
                      'dists': self.dists,
                      'quads': self.quadDict,
                      'polys': self.polyDict,
                      'iSet': self.indexSet})

  def _generateQuadsAndPolys(self,SVL):
    """
      Builds the quadrature objects, polynomial objects, and importance weights for all
      the distributed variables.  Also sets maxPolyOrder.
      @ In, SVL, supervisedContainer object, one of the supervisedContainer objects from the ROM
      @ Out, None
    """
    ROMdata = SVL.interpolationInfo()
    self.maxPolyOrder = SVL.maxPolyOrder
    #check input space consistency
    samVars=self.axisName[:]
    romVars=SVL.features[:]
    try:
      for v in self.axisName:
        samVars.remove(v)
        romVars.remove(v)
    except ValueError:
      self.raiseAnError(IOError, f'variable {v} used in sampler but not ROM features! Collocation requires all vars in both.')
    if len(romVars) > 0:
      self.raiseAnError(IOError, f'variables {romVars} specified in ROM but not sampler! Collocation requires all vars in both.')
    for v in ROMdata.keys():
      if v not in self.axisName:
        self.raiseAnError(IOError, f'variable "{v}" given interpolation rules but variable not in sampler!')
      else:
        self.gridInfo[v] = ROMdata[v] #quad, poly, weight
    #set defaults, then replace them if they're asked for
    for v in self.axisName:
      if v not in self.gridInfo:
        self.gridInfo[v]={'poly': 'DEFAULT', 'quad': 'DEFAULT', 'weight': '1'}
    #establish all the right names for the desired types
    for varName,dat in self.gridInfo.items():
      if dat['poly'] == 'DEFAULT':
        dat['poly'] = self.dists[varName].preferredPolynomials
      if dat['quad'] == 'DEFAULT':
        dat['quad'] = self.dists[varName].preferredQuadrature
      polyType=dat['poly']
      subType = None
      distr = self.dists[varName]
      if polyType == 'Legendre':
        if distr.type == 'Uniform':
          quadType=dat['quad']
        else:
          quadType='CDF'
          subType=dat['quad']
          if subType not in ['Legendre', 'ClenshawCurtis']:
            self.raiseAnError(IOError, f'Quadrature {subType} not compatible with Legendre polys for {distr.type} for variable {varName}!')
      else:
        quadType=dat['quad']
      if quadType not in distr.compatibleQuadrature:
        self.raiseAnError(IOError, f'Quadrature type "{quadType}" is not compatible with variable "{varName}" distribution "{distr.type}"')

      quad = Quadratures.factory.returnInstance(quadType, Subtype=subType)
      quad.initialize(distr)
      self.quadDict[varName]=quad

      poly = OrthoPolynomials.factory.returnInstance(polyType)
      poly.initialize(quad)
      self.polyDict[varName] = poly

      self.importanceDict[varName] = float(dat['weight'])

  def localGenerateInput(self, model, oldInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    try:
      pt,weight = self.sparseGrid[self.counter-1]
    except IndexError:
      raise utils.NoMoreSamplesNeeded

    for v, varName in enumerate(self.sparseGrid.varNames):
      # compute the SampledVarsPb for 1-D distribution
      if self.variables2distributionsMapping[varName]['totDim'] == 1:
        for key in varName.strip().split(','):
          self.values[key] = pt[v]
        self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(pt[v])
        self.inputInfo['ProbabilityWeight-'+varName] = self.inputInfo['SampledVarsPb'][varName]
      # compute the SampledVarsPb for N-D distribution
      # Assume only one N-D distribution is associated with sparse grid collocation method
      elif self.variables2distributionsMapping[varName]['totDim'] > 1 and self.variables2distributionsMapping[varName]['reducedDim'] ==1:
        dist = self.variables2distributionsMapping[varName]['name']
        ndCoordinates = np.zeros(len(self.distributions2variablesMapping[dist]))
        positionList = self.distributions2variablesIndexList[dist]
        for varDict in self.distributions2variablesMapping[dist]:
          var = utils.first(varDict.keys())
          position = utils.first(varDict.values())
          location = -1
          for key in var.strip().split(','):
            if key in self.sparseGrid.varNames:
              location = self.sparseGrid.varNames.index(key)
              break
          if location > -1:
            ndCoordinates[positionList.index(position)] = pt[location]
          else:
            self.raiseAnError(IOError, f'The variables {var} listed in sparse grid collocation sampler, but not used in the ROM!' )
          for key in var.strip().split(','):
            self.values[key] = pt[location]
        self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(ndCoordinates)
        self.inputInfo['ProbabilityWeight-'+dist] = self.inputInfo['SampledVarsPb'][varName]

    self.inputInfo['ProbabilityWeight'] = weight
    self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['SamplerType'] = 'Sparse Grid Collocation'

  def readFromROM(self):
    """
      Reads in required information from ROM and returns a sample supervisedLearning object.
      @ In, None
      @ Out, SVL, supervisedLearning object, SVL object
    """
    self.ROM = self.assemblerDict['ROM'][0][3]
    SVLs = self.ROM.supervisedContainer
    SVL = utils.first(SVLs)
    self.features = SVL.features
    self.sparseGridType = SVL.sparseGridType.lower()

    return SVL

  def flush(self):
    """
      Reset SparsGridCollocation attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.dists = {}
