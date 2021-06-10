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
  This module contains the Umbrella sampling strategy

  Created on Feb 15, 2021
  @author: Tanaya
  supercedes Samplers.py from crisr
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
from operator import mul
from functools import reduce, partial
import scipy
import pandas as pd
from scipy.stats import gamma, norm, multivariate_normal
import operator
import math
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .ForwardSampler import ForwardSampler
from utils import utils,randomUtils,InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class Umbrella(ForwardSampler):
  """
    MONTE CARLO Sampler
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
    inputSpecification = super(Umbrella, cls).getInputSpecification()
    umbrellaInput = InputData.parameterInputFactory("umbrella", contentType=InputTypes.StringType)
    samplerInitInput = InputData.parameterInputFactory("samplerInit")
    dimInput = InputData.parameterInputFactory("dimensions", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(dimInput)
    limit = InputData.parameterInputFactory("limit", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(limit)
    modeLocationInput = InputData.parameterInputFactory("modeLocation", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(modeLocationInput)
    tailProbabilityInput = InputData.parameterInputFactory("tailProbability", contentType=InputTypes.FloatType)
    samplerInitInput.addSub(tailProbabilityInput)
    weightTargetInput = InputData.parameterInputFactory("weightTarget", contentType=InputTypes.FloatType)
    samplerInitInput.addSub(weightTargetInput)
    rhoInput = InputData.parameterInputFactory("rho", contentType=InputTypes.FloatType)
    samplerInitInput.addSub(rhoInput)
    targetDistribution = InputData.parameterInputFactory("distribution")
    samplerInitInput.addSub(targetDistribution)
    samplingTypeInput = InputData.parameterInputFactory("samplingType", contentType=InputTypes.StringType)
    samplerInitInput.addSub(samplingTypeInput)
    inputSpecification.addSub(samplerInitInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    ForwardSampler.__init__(self)
    self.printTag = 'SAMPLER UMBRELLA'
    self.samplingType = None
    self.limit = None
    self.modeLocation = 0
    self.rho = 0
    self.dimension = 0

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    ForwardSampler.readSamplerInit(self,xmlNode)
    self.modeLocation = paramInput.findFirst('samplerInit').findFirst('modeLocation').value
    self.tailProb = paramInput.findFirst('samplerInit').findFirst('tailProbability').value
    self.rho = paramInput.findFirst('samplerInit').findFirst('rho').value
    self.dimension = paramInput.findFirst('samplerInit').findFirst('dimensions').value
    self.weightTarget = paramInput.findFirst('samplerInit').findFirst('weightTarget').value

    if paramInput.findFirst('samplerInit') != None:
      if self.limit is None:
        self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+' needs the limit block (number of samples) in the samplerInit block')
      if paramInput.findFirst('samplerInit').findFirst('samplingType')!= None:
        self.samplingType = paramInput.findFirst('samplerInit').findFirst('samplingType').value
        if self.samplingType not in ['uniform']:
          self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+': specified type of samplingType is not recognized. Allowed type is: uniform')
      else:
        self.samplingType = None
    else:
      self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+' needs the samplerInit block')

  def getStd(self, k, sigma):
    """
         Method to get alpha and beta values of gamma distribution based on mode location, standard deviation and tail probability
         @ In, k, mode location of the gamma distribution specified by the user
         @ In, tail, tail probability of the distribution
         @ Out, sigma, standard distribution of the gamma distribution
    """
    beta = (k + np.sqrt(k ** 2 + 4 * sigma ** 2)) / (2 * sigma ** 2)
    alpha = beta * k + 1
    return (scipy.stats.gamma.cdf(k,
                                  a=alpha,
                                  scale=1 / beta) - (1- self.tailProb))

  def tailProbabilityGamma(self, k, sigma):
    """
         Method to collect parameters of gamma distribution in a dictionary
         @ In, k, mode location of the gamma distribution specified by the user
         @ In, sigma, standard distribution of the gamma distribution
         @ Out, dictionary of gamma parameters
    """
    D = k ** 2 + 4 * sigma ** 2
    beta = (k + D ** 0.5) / (2 * sigma ** 2)
    alpha = beta * k + 1
    value = 1 - gamma.cdf(k, scale=1 / beta, a=alpha)
    return {'alpha': alpha, 'beta': beta, 'tailprob': value,
            'k': k, 'sigma': sigma}

  def getDiscreteSamples(self, n, the_w):
    """
         Method to sample from a discrete distribution
         @ In, n, samples size
         @ In, the_w, weights of the mixture distribution
         @ Out, dataframe, samples from a discrete distribution
    """
    P = np.cumsum(the_w)
    m = len(the_w)
    x = np.arange(1, m + 1)
    uniformSample = np.random.uniform(size=n)
    discreteSample = np.zeros(n)
    for i in range(m - 1, -1, -1):
      discreteSample[uniformSample < P[i]] = x[i]
    return discreteSample

  def constructCornerPoints(self):
    """
         Method to create the vertices of a K dimensional unit square
         @ In, dimensions, dimensions specified by the user
         @ Out, dataframe, vertices in K dimensional unit square
    """
    nrows = 2 ** self.dimension
    vertices = np.zeros(shape=(nrows, self.dimension))
    for i in range(1, nrows + 1):
      for j in range(1, self.dimension + 1):
        power = math.floor((i - 1) / 2 ** (j - 1))
        vertices[i - 1, j - 1] = (-1) ** (power + 2)
    return vertices

  def gammaTailPDF(self, x, vertWeights):
    """
         Method to evaluate a multivariate gamma tail pdf in all hyper-quadrants with
         each tail being given a weight w. These weights need to sum to one.
         @ In, x, samples
         @ In, vertWeights, weights of the mixture distribution
         @ Out, dataframe, dataframe of tail samples and associated probability values
    """
    nvertices = len(vertWeights['vert'])
    nrow = len(x)
    pdfValues = np.zeros(nrow)

    for i in range(nrow):
      the_x = x[i, :]
      total = 0

      for k in range(nvertices):
        vertex = vertWeights['vert'][k]
        x_vertex = the_x - vertex
        x_vertex = x_vertex * vertex
        marginalValues = scipy.stats.gamma.pdf(x_vertex, scale=1 / self.tailProb['beta'], a=self.tailProb['alpha'])
        prodValues = reduce(operator.mul, marginalValues, 1)
        total = total + vertWeights['w'][k] * prodValues
      pdfValues[i] = total
    samples = pd.DataFrame(x)
    samples['pdfValues'] = pdfValues
    return samples

  def umbrellaPDF(self, x, vertWeights):
    """
         Function to evaluate the multivariate umbrella pdf with gamma tails. The gamma tails have their seperate weighing scheme.
         @ In, x, samples
         @ In, vertWeights, weights of the mixture distribution
         @ Out, dataframe, dataframe of samples and associated probability values
    """
    mvnorm = multivariate_normal(self.mu, self.covariance)
    targetPDF = mvnorm.pdf(x)
    # target_pdf = self.distDict['target'].pdf(x)
    gammaTailPDFValues = self.gammaTailPDF(x, vertWeights)
    umbrellaPDFValues = self.weightTarget * targetPDF + (1 - self.weightTarget) * gammaTailPDFValues['pdfValues']
    gammaTailPDFValues['pdfValues'] = umbrellaPDFValues
    return gammaTailPDFValues

  def univGammaTailSample(self, sample_size):
    """
         Method to evaluate a univariate variate gamma tail sample in positive hyper-quadrant
         @ In, sample_size
         @ Out, sample, univariate gamma samples
    """
    sample = gamma.rvs(scale=1 / self.tailProb ['beta'],
                       a=self.tailProb ['alpha'],
                       size=sample_size)
    return sample

  def gammaTailSample(self, sampleSize, vertWeights):
    """
         Method to generate multivariate gamma tail samples
         @ In, sample_size
         @ In, vertWeights, weights of the mixture distribution
         @ Out, dataframe, dataframe of samples
    """
    multiSamples = np.empty(shape=(sampleSize, self.dimension))

    for i in range(0, self.dimension):
      samples = self.univGammaTailSample(sampleSize)
      multiSamples[:, i] = samples
    vertexSample = self.getDiscreteSamples(sampleSize, vertWeights['w'])
    newSamples = multiSamples

    for i in range(0, sampleSize):
      newSamples[i,] = multiSamples[i,] * vertWeights['vert'][int(vertexSample[i]) - 1,]
    newSamplesDF = pd.DataFrame(newSamples)
    newSamplesDF['vert'] = vertexSample
    newSamplesDF['tail_flag'] = 1

    return newSamplesDF

  def umbrellaSample(self, sampleSize, vertWeights):
    """
         Method to generate samples from multivariate umbrella distribution
         @ In, sample_size
         @ In, vertWeights, weights of the mixture distribution
         @ Out, dataframe of umbrella samples
    """
    pmf = [self.weightTarget, 1 - self.weightTarget]
    discreteSamples = self.getDiscreteSamples(sampleSize, pmf)
    targetSampleSize = len(discreteSamples[discreteSamples == 1])
    tailSampleSize = sampleSize - targetSampleSize
    tailSamples = self.gammaTailSample(tailSampleSize, vertWeights)
    targetSamples = np.random.multivariate_normal(self.mu, self.covariance, size=targetSampleSize)
    vertTarget = np.zeros(shape=targetSampleSize)
    targetSamplesDF = pd.DataFrame(targetSamples)
    targetSamplesDF['vert'] = vertTarget
    targetSamplesDF['tail_flag'] = 0
    umbrellaSamples = targetSamplesDF.append(tailSamples, ignore_index=True)
    samples = umbrellaSamples.loc[:, ~umbrellaSamples.columns.isin(['vert', 'tail_flag'])]
    xpoints = samples.to_numpy()
    umbrellaPDFValues = self.umbrellaPDF(xpoints, vertWeights)
    mvnorm = multivariate_normal(self.mu, self.covariance)
    targetPDF = mvnorm.pdf(xpoints)
    importanceWeights = targetPDF / umbrellaPDFValues['pdfValues']  # Evaluating the importance weights
    umbrellaSamples['weights'] = importanceWeights
    return umbrellaSamples

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """

    vertices = self.constructCornerPoints()
    nvertices = len(vertices)
    verticesWeights = np.ones(nvertices) / nvertices
    verticesWeights = dict({"vert": vertices, "w": verticesWeights})
    self.mu = [0] * self.dimension
    self.covariance = np.ones(shape=(self.dimension, self.dimension)) * self.rho
    for i in range(self.dimension):
      self.covariance[i, i] = 1

    sigma = scipy.optimize.root(partial(self.getStd, self.modeLocation), 1, tol=0.0000001)['x'][0]
    self.tailProb = self.tailProbabilityGamma(self.modeLocation, sigma)
    sampleSize = 30
    umbrellaSample = self.umbrellaSample(sampleSize,
                                         verticesWeights)

    print('umbrellaSample')
    print(umbrellaSample)
    # self.inputInfo['SampledVars']['x1'] = umbrellaSample[0].to_numpy()
    self.inputInfo['ProbabilityWeight'] = 1.0
    self.inputInfo['ProbabilityWeight-x1'] = umbrellaSample['weights'].to_numpy()
    self.inputInfo['PointProbability'] =1.0
    self.inputInfo['SamplerType'] = 'Umbrella'