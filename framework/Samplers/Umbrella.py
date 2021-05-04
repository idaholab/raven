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
    print("In this class!!!!!")
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

    # if paramInput.findFirst('samplerInit') != None:
    #   if self.limit is None:
    #     self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+' needs the limit block (number of samples) in the samplerInit block')
    #   if paramInput.findFirst('samplerInit').findFirst('samplingType')!= None:
    #     self.samplingType = paramInput.findFirst('samplerInit').findFirst('samplingType').value
    #     if self.samplingType not in ['uniform']:
    #       self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+': specified type of samplingType is not recognized. Allowed type is: uniform')
    #   else:
    #     self.samplingType = None
    # else:
    #   self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+' needs the samplerInit block')

  def stratified_uniform_sample(self, m_per_bin, n_bins):
    """
    function to generate a uniform stratified sample
    :param m_per_bin:
    :param n_bins:
    :return:
    """
    n = n_bins
    m = m_per_bin
    samples = []
    bounds = [num / n for num in range(0, n + 1)]

    for i in range(0, n):
      LB = bounds[i]
      UB = bounds[i + 1]
      bin_sample = np.random.uniform(LB, UB, m)

      samples[(i * m + 1):((i + 1) * m)] = bin_sample

    return bounds[1:], samples

  def get_std(self, k, sigma):
    ## TODO parameterized tail probability
    beta = (k + np.sqrt(k ** 2 + 4 * sigma ** 2)) / (2 * sigma ** 2)
    alpha = beta * k + 1
    return (scipy.stats.gamma.cdf(k,
                                  a=alpha,
                                  scale=1 / beta) - 0.4)

  def tail_prob_gamma(self, k, sigma):
    D = k ** 2 + 4 * sigma ** 2
    beta = (k + D ** 0.5) / (2 * sigma ** 2)
    alpha = beta * k + 1
    value = 1 - gamma.cdf(k, scale=1 / beta, a=alpha)
    return {'alpha': alpha, 'beta': beta, 'tailprob': value,
            'k': k, 'sigma': sigma}
  # Function for Non stratified importance sampling
  def importance_sample(self, importance_samples):
    """

    :param importance_samples:
    :return:
    """
    # importance_dist = norm(appro_mu, appro_sigma)
    # target_dist = norm(target_mu, target_sigma)

    importance_weights = self.distDict['target'].pdf(
      importance_samples) / self.distDict['importance'].pdf(importance_samples)

    sample_df = pd.DataFrame({'sample': importance_samples,
                              'weights': importance_weights})
    return sample_df

  # Function to evaluate a multivariate gamma tail pdf in all hyper-quadrants with
  # each tail being given a weight w. These weights need to sum to one.
  def mv_gamma_tail_pdf(self, x, vertWeights):

    n_vertices = len(vertWeights['vert'])
    n_row = len(x)
    pdf_values = np.zeros(n_row)

    for i in range(n_row):
      the_x = x[i, :]
      total = 0

      for k in range(n_vertices):
        vertex = vertWeights['vert'][k]
        x_vertex = the_x - vertex 
        x_vertex = x_vertex * vertex
        marg_values = scipy.stats.gamma.pdf(x_vertex, scale=1 / self.tail_prob ['beta'], a=self.tail_prob ['alpha'])
        prod_values = reduce(operator.mul, marg_values, 1)
        total = total + vertWeights['w'][k] * prod_values
      pdf_values[i] = total
    return pd.DataFrame([x[:, 0], x[:, 1], pdf_values]).T

  # Function to sample from a discrete distribution
  def rdiscrete(self, n, the_w):
    P = np.cumsum(the_w)
    m = len(the_w)
    x = np.arange(1, m + 1)
    unif_sample = np.random.uniform(size=n)
    discrete_sample = np.zeros(n)
    for i in range(m - 1, -1, -1):
      discrete_sample[unif_sample < P[i]] = x[i]
    return discrete_sample

  # Function to create the vertices of a K dimensional unit square
  def Construct_Corner_Points(self, dimensions):
    n_rows = 2 ** dimensions
    vertices = np.zeros(shape=(n_rows, dimensions))
    for i in range(1, n_rows + 1):
      for j in range(1, dimensions + 1):
        power = math.floor((i - 1) / 2 ** (j - 1))
        vertices[i - 1, j - 1] = (-1) ** (power + 2)
    return vertices

  def multivariate_umbrella_sample(self, n, w, p, k, mean, Pearsons_Rho_Matrix, vertWeights):

    dimension = 2
    # n_tails = 2 ** dimension
    # tail_wts = (1 - w) / n_tails
    # wts_list = [tail_wts for i in range(0, n_tails)]
    # wts_list.insert(0, w)
    wts_list = vertWeights['w']
    wts_list = np.insert(wts_list, 0, w)
    vertices = vertWeights['vert']
    memb_sample = self.rdiscrete(n, wts_list)
    mixt_sample = np.zeros((n, 2))


    # vertices = Construct_Corner_Points(dimension)
    for i in range(0, len(wts_list)):
      n_sub = len(memb_sample[memb_sample == i])
      if n_sub != 0:
        if i == 0:

          mixt_sample[memb_sample == i] = np.random.multivariate_normal(mean, Pearsons_Rho_Matrix, size=n_sub)
        else:
          for col in range(dimension):
            mixt_sample[memb_sample == i, col] = vertices[i - 1, col] * gamma.rvs(scale=1 / self.tail_prob ['beta'],
                                                                                  a=self.tail_prob ['alpha'],
                                                                                  size=n_sub)
    return mixt_sample

  # Function to evaluate the multivariate umbrella pdf with gamma tails. The gamma tails
  # have their seperate weighing scheme.
  def multivariate_umbrella_pdf(self, x, vertWeights):
    # w is the weight of the target density (1-w) is distributed over the tails
    mv_norm = multivariate_normal([0,0], [[1, 0.5],[0.5, 1]])
    target_pdf = mv_norm.pdf(x)
    # target_pdf = self.distDict['target'].pdf(x)
    gamma_tail_pdf = self.mv_gamma_tail_pdf(x, vertWeights)
    new_pdf_values = self.weightTarget * target_pdf + (1 - self.weightTarget) * gamma_tail_pdf[2]
    gamma_tail_pdf[2] = new_pdf_values
    return gamma_tail_pdf

  def multi_umbrella_with_weights(self,s, w, p, k, mu_orig, Pearsons_Rho_Matrix, vertWeights):
    x_sample = self.multivariate_umbrella_sample(s, w, p, k, mu_orig, Pearsons_Rho_Matrix, vertWeights)
    mv_norm = multivariate_normal(mu_orig, Pearsons_Rho_Matrix)
    target_pdf = mv_norm.pdf(x_sample)
    # Evaluating the importance weights
    imp_weights = target_pdf / self.multivariate_umbrella_pdf(x_sample, w, p, 0, k, vertWeights, mu_orig, Pearsons_Rho_Matrix)[2]

    results = pd.DataFrame(x_sample)
    results['weights'] = imp_weights
    return results

  # Function to evaluate a univariate variate gamma tail sample in positive hyper-quadrant
  def univ_gamma_tail_sample(self, sample_size):

    sample = gamma.rvs(scale=1 / self.tail_prob ['beta'],
                       a=self.tail_prob ['alpha'],
                       size=sample_size)
    return sample

  def multi_gamma_tail_sample(self, sample_size, vertWeights):
    multi_sample = np.empty(shape=(sample_size, self.dimension))

    for i in range(0, self.dimension):
      sample = self.univ_gamma_tail_sample(sample_size)
      multi_sample[:, i] = sample
    vertex_sample = self.rdiscrete(sample_size, vertWeights['w'])
    new_sample = multi_sample

    for i in range(0, sample_size):
      new_sample[i,] = multi_sample[i,] * vertWeights['vert'][int(vertex_sample[i]) - 1,]

    new_sample = pd.DataFrame(
      {'vert': vertex_sample, 'new_sample_x1': new_sample[:, 0], 'new_sample_x2': new_sample[:, 1], 'tail_flag': 1})
    return new_sample

  def multi_umbrella_sample(self, sample_size, vertWeights):
    pmf = [self.weightTarget, 1 - self.weightTarget]
    disc_sample = self.rdiscrete(sample_size, pmf)
    target_sample_size = len(disc_sample[disc_sample == 1])
    tail_sample_size = sample_size - target_sample_size
    tail_sample = self.multi_gamma_tail_sample(tail_sample_size, vertWeights)
    target_sample = np.random.multivariate_normal([0,0], [[1, 0.5],[0.5, 1]], size=target_sample_size)
    vert_target = np.zeros(shape=target_sample_size)
    
    target_sample = pd.DataFrame(
      {'vert': vert_target, 'new_sample_x1': target_sample[:, 0], 'new_sample_x2': target_sample[:, 1], 'tail_flag': 0})
    umbrella_sample = target_sample.append(tail_sample)
    samples = umbrella_sample[['new_sample_x1', 'new_sample_x2']]
    x_points = samples.to_numpy()
    umb_pdf_values = self.multivariate_umbrella_pdf(x_points, vertWeights)
    mv_norm = multivariate_normal([0,0], [[1, 0.5],[0.5, 1]])
    target_pdf = mv_norm.pdf(x_points)
    # Evaluating the importance weights
    imp_weights = target_pdf / umb_pdf_values[2]
   
    umbrella_samples = pd.DataFrame(
      {'vert': umbrella_sample['vert'].to_numpy(), 'weights': imp_weights.to_numpy(), 'x1': x_points[:, 0],
       'x2': x_points[:, 1], 'tail_flag': umbrella_sample['tail_flag'].to_numpy()})

    return umbrella_samples

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """

    vertices = self.Construct_Corner_Points(self.dimension)
    n_vertices = len(vertices)
    verticesWeights = np.ones(n_vertices) / n_vertices
    verticesWeights = dict({"vert": vertices, "w": verticesWeights})
    sigma_k = scipy.optimize.root(partial(self.get_std, self.modeLocation), 1, tol=0.0000001)['x'][0]
    self.tail_prob = self.tail_prob_gamma(self.modeLocation, sigma_k)
    sample_size = 30
    umbrellaSample = self.multi_umbrella_sample(sample_size,
                                             verticesWeights)

    print('umbrellaSample')
    print(umbrellaSample)

    #
    # self.values['x1'] = np.atleast_2d(umbrellaSample[['x1', 'x2']])
    # self.values['x2'] = np.atleast_2d(umbrellaSample[['x1', 'x2']])
    #
    #
    self.inputInfo['SampledVars']['x1'] = umbrellaSample['x1'].to_numpy()
    # # self.inputInfo['SampledVars']['x2'] = umbrellaSample['x2']
    # #
    self.inputInfo['ProbabilityWeight'] = 1.0
    self.inputInfo['ProbabilityWeight-x1'] =  umbrellaSample['weights'].to_numpy()
    # self.inputInfo['ProbabilityWeight-x2'] = umbrellaSample['weights'].to_numpy()
    self.inputInfo['PointProbability'] =1.0
    self.inputInfo['SamplerType'] = 'Umbrella'
    print(self.inputInfo)
