'''
Created on Feb 28, 2013

@author: mandd
'''

import numpy as np
import distribution1D

def LHSsampler(distributions, numberOfSamples):
  """
  Stratified sampler; latin hypercube sampler - like

  Parameters:
      - "distributions"  : list of distributions
      - "discretization" : sampling points that describe the stratification strategy    o____o____o____o
                           Type: list of numpy arrays since discertization may change from variable to variable
      - "numberOfSamples": number of samples that need to be generated (type int)
  """
  numberOfVariables=distributions.len()
  
  samples=np.zeros(numberOfSamples,numberOfVariables)
  
  discretization = generateEquiProbableDiscretization(distributions, numberOfSamples)
   
  for v in range(numberOfVariables): 
    variablesSampling=np.zeros(numberOfSamples,1)
    
    for n in range(numberOfSamples):
    
      numberOfIntervals=discretization[v].len()-1
      
      #Pick interval
      intervalPicked = np.random.rand()*numberOfIntervals
      xMin = discretization[v,floor(intervalPicked)]
      xMax = discretization[v,floor(intervalPicked)+1]
      
      #Pick sample in that interval
      cdfValue = xMin + (xMax-xMin) * np.random.rand()
      
      #Determine corresponding value in the distriution
      variablesSampling[n,1] = distcont.randGen(distributions[v],cdfValue)
      
      np.random.shuffle(variablesSampling)
    
    samples[:,v]=variablesSampling
      
  return samples

def LHSsampler(distributions, numberOfSamples, covMatrix):
  L=np.linalg.cholesky(covMatrix)
  
  samples= LHSsampler(distributions, numberOfSamples)
  
  correlatedSamples=np.dot(samples,L)
  
  return correlatedSamples

def generateEquiProbableDiscretization(distributions, numberOfSamples):
  """
  Parameters
      - distributions   : list of distributions
      - numberOfSamples : array of integers
  """
  numberOfVariabales=distributions.len()
  
  for v in range(numberOfVariabales):
    discretization=zeros(1,numberOfIntervals+1)
    
    intervalLength=1/numberOfIntervals
    
    for i in range(numberOfIntervals+1):
      discretization[1,i] = distcont.randGen(distributions[v],intervalLength*i)
         
  return discretization
