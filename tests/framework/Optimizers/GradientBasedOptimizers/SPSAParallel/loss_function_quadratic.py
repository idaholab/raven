import os
import sys
frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(frameworkDir,'utils'))
import utils
utils.find_crow(frameworkDir)
distribution1D = utils.find_distribution1D()
stochasticEnv = distribution1D.DistributionContainer.Instance()
import math
normal1 = distribution1D.BasicNormalDistribution(1.0, 0.01, 0.0,100.0)

def initialize(self, runInfo, Input):
  self.a1 = 10.0
  self.a2 = 10.0
  self.b1 = 0.5
  self.b2 = 0.5

def run(self,Input):
  # sample f and d
#  rand  = stochasticEnv.random()
#   rand2  = stochasticEnv.random()
#   self.b1 = normal1.InverseCdf(rand1)
#  noise = normal1.InverseCdf(rand)

  a1, a2, b1, b2 = self.a1, self.a2, self.b1, self.b2
  x1, x2 = self.x1, self.x2
  self.c = a1*(x1-b1)**2 + a2*(x2-b2)**2 #+ noise

