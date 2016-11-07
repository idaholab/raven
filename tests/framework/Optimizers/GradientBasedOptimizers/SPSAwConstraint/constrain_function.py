import os
import sys
frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(frameworkDir,'utils'))
import utils
utils.find_crow(frameworkDir)
distribution1D = utils.find_distribution1D()
stochasticEnv = distribution1D.DistributionContainer.instance()
import math
normal1 = distribution1D.BasicNormalDistribution(0.5, 0.05, 0.0,1.0)

def constrain(self):
  B, R = self.B, self.R
  f, d = 0.5, 0.5
  if B + R * f - d > 0:
    returnValue = 1
  else:
  	returnValue = 0
  return returnValue
