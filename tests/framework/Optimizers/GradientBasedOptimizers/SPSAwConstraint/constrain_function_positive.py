
# Debug
import os
import sys
frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(frameworkDir,'utils'))
import utils
utils.find_crow(frameworkDir)
distribution1D = utils.find_distribution1D()
stochasticEnv = distribution1D.DistributionContainer.Instance()
import math
normal1 = distribution1D.BasicNormalDistribution(0.5, 0.05, 0.0,1.0)



def constrain(self):
  B, R = self.B, self.R#, self.f, self.d
  # Debug
  f, d = 0.5, 0.5
#   rand1  = stochasticEnv.random()
#   rand2  = stochasticEnv.random()
#   d = normal1.InverseCdf(rand1)
#   f = normal1.InverseCdf(rand2)
  # End of debug
  if B + R * f - d > 0:
#   if self.B + self.R * self.f - self.d > 0:
    returnValue = 1
    if B < 0.0 or R < 0.0 or B > 1 or R > 1:
      returnValue = 0
  else:
  	returnValue = 0
  #print("f " + str(self.f) + " d " + str(self.d))

  return returnValue
