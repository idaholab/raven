#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
import numpy as np
import time

def evaluate(inp):
  return np.prod(list(1.+n for n in inp))

def run(self,Input):
  self.ans = self.x1**2*self.x2 + self.x1**2 + self.x1*self.x2 + self.x1 + self.x2 + 1.
  self.ans2 = self.x1*self.x2 + self.x1 + self.x2 + 1.
  time.sleep(0.01) #for testing collection before completion

#analytic values:
#
# ans
#
# mean  :  4/ 3 = 1.33333333333333
# second: 44/15 = 2.93333333333333
# var   : 52/45 = 1.15555555555555
#
# ans2
#
# mean  :  1
# var   :  7/ 9 = 0.77777777777777
