import numpy as np

def initialize(self,runInfoDict,inputFiles):
  self.const1 = 3.5
  self.time   = np.zeros(1000)
  self.T1     = np.zeros(1000)
  self.T2     = np.zeros(1000)


def run(self,Input):
  self.Y1 = self.X1*self.X2 + self.const1
  self.Y2 = 0.7*self.X1 + self.X2*self.const1

  for i in range(1000):
    self.time[i] = float(i)
    self.T1[i]   = self.X1*np.sin(float(i)*np.pi/200.0) + self.X2
    self.T2[i]   = np.log(self.T1[i]*10.0/(float(i)+1)+400.0)
