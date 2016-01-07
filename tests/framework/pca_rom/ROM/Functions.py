import numpy as np

def initialize(self,runInfoDict,inputFiles):
  self.response = 3.0
  return

def run(self,Input):
  dim = 308
  # use a to represent the sensitivity
  senVec = np.loadtxt('sensitivity.txt')
  #senVec = np.random.rand(dim)
  varBase = 'x_'
  inputVar = []
  for i in range(dim):
    varname = varBase + str(i)
    inputVar.append(Input[varname])
  inputVar = np.asarray(inputVar)
  self.response = np.dot(senVec,inputVar)



