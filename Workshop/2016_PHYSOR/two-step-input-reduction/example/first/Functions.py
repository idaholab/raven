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
  #tot = 1.
  #for v,var in enumerate(inputVar):
  #  tot *= senVec[v]*var + 1.0
  #self.response = tot
  self.response = np.prod(list(senVec[v]*inputVar[v]+1.0 for v in range(len(inputVar))))
  #self.response = np.dot(senVec,inputVar)



