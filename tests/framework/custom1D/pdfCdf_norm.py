from scipy.stats import norm

def initialize(self,runInfoDict,inputFiles):
  return

def run(self,Input):
  self.x = Input['x']

  self.pdf = norm.pdf(self.x)
  self.cdf = norm.cdf(self.x)


