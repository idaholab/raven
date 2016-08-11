from scipy.stats import expon

def initialize(self,runInfoDict,inputFiles):
  return

def run(self,Input):
  self.x = Input['x']

  self.pdf = expon.pdf(self.x)
  self.cdf = expon.cdf(self.x)


