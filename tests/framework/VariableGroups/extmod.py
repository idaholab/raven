import numpy as np

def run(self,Input):
  vals = [self.x1,self.x2,self.x3,self.x4,self.x5,self.x6]
  self.y1 = np.sum(vals)
  self.y2 = np.average(vals)
