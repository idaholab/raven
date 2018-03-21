import numpy as np

def run(self,Input):
  for key,val in Input.items():
    print key,val
  self.scalarOut = self.scalarIn**2
  self.vectorOut = self.vectorIn**2
  self.t = np.arange(len(self.vectorIn))
