import numpy as np

def initialize(self,runInfoDict,inputFiles):
  self.const1 = 3.5


def run(self,Input):
  self.Y1 = self.X1*self.X2 + self.const1
  self.Y2 = 0.7*self.X1 + self.X2*self.const1

