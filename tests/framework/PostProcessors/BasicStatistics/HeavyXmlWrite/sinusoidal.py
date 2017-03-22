
import numpy as np

# just does a sinusoidal and its square

def run(self,Input):
  freq = Input['frequency']
  mag = Input['magnitude']
  num_steps = 8000
  max_eval = 2.*np.pi
  steps = np.linspace(0,max_eval,num_steps)
  self.single = np.sin(freq*steps)
  self.square = self.single*self.single
  self.time = steps
