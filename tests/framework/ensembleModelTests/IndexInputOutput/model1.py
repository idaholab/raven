import numpy as np

def run(obj,dct):
  times = np.linspace(0,1,11)
  obj.t = times
  obj.hist1 = times*obj.scalar1
