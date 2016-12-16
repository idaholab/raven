import math

def run(self,Input):
  coordinate = math.sqrt(self.x1**2 + self.x2**2)
  if coordinate > 1.0:
    self.out = 0.0
  else:
    self.out = 1.0


