import math

def run(self,Input):
  a =  1.0
  b =  2.0
  c =  3.0
  l = -1.0

  self.y1 = self.x1
  self.y2 = self.x1
  self.y3 = a*self.x1 + b*self.x2 - c*self.x3
  self.y4 = self.x1*self.x1 + self.x1*self.x2*self.x3
  self.y5 = math.exp(l*self.x1)

  if self.y4 < 5.0:
    self.failure = 0.0
  else:
    self.failure = 1.0
