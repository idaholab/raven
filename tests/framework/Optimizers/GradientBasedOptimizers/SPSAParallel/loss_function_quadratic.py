

def initialize(self, runInfo, Input):
  self.a1 = 10.0
  self.a2 = 10.0
  self.b1 = 0.5
  self.b2 = 0.5

def run(self,Input):
  a1, a2, b1, b2 = self.a1, self.a2, self.b1, self.b2
  x1, x2 = self.x1, self.x2
  self.c = a1*(x1-b1)**2 + a2*(x2-b2)**2 #+ noise

