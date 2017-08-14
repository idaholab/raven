def run(self,Inputs):
  # test to make sure other methods were run
  if self.x != 0.0:
    self.y = 1.0/self.x
  else:
    self.y = float('inf') + self.x

