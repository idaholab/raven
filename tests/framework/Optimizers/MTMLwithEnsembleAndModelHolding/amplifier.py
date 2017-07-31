# takes input parameters ans and z
# returns value in "objAns" amplifying the self.ans times (self.z+1.0)

def run(self,Inputs):
  """
    This method just amplifies the response self.ans with the sampled variable self.z
    In a minimization optimization problem, self.z should be get close to 0.0
    @ In, Inputs, list, list of inputs
    @ Out, None
  """
  self.objAns = self.ans*(self.z+1.0)
  print(self.ans,self.z,self.objAns)


