def run(self,Input):
  """
    Method that implement a simple system with three components in a parallel configuration
    @ In, Input, dict, dictionary containing the data
    @ Out, outcome, float, logical status of the system given status of the components
  """
  Bstatus   = Input['Bstatus'][0]
  Cstatus   = Input['Cstatus'][0]
  Dstatus   = Input['Dstatus'][0]

  if (Bstatus == 1.0 and Cstatus == 1.0 and Dstatus==1.0):
    self.outcome = 1.0
  else:
    self.outcome = 0.0

  return self.outcome
