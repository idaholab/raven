
def initialize(self,runInfoDict,inputFiles):
  return

def run(self,Input):
  Bstatus   = Input['Bstatus'][0]
  Cstatus   = Input['Cstatus'][0]
  Dstatus   = Input['Dstatus'][0]

  if (Bstatus == 1.0 and Cstatus == 1.0 and Dstatus==1.0):
    self.outcome = 1.0
  else:
    self.outcome = 0.0

  return self.outcome
