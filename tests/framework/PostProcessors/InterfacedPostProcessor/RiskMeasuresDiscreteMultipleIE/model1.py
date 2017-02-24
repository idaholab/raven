
def initialize(self,runInfoDict,inputFiles):
  return

def run(self,Input):
  Astatus   = Input['Astatus'][0]
  Bstatus   = Input['Bstatus'][0]
  Cstatus   = Input['Cstatus'][0]
  
  if (Astatus == 1.0) or (Bstatus == 1.0 and Cstatus == 1.0):
    self.outcome = 1.0
  else:
    self.outcome = 0.0
  
  return self.outcome
