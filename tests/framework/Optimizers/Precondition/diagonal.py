def evaluate(y):
  return y

def run(self,Inputs):
  # test to make sure other methods were run
  self.fromXML
  self.fromInitialize
  self.fromCreate
  # check constants are grabbed
  self.samplerConstant
  #actual run
  self.y = evaluate(self.x)

def _readMoreXML(self,xmlNode):
  print('DEBUGG xmlNode:',xmlNode.tag)
  self.fromXML = float(xmlNode.find('arbitraryRoot').find('arbitraryData').text)

def initialize(self,runInfo,inputs):
  self.fromInitialize = 12.

def createNewInput(self,inputs,samplerType,**kwargs):
  self.fromCreate = 64.
  return {'fromExtModCreateNewInputReturnDict':32.}
