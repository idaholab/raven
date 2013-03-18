'''
Created on Feb 21, 2013

@author: crisr
'''

class Step:
  def __init__(self,xmlNode):
    self.inputDict   = {}
    self.outputDict  = {}
    self.modelDict   = {}
    self.samplerDict = {}
    self.testerDict  = {}
    self.readXML(xmlNode)
    
  def readXML(self,xmlNode):
    for 
    return
  
  def takeAstep(self,Simulation):
    return

  def run(self):
    return
  
def returnStepClass(dataType,xmlNode):
  '''
  provide an interface generate a Step class of type dataType
  '''
  stepInterfaceDict = {}
  stepInterfaceDict['ProduceData'] = ProduceData
  stepInterfaceDict['GenerateROM'] = GenerateROM
  stepInterfaceDict['Compute']     = Compute
  try:
    if dataType in stepInterfaceDict.keys():
      return stepInterfaceDict[dataType](xmlNode)
  except:
    raise NameError('not known step type'+dataType)