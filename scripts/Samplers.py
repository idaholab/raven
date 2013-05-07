'''
Created on Mar 8, 2013

@author: crisr
'''
import sys
import time
import Datas
from BaseType import BaseType
import xml.etree.ElementTree as ET
import os
import Queue
import copy

class Sampler(BaseType):
  ''' 
  this is the base class for samplers
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.counter = 0
    self.limit   = sys.maxint
    self.workingDir = ""
    self.toBeSampled = {}  #key=feature to be sampled, value = ['type of distribution to be used', 'name of the distribution']
    self.distDict    = {}  #contain the instance of the distribution to be used, it is created every time the sampler is initialize
  
  def readMoreXML(self,xmlNode):
    for child in xmlNode:
      self.toBeSampled[child.text] = [child.attrib['type'],child.attrib['distName']]
  
  def addInitParams(self,tempDict):
    for value in self.toBeSampled.items():
      tempDict[value[0]] = value[1][0]+':'+value[1][1]
  
  def addCurrentSetting(self,tempDict):
    pass
  def initialize(self):
    self.counter = 0
    
  def amIreadyToProvideAnInput(self):
    if(self.counter <= self.limit):
      return True
    else:
      return False
    
  def fillDistribution(self,availableDist):
    for key in self.toBeSampled.keys():
      self.distDict[key] = availableDist[self.toBeSampled[key][1]].inDistr()
    return
  
  def finalizeActualSampling(self,jobObject,model,myInput):
    pass
  
  def generateInputBatch(self,myInput,model,batchSize):
    try:
      if batchSize<=self.limit: newInputs = [None]*batchSize
      else:newInputs = [None]*self.limit
      for i in range(len(newInputs)):
        newInputs[i]=self.generateInput(model,myInput)
    except:
      newInputs = [None]*batchSize
      for i in range(len(newInputs)):
        newInputs[i]=self.generateInput(model,myInput)          
    return newInputs


class MonteCarlo(Sampler):
  def __init__(self):
    Sampler.__init__(self)
    self.limit       = 0        #maximum number of sampler it will perform every time it is used
  def generateInput(self,model,myInput):
    self.counter += 1
    values = {'prefix':str(self.counter)}
    for key in self.distDict:
      values[key] = self.distDict[key].distribution.rvs()
    return model.createNewInput(myInput,self.type,**values)
  def readMoreXML(self,xmlNode):
    try: self.limit = xmlNode.attrib['limit']
    except: raise IOError('not found limit for the Sampler '+self.name)
    return
  def addInitParams(self,tempDict):
    tempDict['limit' ] = self.limit  
  def addCurrentSetting(self,tempDict):
    tempDict['counter' ] = self.counter    
class LatinHyperCube(Sampler):
  def __init__(self):
    Sampler.__init__(self)
    self.limit       = 0        #maximum number of sampler it will perform every time it is used
   
class EquallySpaced(Sampler):
  pass

class DynamicEventTree(Sampler):
  def __init__(self):
    Sampler.__init__(self)
    #optional value... Conditional Probability Cut. If the Probability falls below this value the associated branch is terminated    
    self.CP_cut                  = None
    self.maxSimulTime            = None #(optional) if not present, the sampler will not change the relative keyword in the input file
    self.print_end_xml           = False
    self.branchProbabilities     = {}
    self.branchedLevel           = {}
    self.branchCountOnLevel      = 0
    # actual branch info
    self.actualBranchInfo        = {}
    self.actual_end_time         = 0.0
    self.actual_end_ts           = 0
    # here we store all the info regarding the DET => we create the info for all the
    # branchings and we store them
    self.TreeInfo                = None    
    self.endInfo                 = []
    self.branchCountOnLevel      = 0
    # this dictionary contains the inputs(i.e. the info to create them) are waiting to be run
    self.RunQueue                = {}
    self.RunQueue['identifiers'] = []
    self.RunQueue['queue'      ] = []

  def amIreadyToProvideAnInput(self):
    if(len(self.RunQueue['queue']) != 0):
      return True
    else:
      return False

  def  computeConditionalProbability(self):
    return
  
  def finalizeActualSampling(self,jobObject,model,myInput):
    # we read the info at the end of one branch
    self.workingDir = model.workingDir
    if not self.__readBranchInfo(): return
    
    # we collect the info in a multi-level dictionary
    endInfo = {}
    endInfo['end_time']               = self.actual_end_time
    endInfo['end_ts']                 = self.actual_end_ts
    endInfo['branch_dist']            = self.actualBranchInfo.keys()[0]
    endInfo['branch_changed_params']  = self.actualBranchInfo[endInfo['branch_dist']]
      
    for key in endInfo['branch_changed_params']:
      endInfo['n_branches'] = 1 + int(len(endInfo['branch_changed_params'][key]['actual_value']))
      if(len(endInfo['branch_changed_params'][key]['actual_value']) > 1):
        # multi-branch situation
         unchanged_pb = 0.0
         try:
           for pb in xrange(len(endInfo['branch_changed_params'][key]['associated_pb'])):
             unchanged_pb = unchanged_pb + pb 
         except:
          pass
         if(unchanged_pb <= 1):
           endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0-unchanged_pb
      
      else:
        # two way branch
        pb = self.branchProbabilities[endInfo['branch_dist']][self.branchedLevel[endInfo['branch_dist']]]
        endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0 - pb
        endInfo['branch_changed_params'][key]['associated_pb'] = [pb]
    if(jobObject.identifier == self.TreeInfo.getroot().tag):
      endInfo['parent_node'] = self.TreeInfo.getroot()
    else:
      endInfo['parent_node'] = list(self.TreeInfo.getroot().iter(jobObject.identifier))[0]  
    self.branchCountOnLevel = 0
    # set runEnded and running to true and false respectively   
    endInfo['parent_node'].set('runEnded',True)
    endInfo['parent_node'].set('running',False)
    endInfo['parent_node'].set('end_time',self.actual_end_time)
    # add call to conditional probability calculation
    self.computeConditionalProbability()
    self.branchedLevel[endInfo['branch_dist']]       += 1

    self.endInfo.append(endInfo)
    # we create the input queue for all the branches must be run
    self.__createRunningQueue(model,myInput)
    
    return
  
  def __readBranchInfo(self):
    # function for reading Branch Info from xml file

    # we remove all the elements from the info container
    del self.actualBranchInfo
    self.actualBranchInfo = {}
    filename = "actual_branch_info.xml"

    if not os.path.isabs(filename):
      filename = os.path.join(self.workingDir,filename)
    if not os.path.exists(filename):
      print('branch info file' + filename +' has not been found. => No Branching.')
      branch_present = False
      return branch_present
    try:
      branch_info_tree = ET.parse(filename)
    except:
      branch_info_tree = ET.parse(filename)
      raise IOError ('not able to parse ' + filename)
    root = branch_info_tree.getroot()

    try:
      self.actual_end_time = float(root.attrib['end_time'])
      self.actual_end_ts   = int(root.attrib['end_ts'])
    except:
      pass

    for node in root:
      if node.tag == "Distribution_trigger":
        dist_name = node.attrib['name'].strip()
        self.actualBranchInfo[dist_name] = {}
        for child in node:
          self.actualBranchInfo[dist_name][child.text.strip()] = {}
          self.actualBranchInfo[dist_name][child.text.strip()]['varType'] = child.attrib['type'].strip()
          self.actualBranchInfo[dist_name][child.text.strip()]['actual_value'] = []
          self.actualBranchInfo[dist_name][child.text.strip()]['actual_value'].append(child.attrib['actual_value'].strip())
          self.actualBranchInfo[dist_name][child.text.strip()]['old_value'] = child.attrib['old_value'].strip()
          try:
            as_pb = child.attrib['pb'].strip()
            self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'] = []
            self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'].append(float(as_pb)) 
          except:
            pass
      # we exit the loop here, because only one trigger at the time can be handled     
      break
    # we remove the file
    os.remove(filename)
    branch_present = True
    return branch_present 
  
  def __createRunningQueue(self,model,myInput):
    
    if self.counter >= 1:
      endInfo = self.endInfo.pop(0)
      for i in xrange(endInfo['n_branches']):
        self.counter += 1
        self.branchCountOnLevel += 1
        rname = endInfo['parent_node'].get('name') + ',' + str(self.branchCountOnLevel)
        subGroup = ET.Element(rname)
        subGroup.set('parent', endInfo['parent_node'].get('name'))
        subGroup.set('name', rname)

        for key in endInfo['branch_changed_params'].keys():
          subGroup.set('branch_changed_param',key)
          if self.branchCountOnLevel != 1:
            subGroup.set('branch_changed_param_value',endInfo['branch_changed_params'][key]['actual_value'][self.branchCountOnLevel-2])
            subGroup.set('branch_changed_param_pb',endInfo['branch_changed_params'][key]['associated_pb'][self.branchCountOnLevel-2])
          else:
            subGroup.set('branch_changed_param_value',endInfo['branch_changed_params'][key]['old_value'])
            subGroup.set('branch_changed_param_pb',endInfo['branch_changed_params'][key]['unchanged_pb'])            
        
        subGroup.set('initiator_distribution',endInfo['branch_dist']) 
        subGroup.set('start_time', endInfo['parent_node'].get('end_time'))
        # we initialize the end_time to be equal to the start one... It will modified at the end of this branch
        subGroup.set('end_time', endInfo['parent_node'].get('end_time'))
        subGroup.set('runEnded',False)
        subGroup.set('running',False)
        subGroup.set('queue',True)
#        subGroup.set('restartFileRoot',endInfo['restartRoot'])
        endInfo['parent_node'].append(subGroup)

        values = {'prefix':rname,'end_ts':endInfo['end_ts'],
                  'branch_changed_param':[subGroup.get('branch_changed_param')],
                  'branch_changed_param_value':[subGroup.get('branch_changed_param_value')],
                  'initiator_distribution':[endInfo['branch_dist']],
                  'start_time':endInfo['parent_node'].get('end_time'),
                  'parent_id':subGroup.get('parent')}
        if self.branchedLevel[endInfo['branch_dist']] >= len(self.branchProbabilities[endInfo['branch_dist']]):
          #we set the threshold to 1.1 => no branch possible for this dist anymore.
          values['PbThreshold'] = [1.1]
        else:
          values['PbThreshold'] = [self.branchProbabilities[endInfo['branch_dist']][self.branchedLevel[endInfo['branch_dist']]]]
        
        self.RunQueue['queue'].append(copy.deepcopy(model.createNewInput(myInput,self.type,**values)))
        self.RunQueue['identifiers'].append(values['prefix'])
        del values
    else:
      self.counter += 1
      rname = self.TreeInfo.getroot().tag 
      values = {'prefix':rname}
      values['initiator_distribution'] = []
      values['PbThreshold']            = []
      for key in self.distDict.keys():
        values['initiator_distribution'].append(key)
      for key in self.branchProbabilities.keys():  
        values['PbThreshold'].append(self.branchProbabilities[key][self.branchedLevel[key]])
      if(self.maxSimulTime): values['end_time'] = self.maxSimulTime
      newInputs = model.createNewInput(myInput,self.type,**values)
      self.RunQueue['queue'].append(newInputs)
      self.RunQueue['identifiers'].append(values['prefix'])
      del values
      del newInputs
      
    return  
  
  def __getQueueElement(self):
    if len(self.RunQueue['queue']) == 0:
      # there are no more runs must be run
      # we set the self.limit == self.counter
      # => the simulation ends
      self.limit = self.counter
      return None
    else:
      jobInput = self.RunQueue['queue'].pop(0)
      id       = self.RunQueue['identifiers'].pop(0)
      #set running flags in self.TreeInfo
      root = self.TreeInfo.getroot()
      if (root.tag == id):
        root.set('runEnded',False)
        root.set('running',True)
        root.set('queue',False)
      else:
        subElm = list(root.iter(id))[0]
        if(subElm is not None):
          subElm.set('runEnded',False)
          subElm.set('running',True)
          subElm.set('queue',False)

    return jobInput
    
  def generateInput(self,model,myInput):
    if self.counter <= 1:
      self.__createRunningQueue(model, myInput)
      
    input = self.__getQueueElement()
    if not input:
      print("A Branch ended!!!!")
    return input
    
  def readMoreXML(self,xmlNode):
    elm = ET.Element(xmlNode.attrib['name'] + '_1')
    flag = ""
    flag = xmlNode.attrib['print_end_xml']
    self.print_end_xml = (flag.lower() in ['true','t','yes','si','y','yeah','ja','da','oui','sic','perche no','avojia','certamente','dajie','divertimose'])
    #elm.set('parent', 'root')
    elm.set('name', xmlNode.attrib['name'] + '_1')
    elm.set('start_time', 0.0)
    # we initialize the end_time to be equal to the start one... 
    # It will modified at the end of this branch
    elm.set('end_time', 0.0)
    elm.set('runEnded',False)
    elm.set('running',True)
    elm.set('queue',False)
    # here we store all the info regarding the DET => we create the info for all the
    # branchings and we store them
    self.TreeInfo = ET.ElementTree(elm)    
    
    childreen = xmlNode.find("BranchingSettings")
    try: self.CP_cut = childreen.attrib['CPcut']
    except: self.CP_cut = None
    try: self.maxSimulTime = childreen.attrib['maxSimulationTime']
    except: self.maxSimulTime = None
    Sampler.readMoreXML(self,childreen)
    for child in childreen:
      bv = child.attrib['BranchProbs']
      bvalues = [float(x) for x in bv.split()]
      self.branchProbabilities[child.attrib['distName']] = bvalues
      self.branchedLevel[child.attrib['distName']]       = 0
#function used to generate a Model class
def returnInstance(Type):
  base = 'Sampler'
  InterfaceDict = {}
  InterfaceDict['MonteCarlo'       ] = MonteCarlo
  InterfaceDict['LatinHyperCube'   ] = LatinHyperCube
  InterfaceDict['EquallySpaced'    ] = EquallySpaced
  InterfaceDict['DynamicEventTree' ] = DynamicEventTree
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
  
  
  