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
import numpy as np
import Quadrature
import Distributions

class Sampler(BaseType):
  ''' this is the base class for samplers'''
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
    tempDict['limit' ]        = self.limit
  def addCurrentSetting(self,tempDict):
    tempDict['counter' ] = self.counter    
  def initialize(self):
    self.counter = 0
  def amIreadyToProvideAnInput(self):
    '''this check if the sampler is ready to generate a new input
       it might not since it is waiting for more information or has 
       run out of limit, depends from the type of sampler'''
    if(self.counter <= self.limit): return True
    else: return False
  def fillDistribution(self,availableDist):
    '''generate the instances of the distribution that will be used'''
    for key in self.toBeSampled.keys():
      self.distDict[key] = availableDist[self.toBeSampled[key][1]].inDistr()
    return
  def finalizeActualSampling(self,jobObject,model,myInput):
    '''???'''
    pass
  def generateInput(self):
    '''override this method to place your input generation'''
    raise IOError('This sampler has not an input generation?!?!')
    return
  def generateInputBatch(self,myInput,model,batchSize):
    '''for the first set of run a set of run are started in // so we need more than one input '''
    newInputs = []
    while self.amIreadyToProvideAnInput() and (self.counter < batchSize):
      newInputs.append(self.generateInput(model,myInput))
    return newInputs


class StochasticCollocation(Sampler):
  def __init__(self):
    Sampler.__init__(self)
    self.min_poly_order = 0 #lowest acceptable polynomial order
    self.var_poly_order = dict() #stores poly orders for each var
    self.counter_ord_dict={} #dict of counter versus ordinates used
  def readMoreXML(self,xmlNode):
    # attempt to set minimum total function polynomial expansion order
    try: self.min_poly_order = int(xmlNode.attrib['min_poly_order'])
    except: self.min_poly_order = 0
    # attempt to set polynomial expansion order for individual params
    for var in self.distDict.keys():
      try:
        self.var_poly_order[var] = int(xmlNode.attrib[var+'_poly_order'])
      except: pass
    # assign values to undefined order variables
    if self.min_poly_order>0:
      if len( set(self.var_poly_order) ^ set(self.distDict) )==0: #keys correspond exactly
        if sum(self.var_poly_order.values())<self.min_poly_order:
          raise IOError('Minimum total polynomial order is set greater than sum of all variable orders!')
      else:
        r_keys = set(self.distDict) - set(self.var_poly_order) #var orders not set yet
        r_order = min(len(r_keys),self.min_poly_order-sum(self.var_poly_order.values())) #min remaining order needed
        for key in r_keys: #ones that haven't been set yet
          self.var_poly_order['key']=int(round(0.5+r_order/(len(r_keys))))
    self.generateQuadrature() #FIXME is this where this should go?

  def generateInput(self,model,myInput):
    self.counter+=1
    qps=self.quad.ords[counter-1]
    qp_index = self.quad.ord_index[qps]
    values={'prefix':str(self.counter),'qp indices':str(qp_index)}
    try:
      for var in self.distDict.keys():
        values[var]=self.distDict[var].standardToActualPoint(\
            qps[self.quad.indx_quads[self.distDict[var].quad()]])
      # qps is a tuple of gauss quad points, so use the variable's distribution's quadrature
      #   to look up the index for the right variable,
      #   then use dist.standardToActualPoint to convert the gauss point to a parameter value
      # TODO we could also pass "var" as an argument to the quadrature to make indexing look a lot nicer
    except StopIteration: raise 'No Gauss points left to iterate over!'
    return model.createNewInput(myInput,self.type,**values)

  def generateQuadrature(self):
    quadDict={}
    quadDict[Distributions.Uniform().type]=Quadrature.Legendre
    quadDict[Distributions.Normal().type]=Quadrature.StatHermite
    #TODO need alpha and/or beta values for these two...todo?
    quadDict[Distributions.Gamma().type]=Quadrature.Laguerre
    quadDict[Distributions.Beta().type]=Quadrature.Jacobi
    quads=[]
    for var in self.distDict.keys():
      #TODO see above, this won't work for quads that need addl params
      #  create a dict for addl params lists to *add to quadrature init call?
      #  this for sure works, even if it's empty!
      quads.append(quadDict[self.distDict[var].type](self.var_poly_order[var]))
      self.distDict[var].setQuad(quads[-1],self.var_poly_order[var])
    self.quad=Quadrature.MultiQuad(quads)



class MonteCarlo(Sampler):
  def __init__(self):
    Sampler.__init__(self)
    self.limit       = 0        #maximum number of samples it will perform every time it is used
    self.init_seed   = 0
  def readMoreXML(self,xmlNode):
    try: self.init_seed    = xmlNode.attrib['initial_seed']
    except: self.init_seed = 0 
    try: self.limit    = xmlNode.attrib['limit']
    except: raise IOError(' Monte Carlo sampling needs the attribute limit (number of samplings)')
  def addInitParams(self,tempDict):
    Sampler.addInitParams(self,tempDict)
    tempDict['initial seed' ] = self.init_seed
  def generateInput(self,model,myInput):
    '''returns the model.createNewInput() passing into it the type of sampler,
       the values to be used and the some add info in the values dict'''
    self.counter += 1
    values = {'prefix':str(self.counter),'initial_seed':str(self.init_seed)}
    #evaluate the distributions and fill values{}
    for key in self.distDict:
      values[key] = self.distDict[key].distribution.rvs()
    return model.createNewInput(myInput,self.type,**values)




class LatinHyperCube(Sampler):
  '''implement a latin hyper cube approach only with random picking of the intervals'''
  def __init__(self):
    Sampler.__init__(self)
    self.limit        = 0        #maximum number of sampler it will perform every time it is used
    self.grid         = []       #is a list that for each point in the grid return a dictionary of the distributions where values are the bounds in terms of the random variate
  def addInitParams(self,tempDict):
    Sampler.addInitParams(self,tempDict)
    tempDict['initial seed' ] = self.init_seed
  def addCurrentSetting(self,tempDict):
    i = 0
    for distribution in self.distDict.keys():
      tempDict['interval '+ int(i) + ', distribution ' +distribution+' is in range'] = self.grid[i].distBounds[distribution]
  def initialize(self):
    Sampler.initialize(self)
    self.grid = [None]*self.limit
    nDimension = len(self.distDict)
    takenGlobal = np.zeros((self.limit,nDimension),ndmin=2,dtype=int)
    distList = self.distDict.keys()
    for i in range(self.grid):
      self.grid[i] = dict.fromkeys(self.distDict.keys(),[None]*2)
    for j in range(self.limit):
      for i in range(nDimension):
        placed = False
        while placed == False:
          indexInterval = int(np.random.rand(1)*self.limit)
          if takenGlobal[indexInterval][i] == 0:
            takenGlobal[indexInterval][i] = 1
            distName = distList[i]
            #if equally spaced do not use ppt
            lowerBound = self.distDict[distName].ppt(float((indexInterval-1)/self.limit))
            upperBound = self.distDict[distName].ppt(float((indexInterval)/self.limit))
            self.grid[j].distBounds[distName] = [lowerBound,upperBound]
  def generateInput(self,model,myInput):
    '''returns the model.createNewInput() passing into it the type of sampler,
       the values to be used and the some add info in the values dict'''
    self.counter += 1
    values = {'prefix':str(self.counter),'initial_seed':str(self.init_seed)}
    #evaluate the distributions and fill values{}
    for key in self.distDict:
      upper = self.grid[self.counter][key][1]
      lower = self.grid[self.counter][key][1]
      values[key] = [self.distDict[key].distribution.rvsWithinbounds(lower,upper),lower,upper]
    return model.createNewInput(myInput,self.type,**values)

   
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
    if(len(self.RunQueue['queue']) != 0 or self.counter == 0):
      return True
    else:
      return False
  
  def finalizeActualSampling(self,jobObject,model,myInput):
    # we read the info at the end of one branch
    self.workingDir = model.workingDir
    if not self.__readBranchInfo(jobObject.output): return
    
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
             unchanged_pb = unchanged_pb + endInfo['branch_changed_params'][key]['associated_pb'][pb]
         except:
          pass
         if(unchanged_pb <= 1):
           endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0-unchanged_pb
      
      else:
        # two way branch
        if self.branchedLevel[endInfo['branch_dist']] > len(self.branchProbabilities[endInfo['branch_dist']])-1:
          pb = 1.0
        else:
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

    self.branchedLevel[endInfo['branch_dist']]       += 1
    
    self.endInfo.append(endInfo)
    # compute conditional probability calculation (put the result into self.endInfo)
    self.computeConditionalProbability()

    # we create the input queue for all the branches must be run
    self.__createRunningQueue(model,myInput)
    
    return
  def computeConditionalProbability(self,index=None):
    if not index:
      index = len(self.endInfo)-1
    parent_cond_pb = 0.0  
    try:
      parent_cond_pb = self.endInfo[index]['parent_node'].get('conditional_pb')
      if not parent_cond_pb:
        parent_cond_pb = 1.0
    except:
      parent_cond_pb = 1.0
      
    for key in self.endInfo[index]['branch_changed_params']:
       try:
         testpb = self.endInfo[index]['branch_changed_params'][key]['unchanged_pb']
         self.endInfo[index]['branch_changed_params'][key]['unchanged_cond_pb'] = parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['unchanged_pb'])
         for pb in xrange(len(self.endInfo[index]['branch_changed_params'][key]['associated_pb'])):
           self.endInfo[index]['branch_changed_params'][key]['changed_cond_pb'] = parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['associated_pb'][pb])
       except:
         pass
    return  
  def __readBranchInfo(self,out_base=None):
    # function for reading Branch Info from xml file

    # we remove all the elements from the info container
    del self.actualBranchInfo
    self.actualBranchInfo = {}
    if out_base:
      filename = out_base + "_actual_branch_info.xml"
    else:
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
      # we exit the loop here, because only one trigger at the time can be handled  right now   
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

        cond_pb_un = 0.0
        cond_pb_c =0.0
        for key in endInfo['branch_changed_params'].keys():
          subGroup.set('branch_changed_param',key)
          if self.branchCountOnLevel != 1:
            subGroup.set('branch_changed_param_value',endInfo['branch_changed_params'][key]['actual_value'][self.branchCountOnLevel-2])
            subGroup.set('branch_changed_param_pb',endInfo['branch_changed_params'][key]['associated_pb'][self.branchCountOnLevel-2])
            try:
              cond_pb_c = cond_pb_c + endInfo['branch_changed_params'][key]['changed_cond_pb']
            except:
              pass
          else:
            subGroup.set('branch_changed_param_value',endInfo['branch_changed_params'][key]['old_value'])
            subGroup.set('branch_changed_param_pb',endInfo['branch_changed_params'][key]['unchanged_pb'])
            try:
              cond_pb_un =  cond_pb_un + endInfo['branch_changed_params'][key]['unchanged_cond_pb']
            except:
              pass
        if self.branchCountOnLevel != 1:
          subGroup.set('conditional_pb',cond_pb_c)
        else:
          subGroup.set('conditional_pb',cond_pb_un)
          
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
                  'conditional_prb':[subGroup.get('conditional_pb')],
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
      values['initiator_distribution']     = []
      values['PbThreshold']                = []
      values['branch_changed_param']       = ['None']
      values['branch_changed_param_value'] = ['None']
      values['start_time']                 = 'Initial'
      values['end_ts']                     = 0
      values['conditional_prb']            = [1.0]
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
    try:
      flag = ""
      flag = xmlNode.attrib['print_end_xml']
      self.print_end_xml = (flag.lower() in ['true','t','yes','si','y','yeah','ja','da','oui','sic','perche no','avojia','certamente','dajie','divertimose'])
    except:
      self.print_end_xml = False
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
  InterfaceDict['StochasticCollocation'] = StocasticCollocation
  InterfaceDict['MonteCarlo'       ] = MonteCarlo
  InterfaceDict['LatinHyperCube'   ] = LatinHyperCube
  InterfaceDict['EquallySpaced'    ] = EquallySpaced
  InterfaceDict['DynamicEventTree' ] = DynamicEventTree
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
