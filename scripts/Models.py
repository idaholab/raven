'''
Created on Feb 19, 2013

@author: crisr
'''
import os
import copy
import shutil
import Datas
import numpy as np
from BaseType import BaseType
import SupervisionedLearning 
#import Postprocessors
#import ROM interfaces



######################################################################
#                       FILTER interface                             #
# NB. For readability this types should be moved to a separate module# 
######################################################################

class PrintCSV:
  def __init__(self):
    self.paramters = []
  def readMoreXML(self,xmlNode):
    param = ''
    param = xmlNode.text
    if(param.lower() != 'all'):
      self.paramters = param.strip().split(',')
    else:
      self.paramters.append(param) 
    return
  def finalizeFilter(self,inObj,outObj):
    # check the input type
    if(inObj.type == "HDF5"):
      endGroupNames = inObj.getEndingGroupNames()
      histories = {}
      for index in xrange(len(endGroupNames)):
        histories[endGroupNames[index]] = inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      
      try:
        outType = outObj.type
        #not yet implemented
      except:
#        splitted = outObj.split('.')
#        addfile = splitted[0] + '_additional_info.' + splitted[1]
#        with open(outObj, 'w') as csvfile, open(addfile, 'w') as addcsvfile:
        for key in histories:
          headers = ''
          attributes = histories[key][1]
          for i in xrange(len(attributes['headers'])):
            headers = headers + histories[key][1]['headers'][i] + ','
          try:
            hist = ''
            hist = key
            hist = hist.replace(',','_') 
          except:
            hist = key
          splitted = outObj.split('.')
          addfile = splitted[0] + '_additional_info_' + hist + '.'+splitted[1]
          csvfilen = splitted[0] + '_' + hist + '.'+splitted[1]
          with open(csvfilen, 'w') as csvfile, open(addfile, 'w') as addcsvfile:            
            np.savetxt(csvfile, histories[key][0], delimiter=",",header=headers,comments='history,' + hist +'\n')
            csvfile.write(' '+'\n')
            #process the attributes in a different csv file (different kind of informations)
            addcsvfile.write('history,'+hist+','+'\n')
            addcsvfile.write('________________________________,' + '_'*len(key)+','+'\n')
            addcsvfile.write('number of parameters,'+str(attributes['n_params'])+'\n')
            addcsvfile.write('parameters,'+headers+'\n') 
            addcsvfile.write('parent,'+str(attributes['parent_id'])+'\n') 
            addcsvfile.write('start time,'+str(attributes['start_time'])+'\n')
            addcsvfile.write('end time,'+str(attributes['end_time'])+'\n')
            addcsvfile.write('number of time-steps,'+str(attributes['n_ts'])+'\n')
            try:
              init_dist = attributes['initiator_distribution']
              addcsvfile.write('number of branches in this history,'+str(len(init_dist))+'\n')
              string_work = ''
              for i in xrange(len(init_dist)):
                string_work_2 = ''
                for j in init_dist[i]:
                  string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write('initiator distributions,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              end_ts = attributes['end_timestep']
              for i in xrange(len(end_ts)):
                string_work = string_work + str(end_ts[i]) + ','          
              addcsvfile.write('end time step,'+str(string_work)+'\n')
            except:
              pass             
            try:
              string_work = ''
              branch_changed_param = attributes['branch_changed_param']
              for i in xrange(len(branch_changed_param)):
                string_work_2 = ''
                for j in branch_changed_param[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','          
              addcsvfile.write('changed parameters,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              branch_changed_param_value = attributes['branch_changed_param_value']
              for i in xrange(len(branch_changed_param_value)):
                string_work_2 = ''
                for j in branch_changed_param_value[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                          
              addcsvfile.write('changed parameters values,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              cond_pbs = attributes['conditional_prb']
              for i in xrange(len(cond_pbs)):
                string_work_2 = ''
                for j in cond_pbs[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','                
              addcsvfile.write('conditional probability,'+str(string_work)+'\n')
            except:
              pass
            try:
              string_work = ''
              pb_thresholds = attributes['Probability_threshold']
              for i in xrange(len(pb_thresholds)):
                string_work_2 = ''
                for j in pb_thresholds[i]:
                  if not j:
                    string_work_2 = string_work_2 + 'None' + ' '
                  else:
                    string_work_2 = string_work_2 + str(j) + ' '
                string_work = string_work + string_work_2 + ','
              addcsvfile.write('Probability threshold,'+str(string_work)+'\n')
            except:
              pass            
            addcsvfile.write(' '+'\n')
            
    elif(inObj.type == "Datas"):
      pass
    else:
      raise NameError ('Filter PrintCSV for input type ' + inObj.type + ' not yet implemented.')

class Plot:
  def __init__(self):
    self.paramters = []
    
  def readMoreXML(self,xmlNode):
    param = ''
    param = xmlNode.text
    return
  
  def finalizeFilter(self,inObj,outObj):    
    if(inObj.type == "HDF5"):
      endGroupNames = inObj.getEndingGroupNames()
      histories = {}
      for index in xrange(len(endGroupNames)):
        histories[endGroupNames[index]] = inObj.returnHistory({'history':endGroupNames[index],'filter':'whole'})
      
    elif (inObj.type == "CSV"):
      # do something
      pass
    else:
      raise NameError ('Filter Plot for input type ' + inObj.type + ' not yet implemented.')
    
    for i in range (len(endGroupNames)):
      fig[i]=plt.figure()
      plt.plot(histories[endGroupNames[1]],histories[endGroupNames[i]])
      plt.xlabel('Time')
      plt.ylabel(histories[key][1]['headers'][i])
      plt.title('Plot of history:', i)
      if (outObj.type == "screen"):
        plt.show()
      elif (outObj.type == "jpeg"):
        fileName=str(histories[endGroupNames[i]])+'.jpeg'
        fig[i].savefig(fileName,dpi=fig.dpi)  # dpi=fig.dpi is to keep same same figure rendering of show() also for savefig()
      elif (outObj.type == "png"):
        fileName=str(histories[endGroupNames[i]])+'.png'
        fig[i].savefig(fileName,dpi=fig.dpi)
      elif (outObj.type == "eps"):
        fileName=str(histories[endGroupNames[i]])+'.eps'
        fig[i].savefig(fileName,dpi=fig.dpi)        
      elif (outObj.type == "pdf"):
        fileName=str(histories[endGroupNames[i]])+'.pdf'
        fig[i].savefig(fileName,dpi=fig.dpi)        
      else:
        raise NameError ('Filter Plot for output type ' + outObj.type + ' not implemented.')  
    return
        
def returnFilterInterface(Type):
  base = 'Filter'
  filterInterfaceDict = {}
  filterInterfaceDict['PrintCSV'] = PrintCSV
  filterInterfaceDict['Plot'] = Plot
  try: return filterInterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
class Model(BaseType):
  ''' a model is something that given an input will return an output reproducing some physical model
      it could as complex as a stand alone code or a reduced order model trained somehow'''
  def __init__(self):
    BaseType.__init__(self)
    self.subType  = ''
    self.runQueue = []  
  def readMoreXML(self,xmlNode):
    try: self.subType = xmlNode.attrib['type']
    except: raise 'missed type for the model'+self.name
  def addInitParams(self,tempDict):
    tempDict['subType'] = self.subType
  def reset(self,runInfo,inputs):
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step'''
    raise IOError('the model '+self.name+' that has no reset method' )
  def train(self,trainingSet,stepName):
    '''This needs to be over written if the model requires an initialization'''
    raise IOError('Step '+stepName+' tried to train the model '+self.name+' that has no training step' )
  def run(self):
    '''This call should be over loaded and return a jobHandler.External/InternalRunner'''
    raise IOError('the model '+self.name+' that has no run method' )
  def collectOutput(self,collectFrom,storeTo):
    storeTo.addOutput(collectFrom)
  def createNewInput(self,myInput,samplerType,**Kwargs):
    raise IOError('for this model the createNewInput has not yet being implemented')

class Code(Model):
  '''this is the generic class that import an external code into the framework'''
  def __init__(self):
    Model.__init__(self)
    self.executable         = ''   #name of the executable (abs path)
    self.oriInputFiles      = []   #list of the original input files (abs path)
    self.workingDir         = ''   #location where the code is currently running
    self.outFileRoot        = ''   #root to be used to generate the sequence of output files
    self.currentInputFiles  = []   #list of the modified (possibly) input files (abs path)
    self.infoForOut         = {}   #
  def readMoreXML(self,xmlNode):
    '''extension of info to be read for the Code(model)
    !!!!generate also the code interface for the proper type of code!!!!'''
    import CodeInterfaces
    Model.readMoreXML(self, xmlNode)
    try: 
      self.executable = xmlNode.text
      abspath = os.path.abspath(self.executable)
      if os.path.exists(abspath):
        self.executable = abspath
    except: raise IOError('not found executable '+xmlNode.text)
    self.interface = CodeInterfaces.returnCodeInterface(self.subType)
  def addInitParams(self,tempDict):
    '''extension of addInitParams for the Code(model)'''
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
  def addCurrentSetting(self,originalDict):
    '''extension of addInitParams for the Code(model)'''
    originalDict['current working directory'] = self.workingDir
    originalDict['current output file root']  = self.outFileRoot
    originalDict['current input file']        = self.currentInputFiles
    originalDict['original input file']       = self.oriInputFiles
  def reset(self,runInfoDict,inputFiles):
    '''initialize some of the current setting for the runs and generate the working 
       directory with the starting input files'''
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except: print('warning current working dir '+self.workingDir+'already exists, this might imply deletion of present files')
    for inputFile in inputFiles:
      shutil.copy(inputFile,self.workingDir)
    print('original input files copied in the current working dir: '+self.workingDir)
    print('files copied:')
    print(inputFiles)
    self.oriInputFiles = []
    for i in range(len(inputFiles)):
      self.oriInputFiles.append(os.path.join(self.workingDir,os.path.split(inputFiles[i])[1]))
    self.currentInputFiles        = None
    self.outFileRoot              = None
    return #self.oriInputFiles
  def createNewInput(self,currentInput,samplerType,**Kwargs):
    if currentInput[0].endswith('.i'): index = 0
    else: index = 1
    Kwargs['outfile'] = 'outFrom~'+os.path.split(currentInput[index])[1].split('.')[0]
    self.infoForOut[Kwargs['prefix']] = copy.deepcopy(Kwargs)
    return self.interface.createNewInput(currentInput,self.oriInputFiles,samplerType,**Kwargs)
  def run(self,inputFiles,outputDatas,jobHandler):
    '''return an instance of external runner'''
    self.currentInputFiles = inputFiles
    executeCommand, self.outFileRoot = self.interface.generateCommand(self.currentInputFiles,self.executable)
#    for inputFile in self.currentInputFiles: shutil.copy(inputFile,self.workingDir)
    self.process = jobHandler.submitDict['External'](executeCommand,self.outFileRoot,jobHandler.runInfoDict['TempWorkingDir'])
    if self.currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    print('job "'+ inputFiles[index].split('/')[-1].split('.')[-2] +'" submitted!')
    return self.process
  def collectOutput(self,finisishedjob,output):
    '''collect the output file in the output object'''
    if output.type == "HDF5":
      self.__addDataSetGroup(finisishedjob,output)
    else:
      output.addOutput(os.path.join(self.workingDir,finisishedjob.output) + ".csv")
    return
  def __addDataSetGroup(self,finisishedjob,dataset):
    # add a group into the database
    attributes={}
    attributes["input_file"] = self.currentInputFiles
    attributes["type"] = "csv"
    attributes["name"] = os.path.join(self.workingDir,finisishedjob.output+'.csv')
    if self.infoForOut.has_key(finisishedjob.identifier):
      infoForOut = self.infoForOut.pop(finisishedjob.identifier)
      for key in infoForOut:
        attributes[key] = infoForOut[key]
    dataset.addGroup(attributes,attributes)

class ROM(Model):
  '''ROM stands for Reduced Order Models. All the models here, first learn than predict the outcome'''
  def __init__(self):
    Model.__init__(self)
    self.initializzationOptionDict = {}
  def readMoreXML(self,xmlNode):
    '''read the additional input needed and istanziate the underlying ROM'''
    Model.readMoreXML(self, xmlNode)
    for child in xmlNode:
      try: self.initializzationOptionDict[child.tag] = int(child.text)
      except:
        try: self.initializzationOptionDict[child.tag] = float(child.text)
        except: self.initializzationOptionDict[child.tag] = child.text
    self.test =  SupervisionedLearning.returnInstance(self.subType)
    self.SupervisedEngine = self.test(**self.initializzationOptionDict)   
    #self.SupervisedEngine = SupervisionedLearning.returnInstance(self.subType)(self.initializzationOptionDict) #create an instance of the ROM
  def addInitParams(self,originalDict):
    ROMdict = self.SupervisedEngine.returnInitialParamters()
    for key in ROMdict.keys():
      originalDict[key] = ROMdict[key]
  def addCurrentSetting(self,originalDict):
    ROMdict = self.SupervisedEngine.returnCurrentSetting()
    for key in ROMdict.keys():
      originalDict[key] = ROMdict[key]
  def reset(self):
    self.SupervisedEngine.reset()
#  def run(self):
#    return
#  def collectOutput(self,collectFrom,storeTo):
#    storeTo.addOutput(collectFrom)
#  def createNewInput(self,myInput,samplerType,**Kwargs):
#    raise IOError('for this model the createNewInput has not yet being implemented')




class Filter(Model):
  '''Filter is an Action System. All the models here, take an input and perform an action'''
  def __init__(self):
    Model.__init__(self)
    self.input  = {}     # input source
    self.action = None   # action
  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    self.interface = returnFilterInterface(self.subType)
    self.interface.readMoreXML(xmlNode)
  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)
  def run(self,inObj,outObj):
    '''run calls the interface finalizer'''
    self.interface.finalizeFilter(inObj,outObj)
#    def __returnInputTypeInterface(type):
#      base = 'input'
#      InputInterfaceDict = {}
#      InputInterfaceDict['DataSets'   ] = DataSetsInterface
#
#      try: return InputInterfaceDict[Type]()
#      except: raise NameError('not known '+base+' type '+Type + 'in Filter Model.')
      

def returnInstance(Type):
  '''This function return an instance of the request model type'''
  base = 'model'
  InterfaceDict = {}
  InterfaceDict['ROM'   ] = ROM
  InterfaceDict['Code'  ] = Code
  InterfaceDict['Filter'] = Filter
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
