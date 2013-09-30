'''
Created on Feb 19, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import os,subprocess
import Steps
import Datas
import Samplers
import Models
import Tests
import Distributions
import DataBases
import OutStreams
import math
from JobHandler import JobHandler

class SimulationMode:
  """SimulationMode allows changes to the how the simulation 
  runs are done.  modifySimulation lets the mode change runInfoDict
  and other parameters.  runOverride lets the mode do the running instead
  of simulation. """
  def __init__(self,simulation):
    self.__simulation = simulation
    
  def doOverrideRun(self):
    """If doOverrideRun is true, then use runOverride instead of 
    running the simulation normally.  This method should call 
    simulation.run somehow
    """
    return False

  def runOverride(self):
    """this can completely override the Simulation's run method"""
    pass

  def modifySimulation(self):
    """modifySimulation is called after the runInfoDict has been setup.
    This allows the mode to change any parameters that need changing.
    """
    import multiprocessing
    try:
      if multiprocessing.cpu_count() < self.__simulation.runInfoDict['batchSize']:
        print("WARNING cpu_count",multiprocessing.cpu_count()," < batchSize ",self.__simulation.runInfoDict['batchSize'])
    except:
      pass

class PBSSimulationMode(SimulationMode):
  def __init__(self,simulation):
    self.__simulation = simulation
    self.__in_pbs = "PBS_NODEFILE" in os.environ
    
  def doOverrideRun(self):
    # Check if the simulation has been run in PBS mode and, in case, construct the proper command    
    return not self.__in_pbs

  def runOverride(self):
    assert self.__simulation.runInfoDict['mode'] == 'pbs' and not self.__in_pbs
    # Check if the simulation has been run in PBS mode and, in case, construct the proper command
    batchSize = self.__simulation.runInfoDict['batchSize']
    frameworkDir = self.__simulation.runInfoDict["FrameworkDir"]
    ncpus = self.__simulation.runInfoDict['ParallelProcNumb']
    command = ["qsub","-l",
               "select="+str(batchSize)+":ncpus="+str(ncpus)+":mpiprocs=1",
               "-l","walltime="+self.__simulation.runInfoDict["expectedTime"],
               "-l","place=free","-v",
               'COMMAND="python Driver.py '+
               self.__simulation.runInfoDict["SimulationFile"]+'"',
               os.path.join(frameworkDir,"raven_qsub_command.sh")]
    os.chdir(frameworkDir)
    print(os.getcwd(),command)
    subprocess.call(command)

  def modifySimulation(self):
    if self.__in_pbs:
      #Figure out number of nodes and use for batchsize
      nodefile = os.environ["PBS_NODEFILE"]
      lines = open(nodefile,"r").readlines()
      oldBatchsize =  self.__simulation.runInfoDict['batchSize']
      newBatchsize = len(lines)
      if newBatchsize != oldBatchsize:
        self.__simulation.runInfoDict['batchSize'] = newBatchsize
        print("WARNING: changing batchsize from",oldBatchsize,"to",newBatchsize)
      print("DRIVER        : Using Nodefile to set batchSize:",self.__simulation.runInfoDict['batchSize'])
      self.__simulation.runInfoDict['precommand'] = "pbsdsh -v -n %INDEX1% -- %FRAMEWORK_DIR%/raven_remote.sh out_%CURRENT_ID% %WORKING_DIR% "+self.__simulation.runInfoDict['precommand']
      if(self.__simulation.runInfoDict['ParallelProcNumb'] > 1):
        self.__simulation.runInfoDict['postcommand'] = " --n-threads=%NUM_CPUS% "+self.__simulation.runInfoDict['postcommand']

class MPISimulationMode(SimulationMode):
  def __init__(self,simulation):
    self.__simulation = simulation
    self.__in_pbs = "PBS_NODEFILE" in os.environ

  def modifySimulation(self):
    if self.__in_pbs:
      #Figure out number of nodes and use for batchsize
      nodefile = os.environ["PBS_NODEFILE"]
      lines = open(nodefile,"r").readlines()
      numNode = self.__simulation.runInfoDict['numNode']
      oldBatchsize = self.__simulation.runInfoDict['batchSize']
      newBatchsize = max(int(math.floor(len(lines)/numNode)),1)
      if newBatchsize != oldBatchsize:
        self.__simulation.runInfoDict['batchSize'] = newBatchsize
        print("WARNING: changing batchsize from",oldBatchsize,"to",newBatchsize)
      if newBatchsize > 1:
        #need to split node lines
        workingDir = self.__simulation.runInfoDict['WorkingDir']
        for i in range(newBatchsize):
          node_file = open(os.path.join(workingDir,"node_"+str(i)),"w")
          for line in lines[i*numNode:(i+1)*numNode]:
            node_file.write(line)
          node_file.close()
        #then give each index a separate file.
        nodeCommand = "-f %BASE_WORKING_DIR%/node_%INDEX% "
      else:
        #If only one batch just use original node file
        nodeCommand = "-f $PBS_NODEFILE "
    else:
      #Not in PBS, so can't look at PBS_NODEFILE
      newBatchsize = self.__simulation.runInfoDict['batchSize']
      numNode = self.__simulation.runInfoDict['numNode']
      nodeCommand = " "

    self.__simulation.runInfoDict['precommand'] = "mpiexec "+nodeCommand+" -n "+str(numNode)+" "+self.__simulation.runInfoDict['precommand']
    if(self.__simulation.runInfoDict['ParallelProcNumb'] > 1):
      self.__simulation.runInfoDict['postcommand'] = " --n-threads=%NUM_CPUS% "+self.__simulation.runInfoDict['postcommand']
    print("precommand",self.__simulation.runInfoDict['precommand'],"postcommand",self.__simulation.runInfoDict['postcommand'])

    

class Simulation:
  '''This is a class that contain all the object needed to run the simulation'''
  def __init__(self,inputfile,frameworkDir,debug=False):
    self.debug= debug
    #this dictionary contains the general info to run the simulation
    self.runInfoDict = {}
    self.runInfoDict['SimulationFile'    ] = inputfile    #the xml input file
    self.runInfoDict['ScriptDir'         ] = os.path.join(os.path.dirname(frameworkDir),"scripts") # the location of the pbs script interfaves
    self.runInfoDict['FrameworkDir'      ] = frameworkDir # the directory where the framework is located
    self.runInfoDict['WorkingDir'        ] = ''           # the directory where the framework should be running
    self.runInfoDict['TempWorkingDir'    ] = ''           # the temporary directory where a simulation step is run
    self.runInfoDict['ParallelProcNumb'  ] = 1            # the number of mpi process by run
    self.runInfoDict['ThreadingProcessor'] = 1            # Number of Threats by run
    self.runInfoDict['numProcByRun'      ] = 1            # Total number of core used by one run (number of threats by number of mpi)
    self.runInfoDict['batchSize'         ] = 1            # number of contemporaneous runs
    self.runInfoDict['ParallelCommand'   ] = ''           # the command that should be used to submit jobs in parallel (mpi)
    self.runInfoDict['ThreadingCommand'  ] = ''           # the command should be used to submit multi-threaded  
    self.runInfoDict['numNode'           ] = 1            # number of nodes
    self.runInfoDict['procByNode'        ] = 1            # number of processors by node
    self.runInfoDict['totNumbCores'      ] = 1            # total number of cores available
    self.runInfoDict['quequingSoftware'  ] = ''           # quequing software name 
    self.runInfoDict['stepName'          ] = ''           # the name of the step currently running
    self.runInfoDict['precommand'        ] = ''           # Add to the front of the command that is run
    self.runInfoDict['postcommand'       ] = ''           # Added after the command is run.
    self.runInfoDict['mode'              ] = ''           # Running mode.  Curently the only mode supported is pbs
    self.runInfoDict['expectedTime'      ] = '10:00:00'   # How long the complete input is expected to run.

    #Following a set of dictionaries that, in a manner consistent with their names, collect the instance of all objects needed in the simulation
    #Theirs keywords in the dictionaries are the the user given names of data, sampler, etc.
    #The value corresponding to a keyword is the instance of the corresponding class
    self.stepsDict         = {}
    self.dataDict          = {}
    self.samplersDict      = {}
    self.modelsDict        = {}
    self.testsDict         = {}
    self.DistributionsDict = {}
    self.dataBasesDict     = {}
    self.OutStreamsDict    = {}
    self.filesDict         = {} #this is different, for each file rather than an instance it just returns the absolute path of the file
    
    self.stepSequenceList  = [] #the list of step of the simulation
    
    #list of supported quequing software:
    self.knownQuequingSoftware = []
    self.knownQuequingSoftware.append('None')
    self.knownQuequingSoftware.append('PBS Professional')

    #Class Dictionary when a new function is added to the simulation this dictionary need to be expanded
    #this dictionary is used to generate an instance of a class which name is among the keyword of the dictionary
    self.addWhatDict  = {}
    self.addWhatDict['Steps'         ] = Steps.returnInstance
    self.addWhatDict['Datas'         ] = Datas.returnInstance
    self.addWhatDict['Samplers'      ] = Samplers.returnInstance
    self.addWhatDict['Models'        ] = Models.returnInstance
    self.addWhatDict['Tests'         ] = Tests.returnInstance
    self.addWhatDict['Distributions' ] = Distributions.returnInstance
    self.addWhatDict['DataBases'     ] = DataBases.returnInstance
    self.addWhatDict['OutStreams'    ] = OutStreams.returnInstance

    #Mapping between a class type and the dictionary containing the instances for the simulation
    self.whichDict = {}
    self.whichDict['Steps'        ] = self.stepsDict
    self.whichDict['Datas'        ] = self.dataDict
    self.whichDict['Samplers'     ] = self.samplersDict
    self.whichDict['Models'       ] = self.modelsDict
    self.whichDict['Tests'        ] = self.testsDict
    self.whichDict['RunInfo'      ] = self.runInfoDict
    self.whichDict['Files'        ] = self.filesDict
    self.whichDict['Distributions'] = self.DistributionsDict
    self.whichDict['DataBases'    ] = self.dataBasesDict
    self.whichDict['OutStreams'   ] = self.OutStreamsDict
    
    self.jobHandler = JobHandler()
    self.__modeHandler = SimulationMode(self)

  def XMLread(self,xmlNode):
    '''parses the xml input file, instances the classes need to represent all objects in the simulation,
       and call for the initialization of the simulation'''
    for child in xmlNode:
      if child.tag in self.whichDict.keys():
        Type = child.tag
        if Type != 'RunInfo':
          for childChild in child:
            if childChild.attrib['name'] != None:
              name = childChild.attrib['name']
              #place the instance in the proper dictionary (self.whichDict[Type]) under his name as key,
              #the type is the general class (sampler, data, etc) while childChild.tag is the sub type
              self.whichDict[Type][name] = self.addWhatDict[Type](childChild.tag)
              #now we can read the info for this object
              print(Type,name,(self.whichDict[Type][name]))
              self.whichDict[Type][name].readXML(childChild)
              if self.debug:
                self.whichDict[Type][name].printMe()
              #if self.debug: self.whichDict[Type][name].printMe()
            else: raise IOError('not found name attribute for one '+Type)
        else: self.readRunInfo(child)
      else: raise IOError('the '+child.tag+' is not among the known simulation components '+ET.tostring(child))
    #finalize the initialization
    self.initialize()
    
  def initialize(self):
    '''gets simulation ready to run'''
    #check generate the existence of the working directory
    if not os.path.exists(self.runInfoDict['WorkingDir']):
      os.makedirs(self.runInfoDict['WorkingDir'])
    os.chdir(self.runInfoDict['WorkingDir'])
    #check consistency and fill the missing info for the // runs (threading, mpi, batches)
    self.runInfoDict['numProcByRun'] = self.runInfoDict['ParallelProcNumb']*self.runInfoDict['ThreadingProcessor']
    self.runInfoDict['totNumbCores'] = self.runInfoDict['numProcByRun']*self.runInfoDict['batchSize']
    #transform all files in absolute path
    for key in self.filesDict.keys():
      if os.path.split(key)[0] == '': self.filesDict[key] = os.path.join(self.runInfoDict['WorkingDir'],key)
      elif not os.path.isabs(key)   : self.filesDict[key] = os.path.abspath(key)
      if not os.path.exists(self.filesDict[key]): raise IOError('The file '+ key +' has not been found')
    #
    if self.runInfoDict['mode'] == 'pbs':
      self.__modeHandler = PBSSimulationMode(self)
    elif self.runInfoDict['mode'] == 'mpi':
      self.__modeHandler = MPISimulationMode(self)
    #Let the mode handler do any modification here
    self.__modeHandler.modifySimulation()
    self.jobHandler.initialize(self.runInfoDict)

  def readRunInfo(self,xmlNode):
    '''reads the xml input file for the RunInfo block'''
    for element in xmlNode:
      print(element.tag)
      if   element.tag == 'WorkingDir'        :
        temp_name = element.text
        if os.path.isabs(temp_name):
          self.runInfoDict['WorkingDir'        ] = element.text
        else:
          self.runInfoDict['WorkingDir'        ] = os.path.abspath(element.text)
      elif element.tag == 'ParallelCommand'   : self.runInfoDict['ParallelCommand'   ] = element.text.strip()
      elif element.tag == 'quequingSoftware'  : self.runInfoDict['quequingSoftware'  ] = element.text.strip()
      elif element.tag == 'ThreadingCommand'  : self.runInfoDict['ThreadingCommand'  ] = element.text.strip()
      elif element.tag == 'ThreadingProcessor': self.runInfoDict['ThreadingProcessor'] = int(element.text)
      elif element.tag == 'numNode'           : self.runInfoDict['numNode'           ] = int(element.text)
      elif element.tag == 'procByNode'        : self.runInfoDict['procByNode'        ] = int(element.text)
      elif element.tag == 'numProcByRun'      : self.runInfoDict['numProcByRun'      ] = int(element.text)
      elif element.tag == 'totNumbCores'      : self.runInfoDict['totNumbCores'      ] = int(element.text)
      elif element.tag == 'ParallelProcNumb'  : self.runInfoDict['ParallelProcNumb'  ] = int(element.text)
      elif element.tag == 'batchSize'         : self.runInfoDict['batchSize'         ] = int(element.text)
      elif element.tag == 'MaxLogFileSize'    : self.runInfoDict['MaxLogFileSize'    ] = int(element.text)
      elif element.tag == 'precommand'        : self.runInfoDict['precommand'        ] = element.text
      elif element.tag == 'postcommand'       : self.runInfoDict['postcommand'        ] = element.text
      elif element.tag == 'mode'              : self.runInfoDict['mode'              ] = element.text.strip().lower()
      elif element.tag == 'expectedTime'      : self.runInfoDict['expectedTime'      ] = element.text.strip()
      elif element.tag == 'Sequence':
        for stepName in element.text.split(','):
          self.stepSequenceList.append(stepName.strip())
      elif element.tag == 'Files':
        for fileName in element.text.split(','):
          self.filesDict[fileName] = fileName.strip()

  def printDicts(self):
    '''utility function capable to print a summary of the dictionaries'''
    def prntDict(Dict):
      '''utility function capable to print a dictionary'''
      for key in Dict:
        print(key+'= '+str(Dict[key]))
    prntDict(self.runInfoDict)
    prntDict(self.stepsDict)
    prntDict(self.dataDict)
    prntDict(self.samplersDict)
    prntDict(self.modelsDict)
    prntDict(self.testsDict)
    prntDict(self.filesDict)
    prntDict(self.dataBasesDict)
    prntDict(self.OutStreamsDict)
    prntDict(self.addWhatDict)
    prntDict(self.whichDict)

  def run(self):
    '''run the simulation'''
    if self.debug: print('entering in the run')
    if self.__modeHandler.doOverrideRun():
      self.__modeHandler.runOverride()
      return
    for stepName in self.stepSequenceList:                #loop over the the steps
      stepInstance = self.stepsDict[stepName]             #retrieve the instance of the step
      self.runInfoDict['stepName'] = stepName             #provide the name of the step to runInfoDict
      if self.debug: print('starting a step of type: '+stepInstance.type+', with name: '+stepInstance.name+' '+''.join((['-']*40)))
      inputDict = {}                    #initialize the input dictionary
      inputDict['Input' ] = []          #set the Input to an empty list
      inputDict['Output'] = []          #set the Output to an empty list
      for [key,b,c,d] in stepInstance.parList: #fill the take a a step input dictionary
#        if self.debug: print(a+' is:')
        #if self.debug:print([key,b,c,d])
        if key == 'Input':
          #print('this:',b,d)
          #print(self.whichDict.keys())
          #if self.debug:print(self.whichDict.keys())
          inputDict[key].append(self.whichDict[b][d])
#          if self.debug: print('type '+b+', and name: '+ str(self.whichDict[b][d])+'\n')
        elif key == 'Output':
          inputDict[key].append(self.whichDict[b][d])
#          if self.debug: self.whichDict[b][d].printMe()
        else:
          #Create extra dictionary entry
          #print('keys',inputDict.keys(),self.whichDict.keys())
          #print(key,b,d,self.whichDict.keys())
          inputDict[key] = self.whichDict[b][d]
#          if self.debug: self.whichDict[b][d].printMe()
      inputDict['jobHandler'] = self.jobHandler
      if 'Sampler' in inputDict.keys():
        inputDict['Sampler'].fillDistribution(self.DistributionsDict)
      stepInstance.takeAstep(inputDict)
      for output in inputDict['Output']:
        if "finalize" in dir(output):
          output.finalize()
      
      
      
#checks to be added: no same name within a data general class
#cross check existence of needed data

