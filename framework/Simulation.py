'''
Module that contains the driver for the whole the simulation flow (Simulation Class)
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import os,subprocess
import math
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import Steps
import Datas
import Samplers
import Models
import Tests
import Distributions
import DataBases
import Functions
import OutStreamManager
from JobHandler import JobHandler
#Internal Modules End--------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
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
    This typically modifies the precommand and the postcommand that 
    are put infront of the command and after the command.
    """
    import multiprocessing
    try:
      if multiprocessing.cpu_count() < self.__simulation.runInfoDict['batchSize']:
        print("SIMULATION    : WARNING cpu_count",multiprocessing.cpu_count()," < batchSize ",self.__simulation.runInfoDict['batchSize'])
    except NotImplementedError:
      pass

  def XMLread(self,xmlNode):
    """XMLread is called with the mode node, and can be used to 
    get extra parameters needed for the simulation mode.
    """
    pass

def createAndRunQSUB(simulation):
  """Generates a PBS qsub command to run the simulation"""
  # Check if the simulation has been run in PBS mode and, in case, construct the proper command
  batchSize = simulation.runInfoDict['batchSize']
  frameworkDir = simulation.runInfoDict["FrameworkDir"]
  ncpus = simulation.runInfoDict['NumThreads']
  #Generate the qsub command needed to run input
  command = ["qsub","-l",
             "select="+str(batchSize)+":ncpus="+str(ncpus)+":mpiprocs=1",
             "-l","walltime="+simulation.runInfoDict["expectedTime"],
             "-l","place=free","-v",
             'COMMAND="python Driver.py '+
             " ".join(simulation.runInfoDict["SimulationFiles"])+'"',
             os.path.join(frameworkDir,"raven_qsub_command.py")]
  #Change to frameworkDir so we find raven_qsub_command.sh
  os.chdir(frameworkDir)
  print(os.getcwd(),command)
  subprocess.call(command)
  

#-----------------------------------------------------------------------------------------------------
class PBSDSHSimulationMode(SimulationMode):
  
  def __init__(self,simulation):
    self.__simulation = simulation
    #Check if in pbs by seeing if environmental variable exists
    self.__in_pbs = "PBS_NODEFILE" in os.environ
    
  def doOverrideRun(self):
    # Check if the simulation has been run in PBS mode and, in case, construct the proper command    
    return not self.__in_pbs

  def runOverride(self):
    #Check and see if this is being accidently run
    assert self.__simulation.runInfoDict['mode'] == 'pbsdsh' and not self.__in_pbs
    createAndRunQSUB(self.__simulation)

  def modifySimulation(self):
    if self.__in_pbs:
      #Figure out number of nodes and use for batchsize
      nodefile = os.environ["PBS_NODEFILE"]
      lines = open(nodefile,"r").readlines()
      oldBatchsize =  self.__simulation.runInfoDict['batchSize']
      newBatchsize = len(lines) #the batchsize is just the number of nodes
      # of which there are one per line in the nodefile
      if newBatchsize != oldBatchsize:
        self.__simulation.runInfoDict['batchSize'] = newBatchsize
        print("SIMULATION    : WARNING: changing batchsize from",oldBatchsize,"to",newBatchsize)
      print("SIMULATION    : Using Nodefile to set batchSize:",self.__simulation.runInfoDict['batchSize'])
      #Add pbsdsh command to run.  pbsdsh runs a command remotely with pbs
      self.__simulation.runInfoDict['precommand'] = "pbsdsh -v -n %INDEX1% -- %FRAMEWORK_DIR%/raven_remote.sh out_%CURRENT_ID% %WORKING_DIR% "+self.__simulation.runInfoDict['precommand']
      if(self.__simulation.runInfoDict['NumThreads'] > 1):
        #Add the MOOSE --n-threads command afterwards
        self.__simulation.runInfoDict['postcommand'] = " --n-threads=%NUM_CPUS% "+self.__simulation.runInfoDict['postcommand']

#----------------------------------------------------------------------

class MPISimulationMode(SimulationMode):
  def __init__(self,simulation):
    self.__simulation = simulation
    #Figure out if we are in PBS
    self.__in_pbs = "PBS_NODEFILE" in os.environ
    self.__nodefile = False
    self.__runQsub = False

  def modifySimulation(self):
    if self.__nodefile or self.__in_pbs:
      if not self.__nodefile:
        #Figure out number of nodes and use for batchsize
        nodefile = os.environ["PBS_NODEFILE"]
      else:
        nodefile = self.__nodefile
      lines = open(nodefile,"r").readlines()
      numMPI = self.__simulation.runInfoDict['NumMPI']
      oldBatchsize = self.__simulation.runInfoDict['batchSize']
      #the batchsize is just the number of nodes of which there is one 
      # per line in the nodefile divided by the numMPI (which is per run)
      # and the floor and int and max make sure that the numbers are reasonable
      newBatchsize = max(int(math.floor(len(lines)/numMPI)),1)
      if newBatchsize != oldBatchsize:
        self.__simulation.runInfoDict['batchSize'] = newBatchsize
        print("SIMULATION    : WARNING: changing batchsize from",oldBatchsize,"to",newBatchsize)
      if newBatchsize > 1:
        #need to split node lines so that numMPI nodes are available per run
        workingDir = self.__simulation.runInfoDict['WorkingDir']
        for i in range(newBatchsize):
          node_file = open(os.path.join(workingDir,"node_"+str(i)),"w")
          for line in lines[i*numMPI:(i+1)*numMPI]:
            node_file.write(line)
          node_file.close()
        #then give each index a separate file.
        nodeCommand = "-f %BASE_WORKING_DIR%/node_%INDEX% "
      else:
        #If only one batch just use original node file
        nodeCommand = "-f "+nodefile
    else:
      #Not in PBS, so can't look at PBS_NODEFILE and none supplied in input
      newBatchsize = self.__simulation.runInfoDict['batchSize']
      numMPI = self.__simulation.runInfoDict['NumMPI']
      #TODO, we don't have a way to know which machines it can run on
      # when not in PBS so just distribute it over the local machine:
      nodeCommand = " "

    #Disable MPI processor affinity, which causes multiple processes 
    # to be forced to the same thread.
    os.environ["MV2_ENABLE_AFFINITY"] = "0"

    # Create the mpiexec pre command 
    self.__simulation.runInfoDict['precommand'] = "mpiexec "+nodeCommand+" -n "+str(numMPI)+" "+self.__simulation.runInfoDict['precommand']
    if(self.__simulation.runInfoDict['NumThreads'] > 1):
      #add number of threads to the post command.
      self.__simulation.runInfoDict['postcommand'] = " --n-threads=%NUM_CPUS% "+self.__simulation.runInfoDict['postcommand']
    print("precommand",self.__simulation.runInfoDict['precommand'],"postcommand",self.__simulation.runInfoDict['postcommand'])

  def doOverrideRun(self):
    # Check if the simulation has been run in PBS mode and if run QSUB
    # has been requested, in case, construct the proper command
    return (not self.__in_pbs) and self.__runQsub

  def runOverride(self):
    #Check and see if this is being accidently run
    assert self.__runQsub and not self.__in_pbs
    createAndRunQSUB(self.__simulation)


  def XMLread(self, xmlNode):
    for child in xmlNode:
      if child.tag == "nodefileenv":
        self.__nodefile = os.environ[child.text.strip()]
      elif child.tag == "nodefile":
        self.__nodefile = child.text.strip()
      elif child.tag.lower() == "runqsub":
        self.__runQsub = True
      else:
        print("SIMULATION    : We should do something with child",child)
    return

    
#-----------------------------------------------------------------------------------------------------
class Simulation(object):
  '''This is a class that contain all the object needed to run the simulation
  --Instance--
  myInstance = Simulation()
  myInstance.XMLread(xml.etree.ElementTree.Element) This method generates all the objects living in the simulation

  --usage--
  myInstance = Simulation()
  myInstance.XMLread(xml.etree.ElementTree.Element)  This method generate all the objects living in the simulation
  myInstance.initialize()                            This method takes care of setting up the directory/file environment with proper checks

  --Other external methods--
  myInstance.printDicts prints the dictionaries representing the whole simulation
  myInstance.whoAreYou()                 -see BaseType class-
  myInstance.myInitializzationParams()   -see BaseType class-
  myInstance.myClassmyCurrentSetting()           -see BaseType class-

  --how to add a <myClass> of component to the simulation--
  import the module <MyModule> where the new type of component is defined, you can name you module as you wish but so far we added an 's' t the class name (see Datas...)
  The module should possess a function <MyModule>.returnInstance('<myClass>') that returns a pointer to the class
  add to the class in the __init__: self.<myClass>Dict = {}
  add to the class in the __init__: self.whichDict['<myClass>']=<MyModule>.returnInstance
  add to the class in the __init__: self.whichDict['<myClass>'] = self.<myClass>Dict
  
  Comments on the simulation environment
  -every type of element living in the simulation should be uniquely identified by type and name not by sub-type
  !!!!Wrong:
  Class: distribution, subtype: normal,     name: myDistribution
  Class: distribution, subtype: triangular, name: myDistribution
  Correct:
  type: distribution, subtype: normal,     name: myNormalDist
  type: distribution, subtype: triangular, name: myTriDist
  
  it is therefore discouraged to use the attribute type and sub-type in the xml since they are naming inferred from the tags of the xml
  
  '''
  
  def __init__(self,frameworkDir,debug=False):
    self.debug= debug
    #this dictionary contains the general info to run the simulation
    self.runInfoDict = {}
    self.runInfoDict['DefaultInputFile'  ] = 'test.xml'   #Default input file to use
    self.runInfoDict['SimulationFiles'   ] = []           #the xml input file
    self.runInfoDict['ScriptDir'         ] = os.path.join(os.path.dirname(frameworkDir),"scripts") # the location of the pbs script interfaces
    self.runInfoDict['FrameworkDir'      ] = frameworkDir # the directory where the framework is located
    self.runInfoDict['WorkingDir'        ] = ''           # the directory where the framework should be running
    self.runInfoDict['TempWorkingDir'    ] = ''           # the temporary directory where a simulation step is run
    self.runInfoDict['NumMPI'            ] = 1            # the number of mpi process by run
    self.runInfoDict['NumThreads'        ] = 1            # Number of Threads by run
    self.runInfoDict['numProcByRun'      ] = 1            # Total number of core used by one run (number of threads by number of mpi)
    self.runInfoDict['batchSize'         ] = 1            # number of contemporaneous runs
    self.runInfoDict['ParallelCommand'   ] = ''           # the command that should be used to submit jobs in parallel (mpi)
    self.runInfoDict['ThreadingCommand'  ] = ''           # the command should be used to submit multi-threaded  
    self.runInfoDict['numNode'           ] = 1            # number of nodes
    self.runInfoDict['procByNode'        ] = 1            # number of processors by node
    self.runInfoDict['totalNumCoresUsed' ] = 1            # total number of cores used by driver 
    self.runInfoDict['quequingSoftware'  ] = ''           # quequing software name 
    self.runInfoDict['stepName'          ] = ''           # the name of the step currently running
    self.runInfoDict['precommand'        ] = ''           # Add to the front of the command that is run
    self.runInfoDict['postcommand'       ] = ''           # Added after the command that is run.
    self.runInfoDict['mode'              ] = ''           # Running mode.  Curently the only modes supported are pbsdsh and mpi
    self.runInfoDict['expectedTime'      ] = '10:00:00'   # How long the complete input is expected to run.

    #Following a set of dictionaries that, in a manner consistent with their names, collect the instance of all objects needed in the simulation
    #Theirs keywords in the dictionaries are the the user given names of data, sampler, etc.
    #The value corresponding to a keyword is the instance of the corresponding class
    self.stepsDict         = {}
    self.dataDict          = {}
    self.samplersDict      = {}
    self.modelsDict        = {}
    self.testsDict         = {}
    self.distributionsDict = {}
    self.dataBasesDict     = {}
    self.functionsDict     = {}
    self.filesDict         = {} #this is different, for each file rather than an instance it just returns the absolute path of the file
    self.OutStreamManagerDict = {}
    self.stepSequenceList  = [] #the list of step of the simulation
    
    #list of supported queue-ing software:
    self.knownQuequingSoftware = []
    self.knownQuequingSoftware.append('None')
    self.knownQuequingSoftware.append('PBS Professional')

    #Class Dictionary when a new function is added to the simulation this dictionary need to be expanded
    #this dictionary is used to generate an instance of a class which name is among the keyword of the dictionary
    self.addWhatDict  = {}
    self.addWhatDict['Steps'         ] = Steps
    self.addWhatDict['Datas'         ] = Datas
    self.addWhatDict['Samplers'      ] = Samplers
    self.addWhatDict['Models'        ] = Models
    self.addWhatDict['Tests'         ] = Tests
    self.addWhatDict['Distributions' ] = Distributions
    self.addWhatDict['DataBases'     ] = DataBases
    self.addWhatDict['Functions'     ] = Functions
    self.addWhatDict['OutStreamManager'    ] = OutStreamManager

    #Mapping between a class type and the dictionary containing the instances for the simulation
    #the dictionary keyword should match the subnodes of a step definition so that the step can find the instances
    self.whichDict = {}
    self.whichDict['Steps'        ] = self.stepsDict
    self.whichDict['Datas'        ] = self.dataDict
    self.whichDict['Samplers'     ] = self.samplersDict
    self.whichDict['Models'       ] = self.modelsDict
    self.whichDict['Tests'        ] = self.testsDict
    self.whichDict['RunInfo'      ] = self.runInfoDict
    self.whichDict['Files'        ] = self.filesDict
    self.whichDict['Distributions'] = self.distributionsDict
    self.whichDict['DataBases'    ] = self.dataBasesDict
    self.whichDict['Functions'    ] = self.functionsDict
    self.whichDict['OutStreamManager'   ] = self.OutStreamManagerDict
    
    self.jobHandler    = JobHandler()
    self.__modeHandler = SimulationMode(self)
    self.knownTypes    = self.whichDict.keys()

  def setInputFiles(self,inputFiles):
    '''Can be used to set the input files that the program received.  
    These are currently used for cluster running where the program 
    needs to be restarted on a different node.'''
    self.runInfoDict['SimulationFiles'   ] = inputFiles    

  def getDefaultInputFile(self):
    '''Returns the default input file to read'''
    return self.runInfoDict['DefaultInputFile']

  def __createAbsPath(self,filein):
    '''assuming that the file in is already in the self.filesDict it place as the value the absolute path'''
    if os.path.split(filein)[0] == '': self.filesDict[filein] = os.path.join(self.runInfoDict['WorkingDir'],filein)
    elif not os.path.isabs(filein)   : self.filesDict[filein] = os.path.abspath(filein)
  
  def __checkExistPath(self,filein):
    '''assuming that the file in is already in the self.filesDict it checks the existence'''
    if not os.path.exists(self.filesDict[filein]): raise IOError('The file '+ filein +' has not been found')

  def XMLread(self,xmlNode,runInfoSkip = set()):
    '''parses the xml input file, instances the classes need to represent all objects in the simulation'''
    for child in xmlNode:
      if child.tag in self.knownTypes:
        print('reading Class '+str(child.tag))
        Class = child.tag
        if Class != 'RunInfo':
          for childChild in child:
            if childChild.attrib['name'] != None:
              name = childChild.attrib['name']
              print('SIMULATION    : Reading type '+str(childChild.tag)+' with name '+name)
              #place the instance in the proper dictionary (self.whichDict[Type]) under his name as key,
              #the type is the general class (sampler, data, etc) while childChild.tag is the sub type
              if name not in self.whichDict[Class].keys():  self.whichDict[Class][name] = self.addWhatDict[Class].returnInstance(childChild.tag)
              else: raise IOError('SIMULATION    : Redundant  naming in the input for class '+Class+' and name '+name)
              #now we can read the info for this object
              self.whichDict[Class][name].readXML(childChild)
              if self.debug: self.whichDict[Class][name].printMe()
            else: raise IOError('SIMULATION    : not found name attribute for one '+Class)
        else: self.__readRunInfo(child,runInfoSkip)
      else: raise IOError('SIMULATION    : the '+child.tag+' is not among the known simulation components '+ET.tostring(child))    
    
  def initialize(self):
    '''check/created working directory, check/set up the parallel environment'''
    #check/generate the existence of the working directory 
    if not os.path.exists(self.runInfoDict['WorkingDir']): os.makedirs(self.runInfoDict['WorkingDir'])
    #move the full simulation environment in the working directory
    os.chdir(self.runInfoDict['WorkingDir'])
    #check consistency and fill the missing info for the // runs (threading, mpi, batches)
    self.runInfoDict['numProcByRun'] = self.runInfoDict['NumMPI']*self.runInfoDict['NumThreads']
    oldTotalNumCoresUsed = self.runInfoDict['totalNumCoresUsed']
    self.runInfoDict['totalNumCoresUsed'] = self.runInfoDict['numProcByRun']*self.runInfoDict['batchSize']
    if self.runInfoDict['totalNumCoresUsed'] < oldTotalNumCoresUsed:
      #This is used to reserve some cores
      self.runInfoDict['totalNumCoresUsed'] = oldTotalNumCoresUsed
    elif oldTotalNumCoresUsed > 1: #If 1, probably just default
      print("SIMULATION    : WARNING: overriding totalNumCoresUsed",oldTotalNumCoresUsed,"to",
            self.runInfoDict['totalNumCoresUsed'])
    #transform all files in absolute path
    for key in self.filesDict.keys():
      self.__createAbsPath(key)
    #Let the mode handler do any modification here
    self.__modeHandler.modifySimulation()
    self.jobHandler.initialize(self.runInfoDict)
    
    if self.debug: self.printDicts()
    

  def __readRunInfo(self,xmlNode,runInfoSkip):
    '''reads the xml input file for the RunInfo block'''
    for element in xmlNode:
      if element.tag in runInfoSkip:
        print("SIMULATION    : WARNING: Skipped element ",element.tag)
      elif   element.tag == 'WorkingDir'        :
        temp_name = element.text
        if os.path.isabs(temp_name):            self.runInfoDict['WorkingDir'        ] = element.text
        else:                                   self.runInfoDict['WorkingDir'        ] = os.path.abspath(element.text)
      elif element.tag == 'ParallelCommand'   : self.runInfoDict['ParallelCommand'   ] = element.text.strip()
      elif element.tag == 'quequingSoftware'  : self.runInfoDict['quequingSoftware'  ] = element.text.strip()
      elif element.tag == 'ThreadingCommand'  : self.runInfoDict['ThreadingCommand'  ] = element.text.strip()
      elif element.tag == 'NumThreads'        : self.runInfoDict['NumThreads'        ] = int(element.text)
      elif element.tag == 'numNode'           : self.runInfoDict['numNode'           ] = int(element.text)
      elif element.tag == 'procByNode'        : self.runInfoDict['procByNode'        ] = int(element.text)
      elif element.tag == 'totalNumCoresUsed' : self.runInfoDict['totalNumCoresUsed'   ] = int(element.text)
      elif element.tag == 'NumMPI'            : self.runInfoDict['NumMPI'            ] = int(element.text)
      elif element.tag == 'batchSize'         : self.runInfoDict['batchSize'         ] = int(element.text)
      elif element.tag == 'MaxLogFileSize'    : self.runInfoDict['MaxLogFileSize'    ] = int(element.text)
      elif element.tag == 'precommand'        : self.runInfoDict['precommand'        ] = element.text
      elif element.tag == 'postcommand'       : self.runInfoDict['postcommand'       ] = element.text
      elif element.tag == 'mode'              : 
        self.runInfoDict['mode'] = element.text.strip().lower()
        #parallel environment
        if self.runInfoDict['mode'] == 'pbsdsh':
          self.__modeHandler = PBSDSHSimulationMode(self)
        elif self.runInfoDict['mode'] == 'mpi':
          self.__modeHandler = MPISimulationMode(self)
        self.__modeHandler.XMLread(element)
      elif element.tag == 'expectedTime'      : self.runInfoDict['expectedTime'      ] = element.text.strip()
      elif element.tag == 'Sequence':
        for stepName in element.text.split(','): self.stepSequenceList.append(stepName.strip())
      elif element.tag == 'Files':
        for fileName in element.text.split(','): self.filesDict[fileName] = fileName.strip()
      elif element.tag == 'DefaultInputFile'  : self.runInfoDict['DefaultInputFile'] = element.text.strip()
      else:
        print("SIMULATION    : WARNING: Unhandled element ",element.tag)

  def printDicts(self):
    '''utility function capable to print a summary of the dictionaries'''
    def __prntDict(Dict):
      '''utility function capable to print a dictionary'''
      for key in Dict:
        print(key+'= '+str(Dict[key]))
    __prntDict(self.runInfoDict)
    __prntDict(self.stepsDict)
    __prntDict(self.dataDict)
    __prntDict(self.samplersDict)
    __prntDict(self.modelsDict)
    __prntDict(self.testsDict)
    __prntDict(self.filesDict)
    __prntDict(self.dataBasesDict)
    __prntDict(self.OutStreamManagerDict)
    __prntDict(self.addWhatDict)
    __prntDict(self.whichDict)

  def run(self):
    '''run the simulation'''
    #to do list
    #can we remove the check on the esistence of the file, it might make more sense just to check in case they are input and before the step they are used
    #
    if self.debug: print('SIMULATION    : entering in the run')
    #controlling the PBS environment
    if self.__modeHandler.doOverrideRun():
      self.__modeHandler.runOverride()
      return
    #loop over the steps of the simulation
    for stepName in self.stepSequenceList:
      stepInstance                     = self.stepsDict[stepName]   #retrieve the instance of the step
      self.runInfoDict['stepName']     = stepName                   #provide the name of the step to runInfoDict
      if self.debug: print('SIMULATION    : starting a step of type: '+stepInstance.type+', with name: '+stepInstance.name+' '+''.join((['-']*40)))
      stepInputDict                    = {}                         #initialize the input dictionary for a step. Never use an old one!!!!! 
      stepInputDict['Input' ]          = []                         #set the Input to an empty list
      stepInputDict['Output']          = []                         #set the Output to an empty list
      #fill the take a a step input dictionary just to recall: key= role played in the step b= Class, c= Type, d= user given name
      for [key,b,c,d] in stepInstance.parList: 
        if key == 'Input' or key == 'Output':                        #Only for input and output we allow more than one object passed to the step, so for those we build a list
          stepInputDict[key].append(self.whichDict[b][d])        
        else:
          stepInputDict[key] = self.whichDict[b][d]
        if key == 'Input' and b == 'Files': self.__checkExistPath(d) #if the input is a file, check if it exists 
      #add the global objects
      stepInputDict['jobHandler'] = self.jobHandler
      #generate the needed distributions to send to the step
      if 'Sampler' in stepInputDict.keys(): stepInputDict['Sampler'].generateDistributions(self.distributionsDict)
      #running a step
      stepInstance.takeAstep(stepInputDict)
      #---------------here what is going on? Please add comments-----------------
      for output in stepInputDict['Output']:
        if "finalize" in dir(output):
          output.finalize()
      
      
      
#checks to be added: no same name within a data general class
#cross check existence of needed data

