# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module where the base class and the specialization of different type of Model are
"""
#External Modules------------------------------------------------------------------------------------
import os
import sys
import copy
import shutil
import platform
import shlex
import time
import numpy as np
import pandas as pd
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Model import Model
from ..utils import utils
from ..utils import InputData, InputTypes
from ..Decorators.Parallelization import Parallel
from .. import CsvLoader #note: "from CsvLoader import CsvLoader" currently breaks internalParallel with Files and genericCodeInterface - talbpaul 2017-08-24
from .. import Files
from ..DataObjects import Data
from ..CodeInterfaceClasses import factory
#Internal Modules End--------------------------------------------------------------------------------

class Code(Model):
  """
    This is the generic class that import an external code into the framework
  """
  interfaceFactory = factory

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Code, cls).getInputSpecification()
    inputSpecification.setStrictMode(False) #Code interfaces can allow new elements.
    inputSpecification.addSub(InputData.parameterInputFactory("executable", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("walltime", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("preexec", contentType=InputTypes.StringType))

    ## Begin command line arguments tag
    ClargsInput = InputData.parameterInputFactory("clargs")

    ClargsTypeInput = InputTypes.makeEnumType("clargsType","clargsTypeType",["text","input","output","prepend","postpend","python"])
    ClargsInput.addParam("type", ClargsTypeInput, True)

    ClargsInput.addParam("arg", InputTypes.StringType, False)
    ClargsInput.addParam("extension", InputTypes.StringType, False)
    ClargsInput.addParam("delimiter", InputTypes.StringType, False)
    inputSpecification.addSub(ClargsInput)
    ## End command line arguments tag

    ## Begin file arguments tag
    FileargsInput = InputData.parameterInputFactory("fileargs")

    FileargsTypeInput = InputTypes.makeEnumType("fileargsType", "fileargsTypeType",["input","output","moosevpp"])
    FileargsInput.addParam("type", FileargsTypeInput, True)

    FileargsInput.addParam("arg", InputTypes.StringType, False)
    FileargsInput.addParam("extension", InputTypes.StringType, False)
    inputSpecification.addSub(FileargsInput)
    ## End file arguments tag

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    #FIXME think about how to import the roles to allowed class for the codes. For the moment they are not specialized by executable
    cls.validateDict['Input'] = [cls.testDict.copy()]
    cls.validateDict['Input'  ][0]['class'       ] = 'Files'
    # FIXME there's lots of types that Files can be, so until XSD replaces this, commenting this out
    #validateDict['Input'  ][1]['type'        ] = ['']
    cls.validateDict['Input'  ][0]['required'    ] = False
    cls.validateDict['Input'  ][0]['multiplicity'] = 'n'

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.executable = ''         # name of the executable (abs path)
    self.preExec = None          # name of the pre-executable, if any
    self.oriInputFiles = []      # list of the original input files (abs path)
    self.workingDir = ''         # location where the code is currently running
    self.outFileRoot = ''        # root to be used to generate the sequence of output files
    self.currentInputFiles = []  # list of the modified (possibly) input files (abs path)
    self.codeFlags = None        # flags that need to be passed into code interfaces(if present)
    self.printTag = 'CODE MODEL' # label
    self.createWorkingDir = True # whether to create the requested working dir
    self.foundExecutable = True  # True indicates the executable is found, otherwise not found
    self.foundPreExec = True     # True indicates the pre-executable is found, otherwise not found
    self.maxWallTime = None      # If set, this indicates the maximum CPU time a job can take.
    self._ravenWorkingDir = None # RAVEN's working dir

  def applyRunInfo(self, runInfo):
    """
      Take information from the RunInfo
      @ In, runInfo, dict, RunInfo info
      @ Out, None
    """
    self._ravenWorkingDir = runInfo['WorkingDir']

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Model._readMoreXML(self, xmlNode)
    paramInput = Code.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self.clargs={'text':'', 'input':{'noarg':[]}, 'pre':'', 'post':''} #output:''
    self.fargs={'input':{}, 'output':'', 'moosevpp':''}
    for child in paramInput.subparts:
      if child.getName() =='executable':
        self.executable = child.value if child.value is not None else ''
      if child.getName() =='walltime':
        self.maxWallTime = child.value
      if child.getName() =='preexec':
        self.preExec = child.value
      elif child.getName() == 'clargs':
        argtype    = child.parameterValues['type']      if 'type'      in child.parameterValues else None
        arg        = child.parameterValues['arg']       if 'arg'       in child.parameterValues else None
        ext        = child.parameterValues['extension'] if 'extension' in child.parameterValues else None
        # The default delimiter is one empty space
        delimiter  = child.parameterValues['delimiter'] if 'delimiter' in child.parameterValues else ' '
        if argtype == None:
          self.raiseAnError(IOError,'"type" for clarg not specified!')
        elif argtype == 'text':
          if ext != None:
            self.raiseAWarning('"text" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if not delimiter.strip():
            self.raiseAWarning('"text" nodes only accept "type" and "arg" attributes! Ignoring "delimiter"...')
          if arg == None:
            self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          self.clargs['text']=arg
        elif argtype == 'input':
          if ext == None:
            self.raiseAnError(IOError,'"extension" for clarg '+argtype+' not specified! Enter filetype to be listed for this flag.')
          if arg == None:
            self.clargs['input']['noarg'].append((ext,delimiter))
          else:
            if arg not in self.clargs['input'].keys():
              self.clargs['input'][arg]=[]
            # The delimiter is used to link 'arg' with the input file that have the file extension
            # given by 'extension'. In general, empty space is used. But in some specific cases, the codes may require
            # some specific delimiters to link the 'arg' and input files
            self.clargs['input'][arg].append((ext,delimiter))
        elif argtype == 'output':
          if arg == None:
            self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter flag for output file specification.')
          self.clargs['output'] = arg
        elif argtype == 'prepend':
          if ext != None:
            self.raiseAWarning('"prepend" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if arg == None:
            self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          if 'pre' in self.clargs:
            self.clargs['pre'] = arg+' '+self.clargs['pre']
          else:
            self.clargs['pre'] = arg
        elif argtype == 'python':
          pythonName = utils.getPythonCommand()
          if 'pre' in self.clargs:
            self.clargs['pre'] = self.clargs['pre']+' '+pythonName
          else:
            self.clargs['pre'] = pythonName
        elif argtype == 'postpend':
          if ext != None:
            self.raiseAWarning('"postpend" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if arg == None:
            self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          self.clargs['post'] = arg
        else:
          self.raiseAnError(IOError,'clarg type '+argtype+' not recognized!')
      elif child.getName() == 'fileargs':
        argtype = child.parameterValues['type']      if 'type'      in child.parameterValues else None
        arg     = child.parameterValues['arg']       if 'arg'       in child.parameterValues else None
        ext     = child.parameterValues['extension'] if 'extension' in child.parameterValues else None
        if argtype == None:
          self.raiseAnError(IOError,'"type" for filearg not specified!')
        elif argtype == 'input':
          if arg == None:
            self.raiseAnError(IOError,'filearg type "input" requires the template variable be specified in "arg" attribute!')
          if ext == None:
            self.raiseAnError(IOError,'filearg type "input" requires the auxiliary file extension be specified in "ext" attribute!')
          self.fargs['input'][arg]=[ext]
        elif argtype == 'output':
          if self.fargs['output']!='':
            self.raiseAnError(IOError,'output fileargs already specified!  You can only specify one output fileargs node.')
          if arg == None:
            self.raiseAnError(IOError,'filearg type "output" requires the template variable be specified in "arg" attribute!')
          self.fargs['output']=arg
        elif argtype.lower() == 'moosevpp':
          if self.fargs['moosevpp'] != '':
            self.raiseAnError(IOError,'moosevpp fileargs already specified!  You can only specify one moosevpp fileargs node.')
          if arg == None:
            self.raiseAnError(IOError,'filearg type "moosevpp" requires the template variable be specified in "arg" attribute!')
          self.fargs['moosevpp']=arg
        else:
          self.raiseAnError(IOError,'filearg type '+argtype+' not recognized!')
    if self.executable == '':
      self.raiseAWarning('The node "<executable>" was not found in the body of the code model '+str(self.name)+' so no code will be run...')
    else:
      if utils.stringIsFalse(os.environ.get('RAVENinterfaceCheck','False')):
        if '~' in self.executable:
          self.executable = os.path.expanduser(self.executable)
        abspath = os.path.abspath(str(self.executable))
        if os.path.exists(abspath):
          self.executable = abspath
        else:
          self.raiseAMessage('not found executable '+self.executable,'ExceptedError')
      else:
        self.foundExecutable = False
        self.raiseAMessage('InterfaceCheck: ignored executable '+self.executable, 'ExceptedError')
    if self.preExec is not None:
      if '~' in self.preExec:
        self.preExec = os.path.expanduser(self.preExec)
      abspath = os.path.abspath(self.preExec)
      if os.path.exists(abspath):
        self.preExec = abspath
      else:
        self.foundPreExec = False
        self.raiseAMessage('not found preexec '+self.preExec,'ExceptedError')
    self.code = self.interfaceFactory.returnInstance(self.subType)
    self.code.readXML(xmlNode, workingDir=self._ravenWorkingDir) #TODO figure out how to handle this with InputData
    self.code.setInputExtension(list(a[0].strip('.') for b in (c for c in self.clargs['input'].values()) for a in b))
    self.code.addInputExtension(list(a.strip('.') for b in (c for c in self.fargs ['input'].values()) for a in b))
    self.code.addDefaultExtension()

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Model.getInitParams(self)
    paramDict['executable']=self.executable
    return paramDict

  def getCurrentSetting(self):
    """
      This can be seen as an extension of getInitParams for the Code(model)
      that will return some information regarding the current settings of the
      code.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['current working directory'] = self.workingDir
    paramDict['current output file root' ] = self.outFileRoot
    paramDict['current input file'       ] = self.currentInputFiles
    paramDict['original input file'      ] = self.oriInputFiles
    return paramDict

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input.  By default does nothing.
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    inputInfo['additionalEdits']=self.fargs

  def initialize(self,runInfoDict,inputFiles,initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    self.workingDir = os.path.join(runInfoDict['WorkingDir'], runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    self.oriInputFiles = []
    for inputFile in inputFiles:
      subSubDirectory = os.path.join(self.workingDir,inputFile.subDirectory)
      ## Currently, there are no tests that verify the lines below can be hit
      ## It appears that the folders already exist by the time we get here,
      ## this could change, so we will leave this code here.
      ## -- DPM 8/2/17
      if inputFile.subDirectory.strip() != "" and not os.path.exists(subSubDirectory):
        os.makedirs(subSubDirectory)
      ##########################################################################
      if not os.path.exists(inputFile.getAbsFile()):
        self.raiseAnError(ValueError, 'The input file '+inputFile.getFilename()+' does not exist in directory: '+inputFile.getPath())
      shutil.copy(inputFile.getAbsFile(),subSubDirectory)
      self.oriInputFiles.append(copy.deepcopy(inputFile))
      self.oriInputFiles[-1].setPath(subSubDirectory)
    self.currentInputFiles = None
    self.outFileRoot = None
    if not self.foundExecutable:
      path = os.path.join(runInfoDict['WorkingDir'],self.executable)
      if os.path.exists(path):
        self.executable = path
      else:
        self.raiseAMessage('not found executable '+self.executable,'ExceptedError')
    if not self.foundPreExec:
      path = os.path.join(runInfoDict['WorkingDir'],self.preExec)
      if os.path.exists(path):
        self.preExec = path
      else:
        self.raiseAMessage('not found pre-executable '+self.executable,'ExceptedError')

    if 'initialize' in dir(self.code):
      # the deepcopy is needed to avoid the code interface
      # developer to modify the content of the runInfoDict
      self.code.initialize(copy.deepcopy(runInfoDict), self.oriInputFiles)

  def createNewInput(self,currentInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet only the last set of entries are copied.
      The copied values are returned as a dictionary back
      @ In, currentInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the SampledVars'that contains a dictionary {'name variable':value}
           also 'additionalEdits', similar dictionary for non-variables
      @ Out, createNewInput, tuple, return the new input in a tuple form
    """
    found = False
    newInputSet = copy.deepcopy(currentInput)

    #TODO FIXME I don't think the extensions are the right way to classify files anymore, with the new Files
    #  objects.  However, this might require some updating of many Code Interfaces as well.
    for index, inputFile in enumerate(newInputSet):
      if inputFile.getExt() in self.code.getInputExtension():
        found = True
        break
    if not found:
      self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                  + self.subType +': ' + ' '.join(self.code.getInputExtension()))

    # check if in batch
    brun = kwargs.get('batchRun')
    if brun is not None:
      # if batch, the subDir are a combination of prefix (batch id) and batch run id
      bid = kwargs['prefix'] if 'prefix' in kwargs.keys() else '1'
      subDirectory = os.path.join(self.workingDir,'b{}_r{}'.format(bid,brun))
    else:
      subDirectory = os.path.join(self.workingDir, kwargs['prefix'] if 'prefix' in kwargs.keys() else '1')

    if not os.path.exists(subDirectory):
      os.mkdir(subDirectory)
    for index in range(len(newInputSet)):
      subSubDirectory = os.path.join(subDirectory,newInputSet[index].subDirectory)
      ## Currently, there are no tests that verify the lines below can be hit
      ## It appears that the folders already exist by the time we get here,
      ## this could change, so we will leave this code here.
      ## -- DPM 8/2/17
      if newInputSet[index].subDirectory.strip() != "" and not os.path.exists(subSubDirectory):
        os.makedirs(subSubDirectory)
      ##########################################################################
      newInputSet[index].setPath(subSubDirectory)
      shutil.copy(self.oriInputFiles[index].getAbsFile(),subSubDirectory)

    kwargs['subDirectory'] = subDirectory
    kwargs['alias'] = self.alias

    if 'SampledVars' in kwargs.keys():
      sampledVars = self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'],'input',False)

    newInput    = self.code.createNewInput(newInputSet,self.oriInputFiles,samplerType,**copy.deepcopy(kwargs))

    if 'SampledVars' in kwargs.keys() and len(self.alias['input'].keys()) != 0:
      kwargs['SampledVars'] = sampledVars

    return (newInput,kwargs)

  def _expandCommand(self, origCommand):
    """
      Function to expand a command from string to list.
      RAVEN employs subprocess.Popen to spawn new processes, and RAVEN allows code interface developers
      to control shell argument of Popen. When shell is True, a string is required for the command. When shell
      is False, a sequence, i.e. a list of strings, is required.
      The reasons are: In general, a sequence of arguments is preferred, as it allows the module to
      take care of any required escaping and quoting of arguments. When shell is True, a string is preferred,
      since when a sequence is provided, only the first item specifies the command string, and any additional
      items will be treated as additional arguments to the shell itself.
      @ In, origCommand, string, The command to check for expansion
      @ Out, commandSplit, string or String List, the expanded command or the original if not expanded.
    """
    if origCommand.strip() == '':
      return ['echo', 'no command provided']
    commandSplit = shlex.split(origCommand)
    return commandSplit

  def _expandForWindows(self, origCommand):
    """
      Function to expand a command that has a #! to a windows runnable command
      @ In, origCommand, string, The command to check for expansion
      @ Out, commandSplit, string or String List, the expanded command or the original if not expanded.
    """
    if origCommand.strip() == '':
      return origCommand
    # In Windows Python, you can get some backslashes in your paths
    commandSplit = shlex.split(origCommand.replace("\\","/"))
    executable = commandSplit[0]

    if os.path.exists(executable):
        try:
          with open(executable, "r+b") as executableFile:
            firstTwoChars = executableFile.read(2)
            if firstTwoChars == "#!":
              realExecutable = shlex.split(executableFile.readline())
              self.raiseAMessage("reading #! to find executable:" + repr(realExecutable))
              # The below code should work, and would be better than findMsys,
              # but it doesn't work.
              # winExecutable = subprocess.check_output(['cygpath','-w',realExecutable[0]],shell=True).rstrip()
              # print("winExecutable",winExecutable)
              # realExecutable[0] = winExecutable
              def findMsys():
                """
                  Function to try and figure out where the MSYS64 is.
                  @ In, None
                  @ Out, dir, String, If not None, the directory where msys is.
                """
                dir = os.getcwd()
                head, tail = os.path.split(dir)
                while True:
                  if tail.lower().startswith("msys"):
                    return dir
                  dir = head
                  head, tail = os.path.split(dir)
                return None
              msysDir = findMsys()
              if msysDir is not None:
                beginExecutable = realExecutable[0]
                if beginExecutable.startswith("/"):
                  beginExecutable = beginExecutable.lstrip("/")
                winExecutable = os.path.join(msysDir, beginExecutable)
                self.raiseAMessage("winExecutable " + winExecutable)
                if not os.path.exists(winExecutable) and not os.path.exists(winExecutable + ".exe") and winExecutable.endswith("bash"):
                  #msys64 stores bash in /usr/bin/bash instead of /bin/bash, so try that
                  maybeWinExecutable = winExecutable.replace("bin/bash","usr/bin/bash")
                  if os.path.exists(maybeWinExecutable) or os.path.exists(maybeWinExecutable + ".exe"):
                    winExecutable = maybeWinExecutable
                realExecutable[0] = winExecutable
              else:
                self.raiseAWarning("Could not find msys in "+os.getcwd())
              commandSplit = realExecutable + [executable] + commandSplit[1:]
              return commandSplit
        except PermissionError as e:
            self.raiseAWarning("Permission denied to open executable ! Skipping!")
    return origCommand

  @Parallel()
  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars that contains a dictionary {'name variable':value}
        @ Out, returnValue, tuple, This will hold two pieces of information,
          the first item will be the input data used to generate this sample,
          the second item will be the output of this model given the specified
          inputs
    """
    inputFiles = self.createNewInput(myInput, samplerType, **kwargs)
    self.currentInputFiles, metaData = (copy.deepcopy(inputFiles[0]),inputFiles[1]) if type(inputFiles).__name__ == 'tuple' else (inputFiles, None)
    returnedCommand = self.code.genCommand(self.currentInputFiles,self.executable, flags=self.clargs, fileArgs=self.fargs, preExec=self.preExec)

    ## Given that createNewInput can only return a tuple, I don't think these
    ## checks are necessary (keeping commented out until someone else can verify):
    # if type(returnedCommand).__name__ != 'tuple':
    #   self.raiseAnError(IOError, "the generateCommand method in code interface must return a tuple")
    # if type(returnedCommand[0]).__name__ != 'list':
    #   self.raiseAnError(IOError, "the first entry in tuple returned by generateCommand method needs to be a list of tuples!")
    executeCommand, self.outFileRoot = returnedCommand

    precommand = kwargs['precommand']
    postcommand = kwargs['postcommand']
    bufferSize = kwargs['logfileBuffer']
    fileExtensionsToDelete = kwargs['deleteOutExtension']
    deleteSuccessfulLogFiles = kwargs['delSucLogFiles']

    codeLogFile = self.outFileRoot
    if codeLogFile is None:
      codeLogFile = os.path.join(metaData['subDirectory'],'generalOut')

    ## Before we were temporarily changing directories in order to copy the
    ## correct directory to the subprocess. Instead, we can just set the
    ## directory after we copy it over. -- DPM 5/5/2017
    sampleDirectory = os.path.join(os.getcwd(),metaData['subDirectory'])
    localenv = dict(os.environ)
    localenv['PWD'] = str(sampleDirectory)
    outFileObject = open(os.path.join(sampleDirectory,codeLogFile), 'w', bufferSize)

    found = False
    for index, inputFile in enumerate(self.currentInputFiles):
      if inputFile.getExt() in self.code.getInputExtension():
        found = True
        break
    if not found:
      self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                  + self.subType +': ' + ' '.join(self.getInputExtension()))
    commands=[]
    for runtype,cmd in executeCommand:
      newCommand=''
      if runtype.lower() == 'parallel':
        newCommand += precommand
        newCommand += cmd+' '
        newCommand += postcommand
        commands.append(newCommand)
      elif runtype.lower() == 'serial':
        commands.append(cmd)
      else:
        self.raiseAnError(IOError,'For execution command <'+cmd+'> the run type was neither "serial" nor "parallel"!  Instead received: ',runtype,'\nPlease check the code interface.')

    command = ' && '.join(commands)+' '

    command = command.replace("%INDEX%",kwargs['INDEX'])
    command = command.replace("%INDEX1%",kwargs['INDEX1'])
    command = command.replace("%CURRENT_ID%",kwargs['CURRENT_ID'])
    command = command.replace("%CURRENT_ID1%",kwargs['CURRENT_ID1'])
    command = command.replace("%SCRIPT_DIR%",kwargs['SCRIPT_DIR'])
    command = command.replace("%FRAMEWORK_DIR%",kwargs['FRAMEWORK_DIR'])
    ## Note this is the working directory that the subprocess will use, it is
    ## not the directory I am currently working. This bit me as I moved the code
    ## from the old ExternalRunner because in that case this was filled in after
    ## the process was submitted by the process itself. -- DPM 5/4/17
    command = command.replace("%WORKING_DIR%",sampleDirectory)
    command = command.replace("%BASE_WORKING_DIR%",kwargs['BASE_WORKING_DIR'])
    command = command.replace("%METHOD%",kwargs['METHOD'])
    command = command.replace("%NUM_CPUS%",kwargs['NUM_CPUS'])
    command = command.replace("%PYTHON%", sys.executable)

    self.raiseAMessage('Execution command submitted:',command)
    if platform.system() == 'Windows':
      command = self._expandForWindows(command)
      self.raiseAMessage("modified command to", repr(command))
      for key, value in localenv.items():
        localenv[key]=str(value)
    elif not self.code.getRunOnShell():
      command = self._expandCommand(command)
    self.raiseADebug(f'shell execution command: "{command}"')
    self.raiseADebug('shell cwd: "'+localenv['PWD']+'"')
    self.raiseADebug('self pid:' + str(os.getpid())+' ppid: '+str(os.getppid()))
    ## reset python path
    localenv.pop('PYTHONPATH',None)
    ## This code should be evaluated by the job handler, so it is fine to wait
    ## until the execution of the external subprocess completes.
    process = utils.pickleSafeSubprocessPopen(command, shell=self.code.getRunOnShell(), stdout=outFileObject, stderr=outFileObject, cwd=localenv['PWD'], env=localenv)
    if self.maxWallTime is not None:
      timeout = time.time() + self.maxWallTime
      while True:
        time.sleep(0.5)
        process.poll()
        if time.time() > timeout and process.returncode is None:
          self.raiseAWarning('walltime exceeded in run in working dir: '+str(metaData['subDirectory'])+'. Killing the run...')
          process.kill()
          process.returncode = -1
        if process.returncode is not None or time.time() > timeout:
          break
    else:
      process.wait()

    returnCode = process.returncode
    self.raiseADebug(" Process "+str(process.pid)+" finished "+time.ctime()+
                     " with returncode "+str(process.returncode))
    # procOutput = process.communicate()[0]

    ## If the returnCode is already non-zero, we should maintain our current
    ## value as it may have some meaning that can be parsed at some point, so
    ## only set the returnCode to -1 in here if we did not already catch the
    ## failure.
    if returnCode == 0 and 'checkForOutputFailure' in dir(self.code):
      codeFailed = self.code.checkForOutputFailure(codeLogFile, metaData['subDirectory'])
      if codeFailed:
        returnCode = -1
    # close the log file
    outFileObject.close()
    ## We should try and use the output the code interface gives us first, but
    ## in lieu of that we should fall back on the standard output of the code
    ## (Which was deleted above in some cases, so I am not sure if this was
    ##  an intentional design by the original developer or accidental and should
    ##  be revised).
    ## My guess is that every code interface implements this given that the code
    ## below always adds .csv to the filename and the standard output file does
    ## not have an extension. - (DPM 4/6/2017)
    outputFile, isStr = codeLogFile, True
    if 'finalizeCodeOutput' in dir(self.code) and returnCode == 0:
      finalCodeOutput = self.code.finalizeCodeOutput(command, codeLogFile, metaData['subDirectory'])
      ## Special case for RAVEN interface --ALFOA 09/17/17
      ravenCase = type(finalCodeOutput).__name__ == 'dict' and self.code.__class__.__name__ == 'RAVEN'
      # check return of finalizecode output
      if finalCodeOutput is not None:
        isDict = isinstance(finalCodeOutput,dict)
        isStr = isinstance(finalCodeOutput,str)
        if not isDict and not isStr:
          self.raiseAnError(RuntimeError, 'The return argument from "finalizeCodeOutput" must be either a str' +
                                          'containing the new output file root or a dict of data!')
      if finalCodeOutput and not ravenCase:
        if not isDict:
          outputFile = finalCodeOutput
        else:
          returnDict = finalCodeOutput
    ## If the run was successful
    if returnCode == 0:
      ## This may be a tautology at this point --DPM 4/12/17
      ## Special case for RAVEN interface. Added ravenCase flag --ALFOA 09/17/17
      if outputFile and isStr and not ravenCase:
        outFile = Files.CSV()
        ## Should we be adding the file extension here?
        outFile.initialize(outputFile+'.csv', path=metaData['subDirectory'])

        csvLoader = CsvLoader.CsvLoader()
        # does this CodeInterface have sufficiently intense (or limited) CSV files that
        #   it needs to assume floats and use numpy, or can we use pandas?
        loadUtility = self.code.getCsvLoadUtil()
        csvData = csvLoader.loadCsvFile(outFile.getAbsFile(), nullOK=False, utility=loadUtility)
        returnDict = csvLoader.toRealization(csvData)

      if not ravenCase:
        # check if the csv needs to be printed
        if self.code.getIfWriteCsv():
          csvFileName = os.path.join(metaData['subDirectory'],outputFile+'.csv')
          pd.DataFrame.from_dict(returnDict).to_csv(path_or_buf=csvFileName,index=False)
        self._replaceVariablesNamesWithAliasSystem(returnDict, 'inout', True)
        returnDict.update(kwargs)
        returnValue = (kwargs['SampledVars'],returnDict)
        exportDict = self.createExportDictionary(returnValue)
      else:
        # we have the DataObjects -> raven-runs-raven case only so far
        # we have two tasks to do: collect the input/output/meta/indexes from the INNER raven run, and ALSO the input from the OUTER raven run.
        #  -> in addition, we have to fix the probability weights.
        ## get the number of realizations
        ### we already checked consistency in the CodeInterface, so just get the length of the first data object
        numRlz = len(utils.first(finalCodeOutput.values()))
        ## set up the return container
        exportDict = {'RAVEN_isBatch':True,'realizations':[]}
        ## set up each realization
        for n in range(numRlz):
          rlz = {}
          ## collect the results from INNER, both point set and history set
          for dataObj in finalCodeOutput.values():
            # TODO FIXME check for overwriting data.  For now just replace data if it's duplicate!
            new = dict((var,np.atleast_1d(val)) for var,val in dataObj.realization(index=n,unpackXArray=True).items())
            rlz.update( new )
          ## add OUTER input space
          # TODO FIXME check for overwriting data.  For now just replace data if it's duplicate!
          new = dict((var,np.atleast_1d(val)) for var,val in kwargs['SampledVars'].items())
          rlz.update( new )
          ## combine ProbabilityWeights # TODO FIXME these are a rough attempt at getting it right!
          rlz['ProbabilityWeight'] = np.atleast_1d(rlz.get('ProbabilityWeight',1.0) * kwargs.get('ProbabilityWeight',1.0))
          rlz['PointProbability'] = np.atleast_1d(rlz.get('PointProbability',1.0) * kwargs.get('PointProbability',1.0))
          # FIXME: adding "_n" to Optimizer samples scrambles its ability to find evaluations!
          ## temporary fix: only append if there's multiple realizations, and error out if sampler is an optimizer.
          if numRlz > 1:
            if '_' in kwargs['prefix']:
              self.raiseAnError(RuntimeError,'OUTER RAVEN is using an OPTIMIZER, but INNER RAVEN is returning multiple realizations!')
            addon = '_{}'.format(n)
          else:
            addon = ''
          rlz['prefix'] = np.atleast_1d(kwargs['prefix']+addon)
          ## add the rest of the metadata # TODO slow
          for var,val in kwargs.items():
            if var not in rlz.keys():
              rlz[var] = np.atleast_1d(val)
          self._replaceVariablesNamesWithAliasSystem(rlz,'inout',True)
          exportDict['realizations'].append(rlz)

      ## The last thing before returning should be to delete the temporary log
      ## file and any other file the user requests to be cleared
      if deleteSuccessfulLogFiles:
        self.raiseAMessage(' Run "' +kwargs['prefix']+'" ended smoothly, removing log file!')
        codeLofFileFullPath = os.path.join(metaData['subDirectory'],codeLogFile)
        if os.path.exists(codeLofFileFullPath):
          os.remove(codeLofFileFullPath)

      ## Check if the user specified any file extensions for clean up
      for fileExt in fileExtensionsToDelete:
        fileList = [ os.path.join(metaData['subDirectory'],f) for f in os.listdir(metaData['subDirectory']) if f.endswith(fileExt) ]
        for f in fileList:
          os.remove(f)

      return exportDict

    else:
      self.raiseAMessage("*"*50)
      self.raiseAMessage(" Process Failed "+str(command)+" returnCode "+str(returnCode))
      absOutputFile = os.path.join(sampleDirectory,outputFile)
      if os.path.exists(absOutputFile):
        if getattr(self.code, 'printFailedRuns', True):
          self.raiseAMessage(repr(open(absOutputFile,"r").read()).replace("\\n","\n"))
        else:
          self.raiseAMessage(f'Ouput is in "{os.path.abspath(absOutputFile)}"')
      else:
        self.raiseAMessage(" No output " + absOutputFile)
      self.raiseAMessage("*"*50)

      ## If you made it here, then the run must have failed
      return None

  def createExportDictionary(self, evaluation):
    """
      Method that is aimed to create a dictionary with the sampled and output variables that can be collected by the different
      output objects.
      @ In, evaluation, tuple, (dict of sampled variables, dict of code outputs)
      @ Out, outputEval, dict, dictionary containing the output/input values: {'varName':value}
    """
    sampledVars,outputDict = evaluation

    if type(outputDict).__name__ == "tuple":
      outputEval = outputDict[0]
    else:
      outputEval = outputDict

    for key, value in outputEval.items():
      outputEval[key] = np.atleast_1d(value)

    for key, value in sampledVars.items():
      if key in outputEval.keys():
        if not utils.compare(value,np.atleast_1d(outputEval[key])[-1],relTolerance = 1e-8):
          self.raiseAWarning('The model '+self.type+' reported a different value ({}) for {} '.format(outputEval[key][0], key) +
                             'than raven\'s suggested sample ({}). Using the value reported '.format(value) +
                             'by the raven ({}).'.format(value))
      outputEval[key] = np.atleast_1d(value)
    self._replaceVariablesNamesWithAliasSystem(outputEval, 'input',True)

    return outputEval

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if not hasattr(evaluation, 'pop'):
      self.raiseAWarning("No pop in evaluation " + repr(evaluation) + " for job" + repr(finishedJob) + " with return code "+ repr(finishedJob.getReturnCode()))


    self._replaceVariablesNamesWithAliasSystem(evaluation, 'input',True)
    # in the event a batch is run, the evaluations will be a dict as {'RAVEN_isBatch':True, 'realizations': [...]}
    if isinstance(evaluation,dict) and evaluation.get('RAVEN_isBatch',False):
      for rlz in evaluation['realizations']:
        output.addRealization(rlz)
    # otherwise, we received a single realization
    else:
      output.addRealization(evaluation)

    ##TODO How to handle restart?
    ##TODO How to handle collectOutputFromDataObject

    return

  ###################################################################################
  ## THIS METHOD NEEDS TO BE REWORKED WHEN THE NEW DATAOBJECT STRUCURE IS IN PLACE ##
  ###################################################################################
  def collectOutputFromDataObject(self,exportDict,output):
    """
      Method to collect the output from a DataObject (if it is not a dataObject, it just returns a list with one single exportDict)
      @ In, exportDict, dict, the export dictionary
                               ({'inputSpaceParams':{var1:value1,var2:value2},
                                 'outputSpaceParams':{outstreamName1:DataObject1,outstreamName2:DataObject2},
                                 'metadata':{'metadataName1':value1,'metadataName2':value2}})
      @ Out, returnList, list, list of export dictionaries
    """
    returnList = []
    if utils.first(exportDict['outputSpaceParams'].values()).__class__.__base__.__name__ != 'Data':
      returnList.append(exportDict)
    else:
      # get the DataObject that is compatible with this output
      compatibleDataObject = None
      for dataObj in exportDict['outputSpaceParams'].values():
        if output.type == dataObj.type:
          compatibleDataObject = dataObj
          break
        if output.type == 'HDF5' and dataObj.type == 'HistorySet':
          compatibleDataObject = dataObj
          break
      if compatibleDataObject is None:
        # if none found (e.g. => we are filling an HistorySet with a PointSet), we take the first one
        compatibleDataObject = utils.first(exportDict['outputSpaceParams'].values())
      # get the values
      inputs = compatibleDataObject.getParametersValues('inputs',nodeId = 'RecontructEnding')
      unstructuredInputs = compatibleDataObject.getParametersValues('unstructuredinputs',nodeId = 'RecontructEnding')
      outputs = compatibleDataObject.getParametersValues('outputs',nodeId = 'RecontructEnding')
      metadata = compatibleDataObject.getAllMetadata(nodeId = 'RecontructEnding')
      inputKeys = inputs.keys() if compatibleDataObject.type == 'PointSet' else utils.first(inputs.values()).keys()
      # expand inputspace of current RAVEN
      for i in range(len(compatibleDataObject)):
        appendDict = {'inputSpaceParams':{},'outputSpaceParams':{},'metadata':{}}
        appendDict['inputSpaceParams'].update(exportDict['inputSpaceParams'])
        appendDict['metadata'].update(exportDict['metadata'])
        if compatibleDataObject.type == 'PointSet':
          for inKey, value in inputs.items():
            appendDict['inputSpaceParams'][inKey] = value[i]
          for inKey, value in unstructuredInputs.items():
            appendDict['inputSpaceParams'][inKey] = value[i]
          for outKey, value in outputs.items():
            appendDict['outputSpaceParams'][outKey] = value[i]
        else:
          for inKey, value in inputs.values()[i].items():
            appendDict['inputSpaceParams'][inKey] = value
          if len(unstructuredInputs) > 0:
            for inKey, value in unstructuredInputs.values()[i].items():
              appendDict['inputSpaceParams'][inKey] = value
          for outKey, value in outputs.values()[i].items():
            appendDict['outputSpaceParams'][outKey] = value
        # add metadata for both dataobject types
        for metadataToExport in ['SampledVars','SampledVarsPb']:
          if metadataToExport in metadata:
            appendDict['metadata'][metadataToExport].update(metadata[metadataToExport][i])
        weightForVars = ['ProbabilityWeight-'+var.strip()  for var in inputKeys]
        for metadataToMerge in ['ProbabilityWeight', 'PointProbability']+weightForVars:
          if metadataToMerge in appendDict['metadata']:
            if metadataToMerge in metadata:
              appendDict['metadata'][metadataToMerge]*= metadata[metadataToMerge][i]
          else:
            if metadataToMerge in metadata:
              appendDict['metadata'][metadataToMerge] = metadata[metadataToMerge][i]
        returnList.append(appendDict)
    return returnList


  #TODO: Seems to me, this function can be removed --- wangc Dec. 2017
  def collectOutputFromDict(self,exportDict,output,options=None):
    """
      Collect results from dictionary
      @ In, exportDict, dict, contains 'inputs','outputs','metadata'
      @ In, output, instance, the place to write to
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    prefix = exportDict.pop('prefix',None)
    if 'inputSpaceParams' in exportDict.keys():
      inKey = 'inputSpaceParams'
      outKey = 'outputSpaceParams'
    else:
      inKey = 'inputs'
      outKey = 'outputs'

    rlz = {}
    rlz.update(exportDict[inKey])
    rlz.update(exportDict[outKey])
    rlz.update(exportDict['metadata'])
    for key, value in rlz.items():
      rlz[key] = np.atleast_1d(value)
    output.addRealization(rlz)

    return

  def submit(self, myInput, samplerType, jobHandler, **kwargs):
    """
        This will submit an individual sample to be evaluated by this model to a
        specified jobHandler. Note, some parameters are needed by createNewInput
        and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In,  jobHandler, JobHandler instance, the global job handler instance
        @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, None
    """

    nRuns = 1
    batchMode =  kwargs.get("batchMode", False)
    if batchMode:
      nRuns = kwargs["batchInfo"]['nRuns']

    for i in range(nRuns):
      if batchMode:
        kw =  kwargs['batchInfo']['batchRealizations'][i]
      else:
        kw = kwargs

      prefix = kw.get("prefix")
      uniqueHandler = kw.get("uniqueHandler",'any')
      # if batch mode is on, lets record the run id within the batch
      if batchMode:
        kw['batchRun'] = i+1

      ## These two are part of the current metadata, so they will be added before
      ## the job is started, so that they will be captured in the metadata and match
      ## the current behavior of the system. If these are not desired, then this
      ## code can be moved to later.  -- DPM 4/12/17
      kw['executable'] = self.executable
      kw['outfile'] = None

      #TODO FIXME I don't think the extensions are the right way to classify files anymore, with the new Files
      #  objects.  However, this might require some updating of many CodeInterfaces``````           1  Interfaces as well.
      for index, inputFile in enumerate(myInput):
        if inputFile.getExt() in self.code.getInputExtension():
          kw['outfile'] = 'out~'+myInput[index].getBase()
          break
      if kw['outfile'] is None:
        self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                  + self.subType +': ' + ' '.join(self.code.getInputExtension()))

      ## These kw are updated by createNewInput, so the job either should not
      ## have access to the metadata, or it needs to be updated from within the
      ## evaluateSample function, which currently is not possible since that
      ## function does not know about the job instance.
      metadata = copy.copy(kw)

      ## These variables should not be part of the metadata, so add them after
      ## we copy this dictionary (Caught this when running an ensemble model)
      ## -- DPM 4/11/17
      nodesList                = jobHandler.runInfoDict.get('Nodes',[])
      kw['logfileBuffer'     ] = jobHandler.runInfoDict['logfileBuffer']
      kw['precommand'        ] = jobHandler.runInfoDict['precommand']
      kw['postcommand'       ] = jobHandler.runInfoDict['postcommand']
      kw['delSucLogFiles'    ] = jobHandler.runInfoDict['delSucLogFiles']
      kw['deleteOutExtension'] = jobHandler.runInfoDict['deleteOutExtension']
      kw['NumMPI'            ] = jobHandler.runInfoDict.get('NumMPI',1)
      kw['numberNodes'       ] = len(nodesList)

      ## This may look a little weird, but due to how the parallel python library
      ## works, we are unable to pass a member function as a job because the
      ## pp library loses track of what self is, so instead we call it from the
      ## class and pass self in as the first parameter
      jobHandler.addJob((self, myInput, samplerType, kw), self.__class__.evaluateSample, prefix, metadata=metadata,
                        uniqueHandler=uniqueHandler, groupInfo={'id': kwargs['batchInfo']['batchId'], 'size': nRuns} if batchMode else None)
      if nRuns == 1:
        self.raiseAMessage('job "' + str(prefix) + '" submitted!')
      else:
        self.raiseAMessage('job "' + str(i+1) + '" in batch "'+str(kwargs['batchInfo']['batchId']) + '" submitted!')
