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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import shutil
import importlib
import platform
import shlex
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Model import Model
from utils import utils
from utils import InputData
import CsvLoader #note: "from CsvLoader import CsvLoader" currently breaks internalParallel with Files and genericCodeInterface - talbpaul 2017-08-24
import Files
from DataObjects import Data
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class Code(Model):
  """
    This is the generic class that import an external code into the framework
  """
  CodeInterfaces = importlib.import_module("CodeInterfaces")

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
    inputSpecification.addSub(InputData.parameterInputFactory("executable", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("preexec", contentType=InputData.StringType))

    ## Begin command line arguments tag
    ClargsInput = InputData.parameterInputFactory("clargs")

    ClargsTypeInput = InputData.makeEnumType("clargsType","clargsTypeType",["text","input","output","prepend","postpend"])
    ClargsInput.addParam("type", ClargsTypeInput, True)

    ClargsInput.addParam("arg", InputData.StringType, False)
    ClargsInput.addParam("extension", InputData.StringType, False)
    inputSpecification.addSub(ClargsInput)
    ## End command line arguments tag

    ## Begin file arguments tag
    FileargsInput = InputData.parameterInputFactory("fileargs")

    FileargsTypeInput = InputData.makeEnumType("fileargsType", "fileargsTypeType",["input","output","moosevpp"])
    FileargsInput.addParam("type", FileargsTypeInput, True)

    FileargsInput.addParam("arg", InputData.StringType, False)
    FileargsInput.addParam("extension", InputData.StringType, False)
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

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Model.__init__(self,runInfoDict)
    self.executable         = ''   #name of the executable (abs path)
    self.preExec            = None   #name of the pre-executable, if any
    self.oriInputFiles      = []   #list of the original input files (abs path)
    self.workingDir         = ''   #location where the code is currently running
    self.outFileRoot        = ''   #root to be used to generate the sequence of output files
    self.currentInputFiles  = []   #list of the modified (possibly) input files (abs path)
    self.codeFlags          = None #flags that need to be passed into code interfaces(if present)
    self.printTag           = 'CODE MODEL'
    self.createWorkingDir   = True

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
        self.executable = str(child.value)
      if child.getName() =='preexec':
        self.preExec = child.value
      elif child.getName() == 'clargs':
        argtype = child.parameterValues['type']      if 'type'      in child.parameterValues else None
        arg     = child.parameterValues['arg']       if 'arg'       in child.parameterValues else None
        ext     = child.parameterValues['extension'] if 'extension' in child.parameterValues else None
        if argtype == None:
          self.raiseAnError(IOError,'"type" for clarg not specified!')
        elif argtype == 'text':
          if ext != None:
            self.raiseAWarning('"text" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if arg == None:
            self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          self.clargs['text']=arg
        elif argtype == 'input':
          if ext == None:
            self.raiseAnError(IOError,'"extension" for clarg '+argtype+' not specified! Enter filetype to be listed for this flag.')
          if arg == None:
            self.clargs['input']['noarg'].append(ext)
          else:
            if arg not in self.clargs['input'].keys():
              self.clargs['input'][arg]=[]
            self.clargs['input'][arg].append(ext)
        elif argtype == 'output':
          if arg == None:
            self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter flag for output file specification.')
          self.clargs['output'] = arg
        elif argtype == 'prepend':
          if ext != None:
            self.raiseAWarning('"prepend" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if arg == None:
            self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          self.clargs['pre'] = arg
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
      if '~' in self.executable:
        self.executable = os.path.expanduser(self.executable)
      abspath = os.path.abspath(str(self.executable))
      if os.path.exists(abspath):
        self.executable = abspath
      else:
        self.raiseAMessage('not found executable '+self.executable,'ExceptedError')
    if self.preExec is not None:
      if '~' in self.preExec:
        self.preExec = os.path.expanduser(self.preExec)
      abspath = os.path.abspath(self.preExec)
      if os.path.exists(abspath):
        self.preExec = abspath
      else:
        self.raiseAMessage('not found preexec '+self.preExec,'ExceptedError')
    self.code = Code.CodeInterfaces.returnCodeInterface(self.subType,self)
    self.code.readMoreXML(xmlNode) #TODO figure out how to handle this with InputData
    self.code.setInputExtension(list(a.strip('.') for b in (c for c in self.clargs['input'].values()) for a in b))
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
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    self.oriInputFiles = []
    for inputFile in inputFiles:
      subSubDirectory = os.path.join(self.workingDir,inputFile.subDirectory)
      ## Currently, there are no tests that verify the lines below can be hit
      ## It appears that the folders already exist by the time we get here,
      ## this could change, so we will leave this code here.
      ## -- DPM 8/2/17
      if inputFile.subDirectory.strip() != "" and not os.path.exists(subSubDirectory):
        os.mkdir(subSubDirectory)
      ##########################################################################
      shutil.copy(inputFile.getAbsFile(),subSubDirectory)
      self.oriInputFiles.append(copy.deepcopy(inputFile))
      self.oriInputFiles[-1].setPath(subSubDirectory)
    self.currentInputFiles        = None
    self.outFileRoot              = None

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
        os.mkdir(subSubDirectory)
      ##########################################################################
      newInputSet[index].setPath(subSubDirectory)
      shutil.copy(self.oriInputFiles[index].getAbsFile(),subSubDirectory)

    kwargs['subDirectory'] = subDirectory

    if 'SampledVars' in kwargs.keys():
      sampledVars = self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'],'input',False)

    newInput    = self.code.createNewInput(newInputSet,self.oriInputFiles,samplerType,**copy.deepcopy(kwargs))

    if 'SampledVars' in kwargs.keys() and len(self.alias['input'].keys()) != 0:
      kwargs['SampledVars'] = sampledVars

    return (newInput,kwargs)

  def _expandForWindows(self, origCommand):
    """
      Function to expand a command that has a #! to a windows runnable command
      @ In, origCommand, string, The command to check for expantion
      @ Out, commandSplit, string or String List, the expanded command or the original if not expanded.
    """
    if origCommand.strip() == '':
      return origCommand
    # In Windows Python, you can get some backslashes in your paths
    commandSplit = shlex.split(origCommand.replace("\\","/"))
    executable = commandSplit[0]

    if os.path.exists(executable):
      executableFile = open(executable, "r")

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
          self.raiseAMessage("winExecutable" + winExecutable)
          realExecutable[0] = winExecutable
        else:
          self.raiseAWarning("Could not find msys in "+os.getcwd())
        commandSplit = realExecutable + [executable] + commandSplit[1:]
        return commandSplit
    return origCommand

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
    bufferSize = kwargs['bufferSize']
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

    self.raiseAMessage('Execution command submitted:',command)
    if platform.system() == 'Windows':
      command = self._expandForWindows(command)
      self.raiseAMessage("modified command to" + repr(command))

    ## This code should be evaluated by the job handler, so it is fine to wait
    ## until the execution of the external subprocess completes.
    process = utils.pickleSafeSubprocessPopen(command, shell=True, stdout=outFileObject, stderr=outFileObject, cwd=localenv['PWD'], env=localenv)
    process.wait()

    returnCode = process.returncode
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
    outputFile = codeLogFile
    if 'finalizeCodeOutput' in dir(self.code):
      finalCodeOutputFile = self.code.finalizeCodeOutput(command, codeLogFile, metaData['subDirectory'])
      if finalCodeOutputFile:
        outputFile = finalCodeOutputFile

    ## If the run was successful
    if returnCode == 0:

      returnDict = {}
      ## This may be a tautology at this point --DPM 4/12/17
      if outputFile is not None:
        outFile = Files.CSV()
        ## Should we be adding the file extension here?
        outFile.initialize(outputFile+'.csv',self.messageHandler,path=metaData['subDirectory'])

        csvLoader = CsvLoader.CsvLoader(self.messageHandler)
        csvData = csvLoader.loadCsvFile(outFile)
        headers = csvLoader.getAllFieldNames()

        ## Numpy by default iterates over rows, thus we transpose the data and
        ## zip it with the headers in order to do store it very cleanly into a
        ## dictionary.
        for header,data in zip(headers, csvData.T):
          returnDict[header] = data

      self._replaceVariablesNamesWithAliasSystem(returnDict, 'input', True)
      self._replaceVariablesNamesWithAliasSystem(returnDict, 'output', True)

      ## The last thing before returning should be to delete the temporary log
      ## file and any other file the user requests to be cleared
      if deleteSuccessfulLogFiles:
        self.raiseAMessage(' Run "' +kwargs['prefix']+'" ended smoothly, removing log file!')
        codeLofFileFullPath = os.path.join(metaData['subDirectory'],codeLogFile)
        if os.path.exists(codeLofFileFullPath):
          os.remove(codeLofFileFullPath)

      ## Check if the user specified any file extensions for clean up
      for fileExt in fileExtensionsToDelete:
        if not fileExt.startswith("."):
          fileExt = "." + fileExt

        fileList = [ os.path.join(metaData['subDirectory'],f) for f in os.listdir(metaData['subDirectory']) if f.endswith(fileExt) ]

        for f in fileList:
          os.remove(f)

      returnValue = (kwargs['SampledVars'],returnDict)
      return returnValue
    else:
      self.raiseAMessage(" Process Failed "+str(command)+" returnCode "+str(returnCode))
      absOutputFile = os.path.join(sampleDirectory,outputFile)
      if os.path.exists(absOutputFile):
        self.raiseAMessage(repr(open(absOutputFile,"r").read()).replace("\\n","\n"))
      else:
        self.raiseAMessage(" No output " + absOutputFile)

      ## If you made it here, then the run must have failed
      return None

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    ## First, if the output is a data object, let's see what inputs it requests
    if options is None:
        options = {}
    if isinstance(output,Data):
      inputParams = output.getParaKeys('input')
      options["inputFile"] = self.currentInputFiles
    else:
      inputParams = []
    # create the export dictionary
    if 'exportDict' in options:
      exportDict = options['exportDict']
    else:
      exportDict = self.createExportDictionaryFromFinishedJob(finishedJob, True, inputParams)
    options['alias'] = self.alias
    metadata = exportDict['metadata']
    if metadata:
      options['metadata'] = metadata

    self.addOutputFromExportDictionary(exportDict, output, options, finishedJob.identifier)

  def collectOutputFromDict(self,exportDict,output,options=None):
    """
      Collect results from dictionary
      @ In, exportDict, dict, contains 'inputs','outputs','metadata'
      @ In, output, instance, the place to write to
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    prefix = exportDict.pop('prefix')
    #convert to *spaceParams instead of inputs,outputs
    if 'inputs' in exportDict.keys():
      inp = exportDict.pop('inputs')
      exportDict['inputSpaceParams'] = inp
    if 'outputs' in exportDict.keys():
      out = exportDict.pop('outputs')
      exportDict['outputSpaceParams'] = out
    if output.type == 'HDF5':
      output.addGroupDataObjects({'group':self.name+str(prefix)},exportDict,False)
    else:
      #point set
      for key in output.getParaKeys('inputs'):
        if key in exportDict['inputSpaceParams']:
          output.updateInputValue(key,exportDict['inputSpaceParams'][key],options)
        else:
          self.raiseAnError(Exception, "the input parameter "+key+" requested in the DataObject "+output.name+
                                       " has not been found among the Model input paramters ("+",".join(exportDict['inputSpaceParams'].keys())+"). Check your input!")
      for key in output.getParaKeys('outputs'):
        if key in exportDict['outputSpaceParams']:
          output.updateOutputValue(key,exportDict['outputSpaceParams'][key],options)
        else:
          self.raiseAnError(Exception, "the output parameter "+key+" requested in the DataObject "+output.name+
                                       " has not been found among the Model output paramters ("+",".join(exportDict['outputSpaceParams'].keys())+"). Check your input!")
      for key in exportDict['metadata']:
        output.updateMetadata(key,exportDict['metadata'][key], options)
      output.numAdditionalLoadPoints += 1 #prevents consistency problems for entries from restart

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
    prefix = kwargs['prefix'] if 'prefix' in kwargs else None
    uniqueHandler = kwargs['uniqueHandler'] if 'uniqueHandler' in kwargs.keys() else 'any'

    ## These two are part of the current metadata, so they will be added before
    ## the job is started, so that they will be captured in the metadata and match
    ## the current behavior of the system. If these are not desired, then this
    ## code can be moved to later.  -- DPM 4/12/17
    kwargs['executable'] = self.executable
    kwargs['outfile'] = None

    #TODO FIXME I don't think the extensions are the right way to classify files anymore, with the new Files
    #  objects.  However, this might require some updating of many CodeInterfaces``````           1  Interfaces as well.
    for index, inputFile in enumerate(myInput):
      if inputFile.getExt() in self.code.getInputExtension():
        kwargs['outfile'] = 'out~'+myInput[index].getBase()
        break
    if kwargs['outfile'] is None:
      self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                + self.subType +': ' + ' '.join(self.code.getInputExtension()))

    ## These kwargs are updated by createNewInput, so the job either should not
    ## have access to the metadata, or it needs to be updated from within the
    ## evaluateSample function, which currently is not possible since that
    ## function does not know about the job instance.
    metadata = copy.copy(kwargs)

    ## These variables should not be part of the metadata, so add them after
    ## we copy this dictionary (Caught this when running an ensemble model)
    ## -- DPM 4/11/17
    kwargs['bufferSize'] = jobHandler.runInfoDict['logfileBuffer']
    kwargs['precommand'] = jobHandler.runInfoDict['precommand']
    kwargs['postcommand'] = jobHandler.runInfoDict['postcommand']
    kwargs['delSucLogFiles'] = jobHandler.runInfoDict['delSucLogFiles']
    kwargs['deleteOutExtension'] = jobHandler.runInfoDict['deleteOutExtension']

    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter
    jobHandler.addJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, prefix, metadata=metadata, modulesToImport=self.mods, uniqueHandler=uniqueHandler)
    self.raiseAMessage('job "' + str(prefix) + '" submitted!')
