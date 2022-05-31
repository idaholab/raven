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
  SingleRun module
  This module contains the step that is aimed to evaluate a Model just
  once (e.g. single code run, PostProcessing, etc.)
  Created on May 6, 2021
  @author: alfoa
  supercedes Steps.py from alfoa (2/16/2013)
"""
# External Modules----------------------------------------------------------------------------------
import atexit
import time
import os
import copy
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .. import Models
from .Step import Step
from ..utils import utils
from ..OutStreams import OutStreamEntity
from ..Databases import Database
# Internal Modules End------------------------------------------------------------------------------

class SingleRun(Step):
  """
    This is the step that will perform just one evaluation
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.samplerType = 'Sampler'
    self.failedRuns = []
    self.lockedFileName = "ravenLocked.raven"
    self.printTag = 'STEP SINGLERUN'

  def _localInputAndCheckParam(self, paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    self.raiseADebug('the mapping used in the model for checking the compatibility of usage should be more similar to self.parList to avoid the double mapping below', 'FIXME')
    found     = 0
    rolesItem = []
    # collect model, other entries
    for index, parameter in enumerate(self.parList):
      if parameter[0]=='Model':
        found += 1
        modelIndex = index
      else:
        rolesItem.append(parameter[0])
    # test the presence of one and only one model
    if found > 1:
      self.raiseAnError(IOError, f'Only one model is allowed for the step named {self.name}')
    elif found == 0:
      self.raiseAnError(IOError, f'No model has been found for the step named {self.name}')
    # clarify run by roles
    roles      = set(rolesItem)
    if 'Optimizer' in roles:
      self.samplerType = 'Optimizer'
      if 'Sampler' in roles:
        self.raiseAnError(IOError, f'Only Sampler or Optimizer is alloweed for the step named {self.name}')
    # if single run, make sure model is an instance of Code class
    if self.type == 'SingleRun':
      if self.parList[modelIndex][2] != 'Code':
        self.raiseAnError(IOError, '<SingleRun> steps only support running "Code" model types!  Consider using a <MultiRun> step using a "Custom" sampler for other models.')
      if 'Optimizer' in roles or 'Sampler' in roles:
        self.raiseAnError(IOError, '<SingleRun> steps does not allow the usage of <Sampler> or <Optimizer>!  Consider using a <MultiRun> step.')
      if 'SolutionExport' in roles:
        self.raiseAnError(IOError, '<SingleRun> steps does not allow the usage of <SolutionExport>!  Consider using a <MultiRun> step with a <Sampler>/<Optimizer> that allows its usage.')
    # build entry list for verification of correct input types
    toBeTested = {}
    for role in roles:
      toBeTested[role]=[]
    for  myInput in self.parList:
      if myInput[0] in rolesItem:
        toBeTested[ myInput[0]].append({'class': myInput[1],'type': myInput[2]})
    # use the models static testing of roles compatibility
    for role in roles:
      if role not in self._excludeFromModelValidation:
        Models.validate(self.parList[modelIndex][2], role, toBeTested[role])
    self.raiseADebug('reactivate check on Input as soon as loadCsv gets out from the PostProcessor models!')
    if 'Output' not in roles:
      self.raiseAnError(IOError, 'It is not possible to run without an Output!')

  def _localInitializeStep(self, inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    # Model initialization
    modelInitDict = {'Output': inDictionary['Output']}
    if 'SolutionExport' in inDictionary:
      modelInitDict['SolutionExport'] = inDictionary['SolutionExport']
    if inDictionary['Model'].createWorkingDir:
      currentWorkingDirectory = os.path.join(inDictionary['jobHandler'].runInfoDict['WorkingDir'],
                                             inDictionary['jobHandler'].runInfoDict['stepName'])
      workingDirReady = False
      alreadyTried = False
      while not workingDirReady:
        try:
          os.mkdir(currentWorkingDirectory)
          workingDirReady = True
        except FileExistsError:
          if utils.checkIfPathAreAccessedByAnotherProgram(currentWorkingDirectory,3.0):
            self.raiseAWarning(f'directory {currentWorkingDirectory} is likely used by another program!!! ')
          if utils.checkIfLockedRavenFileIsPresent(currentWorkingDirectory,self.lockedFileName):
            self.raiseAnError(RuntimeError, self, f"another instance of RAVEN is running in the working directory {currentWorkingDirectory}. Please check your input!")
          if self._clearRunDir and not alreadyTried:
            self.raiseAWarning(f'The calculation run directory {currentWorkingDirectory} already exists, ' +
                              'clearing existing files. This action can be disabled through the RAVEN Step input.')

            utils.removeDir(currentWorkingDirectory)
            alreadyTried = True
            continue
          else:
            if alreadyTried:
              self.raiseAWarning(f'The calculation run directory {currentWorkingDirectory} already exists, ' +
                                'and was not able to be removed. ' +
                                'Files present in this directory may be replaced, and error handling may not occur as expected.')
            else:
              self.raiseAWarning(f'The calculation run directory {currentWorkingDirectory} already exists. ' +
                                'Files present in this directory may be replaced, and error handling may not occur as expected.')
            workingDirReady = True
          # register function to remove the locked file at the end of execution
        atexit.register(utils.removeFile,os.path.join(currentWorkingDirectory,self.lockedFileName))
    inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'],modelInitDict)

    self.raiseADebug(f'for the role Model, the item of class {inDictionary["Model"].type} and name {inDictionary["Model"].name} has been initialized')

    #Database initialization
    for i in range(len(inDictionary['Output'])):
      if isinstance(inDictionary['Output'][i], Database):
        inDictionary['Output'][i].initialize(self.name)
      elif isinstance(inDictionary['Output'][i], OutStreamEntity):
        inDictionary['Output'][i].initialize(inDictionary)
      self.raiseADebug(f'for the role Output the item of class {inDictionary["Output"][i].type} and name {inDictionary["Output"][i].name} has been initialized')
    self._registerMetadata(inDictionary)

  def _localTakeAstepRun(self, inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    jobHandler = inDictionary['jobHandler']
    model      = inDictionary['Model'     ]
    sampler    = inDictionary.get(self.samplerType,None)
    inputs     = inDictionary['Input'     ]
    outputs    = inDictionary['Output'    ]

    # the input provided by a SingleRun is simply the file to be run.  model.run, however, expects stuff to perturb.
    # get an input to run -> different between SingleRun and PostProcessor runs
    # if self.type == 'SingleRun':
    #   newInput = model.createNewInput(inputs,'None',**{'SampledVars':{},'additionalEdits':{}})
    # else:
    #   newInput = inputs

    # The single run should still collect its SampledVars for the output maybe?
    # The problem here is when we call Code.collectOutput(), the sampledVars
    # is empty... The question is where do we ultimately get this information
    # the input object's input space or the desired output of the Output object?
    # I don't think all of the outputs need to specify their domain, so I suppose
    # this should default to all of the ones in the input? Is it possible to
    # get an input field in the outputs variable that is not in the inputs
    # variable defined above? - DPM 4/6/2017
    # empty dictionary corresponds to sampling data in MultiRun
    model.submit(inputs, None, jobHandler, **{'SampledVars': {'prefix':'None'}, 'additionalEdits': {}})
    while True:
      finishedJobs = jobHandler.getFinished()
      for finishedJob in finishedJobs:
        if finishedJob.getReturnCode() == 0:
          # if the return code is > 0 => means the system code crashed... we do not want to make the statistics poor => we discard this run
          for output in outputs:
            if not isinstance(output, OutStreamEntity):
              model.collectOutput(finishedJob, output)
            else:
              output.addOutput()
        else:
          self.raiseADebug(f'the job "{finishedJob.identifier}" has failed.')
          if self.failureHandling['fail']:
            #add run to a pool that can be sent to the sampler later
            self.failedRuns.append(copy.copy(finishedJob))
          else:
            if finishedJob.identifier not in self.failureHandling['jobRepetitionPerformed']:
              self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] = 1
            if self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] <= self.failureHandling['repetitions']:
              # we re-add the failed job
              jobHandler.reAddJob(finishedJob)
              self.raiseAWarning(f'As prescribed in the input, trying to re-submit the job "{finishedJob.identifier}". Trial {self.failureHandling["jobRepetitionPerformed"][finishedJob.identifier]}/{self.failureHandling["repetitions"]}')
              self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] += 1
            else:
              #add run to a pool that can be sent to the sampler later
              self.failedRuns.append(copy.copy(finishedJob))
              self.raiseAWarning(f'The job "{finishedJob.identifier}" has been submitted {self.failureHandling["repetitions"]} times, failing every time!!!')
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
        break
      time.sleep(self.sleepTime)
    if sampler is not None:
      sampler.handleFailedRuns(self.failedRuns)
    else:
      if len(self.failedRuns)>0:
        self.raiseAWarning(f'There were {len(self.failedRuns)} failed runs!')

  def _localGetInitParams(self):
    """
      Place here a specialization of the exporting of what in the step is added to the initial parameters
      the printing format of paramDict is key: paramDict[key]
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}

  def flushStep(self):
    """
      Reset SingleRun attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flushStep()
    self.failedRuns = []
