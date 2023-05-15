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
  MultiRun module
  This module contains the Step that is aimed to be employed when
  either Sampling or Optimization analyses are requested
  results of a RAVEN (or not) analysis.
  Created on May 6, 2021
  @author: alfoa
  supercedes Steps.py from alfoa (2/16/2013)
"""
#External Modules------------------------------------------------------------------------------------
import time
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SingleRun import SingleRun
from .. import Models
from ..utils import utils
from ..OutStreams import OutStreamEntity
#Internal Modules End--------------------------------------------------------------------------------


class MultiRun(SingleRun):
  """
    This class implements one step of the simulation where several runs are needed
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._samplerInitDict = {}          # dictionary that gets sent to the initialization of the sampler
    self.counter = 0                    # counter of the runs already performed
    self._outputCollectionLambda = None # lambda function list to collect the output without checking the type
    self.printTag = 'STEP MULTIRUN'

  def _localInputAndCheckParam(self, paramInput):
    """
      Specialized reading, input consistency check and initialization of what will not change during
      the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    SingleRun._localInputAndCheckParam(self,paramInput)
    if self.samplerType not in [item[0] for item in self.parList]:
      self.raiseAnError(IOError, 'Multi-run not possible without a sampler or optimizer!')

  def _initializeSampler(self, inDictionary):
    """
      Method to initialize the sampler
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    if 'SolutionExport' in inDictionary:
      self._samplerInitDict['solutionExport']=inDictionary['SolutionExport']

    inDictionary[self.samplerType].initialize(**self._samplerInitDict)
    self.raiseADebug(f'for the role of sampler the item of class {inDictionary[self.samplerType].type} and name {inDictionary[self.samplerType].name} has been initialized')
    self.raiseADebug(f'Sampler initialization dictionary: {self._samplerInitDict}')

  def _localInitializeStep(self, inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances (or list of instances if more than one is allowed)
      for each possible role supported in the step (dictionary keywords)
      The role of _localInitializeStep is to call the initialize method if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    SingleRun._localInitializeStep(self, inDictionary)
    # check that no input data objects are also used as outputs?
    for out in inDictionary['Output']:
      if out.type not in ['PointSet', 'HistorySet', 'DataSet']:
        continue
      for inp in inDictionary['Input']:
        if inp.type not in ['PointSet', 'HistorySet', 'DataSet']:
          continue
        if inp == out:
          self.raiseAnError(IOError,'The same data object should not be used as both <Input> and <Output> in the same MultiRun step! ' \
              + f'Step: "{self.name}", DataObject: "{out.name}"')
    self.counter = 0
    self._samplerInitDict['externalSeeding'] = self.initSeed
    self._initializeSampler(inDictionary)
    #generate lambda function list to collect the output without checking the type
    self._outputCollectionLambda = []
    # set up output collection lambdas
    for outIndex, output in enumerate(inDictionary['Output']):
      if not isinstance(output, OutStreamEntity):
        if 'SolutionExport' in inDictionary and output.name == inDictionary['SolutionExport'].name:
          self._outputCollectionLambda.append((lambda x:None, outIndex))
        else:
          self._outputCollectionLambda.append( (lambda x: inDictionary['Model'].collectOutput(x[0],x[1]), outIndex) )
      else:
        self._outputCollectionLambda.append((lambda x: x[1].addOutput(), outIndex))
    self._registerMetadata(inDictionary)
    self.raiseADebug(f'Generating input batch of size {inDictionary["jobHandler"].runInfoDict["batchSize"]}')
    # set up and run the first batch of samples
    # FIXME this duplicates a lot of code from _locatTakeAstepRun, which should be consolidated
    # first, check and make sure the model is ready
    model = inDictionary['Model']
    if isinstance(model,Models.ROM):
      if not model.amITrained:
        model.raiseAnError(RuntimeError, f'ROM model "{model.name}" has not been trained yet, so it cannot be sampled!'+\
                                        ' Use a RomTrainer step to train it.')
    for inputIndex in range(inDictionary['jobHandler'].runInfoDict['batchSize']):
      if inDictionary[self.samplerType].amIreadyToProvideAnInput():
        try:
          newInput = self._findANewInputToRun(inDictionary[self.samplerType], inDictionary['Model'], inDictionary['Input'], inDictionary['Output'], inDictionary['jobHandler'])
          if newInput is not None:
            inDictionary["Model"].submit(newInput, inDictionary[self.samplerType].type, inDictionary['jobHandler'], **copy.deepcopy(inDictionary[self.samplerType].inputInfo))
            self.raiseADebug(f'Submitted input {inputIndex+1}')
        except utils.NoMoreSamplesNeeded:
          self.raiseAMessage('Sampler returned "NoMoreSamplesNeeded".  Continuing...')

  @profile
  def _localTakeAstepRun(self, inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    jobHandler = inDictionary['jobHandler']
    model      = inDictionary['Model'     ]
    inputs     = inDictionary['Input'     ]
    outputs    = inDictionary['Output'    ]
    sampler    = inDictionary[self.samplerType]
    # check to make sure model can be run
    # first, if it's a ROM, check that it's trained
    if isinstance(model, Models.ROM):
      if not model.amITrained:
        model.raiseAnError(RuntimeError, f'ROM model "{model.name}" has not been trained yet, so it cannot be sampled!'+\
                                        ' Use a RomTrainer step to train it.')
    # Every reportDeltaTime seconds, write some debug information for this step.
    reportDeltaTime = 60.0
    nextReportTime = time.time() + reportDeltaTime
    # run step loop
    while True:
      # collect finished jobs
      finishedJobs = jobHandler.getFinished()

      # FIXME: THE BATCH STRATEGY IS TOO INTRUSIVE. A MORE ELEGANT WAY NEEDS TO BE FOUND (E.G. REALIZATION OBJECT)
      for finishedJobObjs in finishedJobs:
        # NOTE: HERE WE RETRIEVE THE JOBS. IF BATCHING, THE ELEMENT IN finishedJobs is a LIST
        #       WE DO THIS in this way because:
        #           in case of BATCHING, the finalizeActualSampling method MUST BE called ONCE/BATCH
        #           otherwise, the finalizeActualSampling method MUST BE called ONCE/job
        # FIXME: This method needs to be improved since it is very intrusise
        if type(finishedJobObjs).__name__ in 'list':
          finishedJobList = finishedJobObjs
          self.raiseADebug(f'BATCHING: Collecting JOB batch named "{finishedJobList[0].groupId}".')
        else:
          finishedJobList = [finishedJobObjs]
        currentFailures = []
        for finishedJob in finishedJobList:
          finishedJob.trackTime('step_collected')
          # update number of collected runs
          self.counter += 1
          # collect run if it succeeded
          if finishedJob.getReturnCode() == 0:
            for myLambda, outIndex in self._outputCollectionLambda:
              myLambda([finishedJob,outputs[outIndex]])
              self.raiseADebug(f'Just collected job {finishedJob.identifier} and sent to output "{inDictionary["Output"][outIndex].name}"')
          # pool it if it failed, before we loop back to "while True" we'll check for these again
          else:
            self.raiseADebug(f'the job "{finishedJob.identifier}" has failed.')
            if self.failureHandling['fail']:
              # is this sampler/optimizer able to handle failed runs? If not, add the failed run in the pool
              if not sampler.ableToHandelFailedRuns:
                # add run to a pool that can be sent to the sampler later
                self.failedRuns.append(copy.copy(finishedJob))
            else:
              if finishedJob.identifier not in self.failureHandling['jobRepetitionPerformed']:
                self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] = 1
              if self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] <= self.failureHandling['repetitions']:
                # we re-add the failed job
                jobHandler.reAddJob(finishedJob)
                self.raiseAWarning('As prescribed in the input, trying to re-submit the job "'+finishedJob.identifier+'". Trial '+
                                 str(self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier]) +'/'+str(self.failureHandling['repetitions']))
                self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] += 1
              else:
                # is this sampler/optimizer able to handle failed runs? If not, add the failed run in the pool
                if not sampler.ableToHandelFailedRuns:
                  self.failedRuns.append(copy.copy(finishedJob))
                self.raiseAWarning('The job "'+finishedJob.identifier+'" has been submitted '+ str(self.failureHandling['repetitions'])+' times, failing all the times!!!')
            if sampler.ableToHandelFailedRuns:
              self.raiseAWarning('The sampler/optimizer "'+sampler.type+'" is able to handle failed runs!')
            #collect the failed job index from the list
            currentFailures.append(finishedJobList.index(finishedJob))
        if currentFailures:
          # In the previous approach, the job was removed directly in the list of jobs on which we were iterating,
          # determining a messing-up of the loop. Since now we collect only the indices
          # I need to reverse it so I can remove the jobs starting from the last and back.
          # For example, if currentFailures=[0,2,4] If we do not sort it (i.e. correntFailures = [4,2,0])
          # when we start removing the jobs from the list we would mess up the indices...=> If I remove 0 first,
          # then the index 2 should become 1 and index 4 should become 3 (and so on)
          currentFailures.sort(reverse=True)
          for idx in currentFailures:
            finishedJobList.pop(idx)

        if type(finishedJobObjs).__name__ in 'list': # TODO: should be consistent, if no batching should batch size be 1 or 0 ?
          # if sampler claims it's batching, then only collect once, since it will collect the batch
          # together, not one-at-a-time
          # FIXME: IN HERE WE SEND IN THE INSTANCE OF THE FIRST JOB OF A BATCH
          # FIXME: THIS IS DONE BECAUSE CURRENTLY SAMPLERS/OPTIMIZERS RETRIEVE SOME INFO from the Runner instance but it can be
          # FIXME: dangerous if the sampler/optimizer requires info from each job. THIS MUST BE FIXED.
          if finishedJobList:
            sampler.finalizeActualSampling(finishedJobList[0],model,inputs)
        else:
          # sampler isn't intending to batch, so we send them in one-at-a-time as per normal
          for finishedJob in finishedJobList:
            # finalize actual sampler
            sampler.finalizeActualSampling(finishedJob,model,inputs)
        for finishedJob in finishedJobList:
          finishedJob.trackTime('step_finished')

        # terminate jobs as requested by the sampler, in case they're not needed anymore
        # TODO is this a safe place to put this?
        # If it's placed after adding new jobs and IDs are re-used i.e. for failed tests,
        # -> then the new jobs will be killed if this is placed after new job submission!
        jobHandler.terminateJobs(sampler.getJobsToEnd(clear=True))

        # add new jobs, for DET-type samplers
        # put back this loop (do not take it away again. it is NEEDED for NOT-POINT samplers(aka DET)). Andrea
        # NOTE for non-DET samplers, this check also happens outside this collection loop
        if sampler.onlySampleAfterCollecting:
          self._addNewRuns(sampler, model, inputs, outputs, jobHandler, inDictionary)
      # END for each collected finished run ...
      # If all of the jobs given to the job handler have finished, and the sampler
      # has nothing else to provide, then we are done with this step.
      if jobHandler.isFinished() and not sampler.amIreadyToProvideAnInput():
        self.raiseADebug(f'Sampling finished with {jobHandler.numSubmitted()} runs submitted, {jobHandler.numRunning()} jobs running, and {len(jobHandler.getFinishedNoPop())} completed jobs waiting to be processed.')
        break
      currentTime = time.time()
      if currentTime > nextReportTime:
        nextReportTime = currentTime + reportDeltaTime
        numRunning = jobHandler.numRunning()
        numTotalRunning = jobHandler.numRunningTotal()
        self.raiseADebug((f"Continuing to run. isFinished: {jobHandler.isFinished()} "
                          f"running: {numRunning} rest running: {numTotalRunning - numRunning} "
                          f"unclaimed runs: {len(jobHandler.getFinishedNoPop())} "
                          f"queued: {jobHandler._numQueuedTotal()}"))
      # Note: calling amIreadyToProvideAnInput can change results,
      # but might be helpful for debugging sometimes
      # "sampler ready with input: %r" sampler.amIreadyToProvideAnInput()
      if not sampler.onlySampleAfterCollecting:
        # NOTE for some reason submission outside collection breaks the DET
        # however, it is necessary i.e. batch sampling
        self._addNewRuns(sampler, model, inputs, outputs, jobHandler, inDictionary, verbose=False)
      time.sleep(self.sleepTime)
    # END while loop that runs the step iterations (collection and submission-for-DET)
    # if any collected runs failed, let the sampler treat them appropriately, and any other closing-out actions
    sampler.finalizeSampler(self.failedRuns)

  def _addNewRuns(self, sampler, model, inputs, outputs, jobHandler, inDictionary, verbose=True):
    """
      Checks for open spaces and adds new runs to jobHandler queue (via model.submit currently)
      @ In, sampler, Sampler, the sampler in charge of generating the sample
      @ In, model, Model, the model in charge of evaluating the sample
      @ In, inputs, object, the raven object used as the input in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, outputs, object, the raven object used as the output in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, jobHandler, object, the raven object used to handle jobs
      @ In, inDictionary, dict, additional step objects map
      @ In, verbose, bool, optional, if True print DEBUG statements
      @ Out, None
    """
    isEnsemble = isinstance(model, Models.EnsembleModel)
    # In order to ensure that the queue does not grow too large, we will
    # employ a threshold on the number of jobs the jobHandler can take,
    # in addition, we cannot provide more jobs than the sampler can provide.
    # So, we take the minimum of these two values.
    if verbose:
      self.raiseADebug('Testing if the sampler is ready to generate a new input')
    for _ in range(min(jobHandler.availability(isEnsemble), sampler.endJobRunnable())):
      if sampler.amIreadyToProvideAnInput():
        try:
          newInput = self._findANewInputToRun(sampler, model, inputs, outputs, jobHandler)
          if newInput is not None:
            model.submit(newInput, inDictionary[self.samplerType].type, jobHandler, **copy.deepcopy(sampler.inputInfo))
        except utils.NoMoreSamplesNeeded:
          self.raiseAMessage(' ... Sampler returned "NoMoreSamplesNeeded".  Continuing...')
          break
      else:
        if verbose:
          self.raiseADebug(' ... sampler has no new inputs currently.')
        break
    else:
      if verbose:
        self.raiseADebug(' ... no available JobHandler spots currently (or the Sampler is done.)')

  def _findANewInputToRun(self, sampler, model, inputs, outputs, jobHandler):
    """
      Repeatedly calls Sampler until a new run is found or "NoMoreSamplesNeeded" is raised.
      @ In, sampler, Sampler, the sampler in charge of generating the sample
      @ In, model, Model, the model in charge of evaluating the sample
      @ In, inputs, object, the raven object used as the input in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, outputs, object, the raven object used as the output in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, jobHandler, object, the raven object used to handle jobs
      @ Out, newInp, list, list containing the new inputs (or None if a restart)
    """
    # The value of "found" determines what the Sampler is ready to provide.
    #  case 0: a new sample has been discovered and can be run, and newInp is a new input list.
    #  case 1: found the input in restart, and newInp is a realization dictionary of data to use
    found, newInp = sampler.generateInput(model,inputs)
    if found == 1:
      kwargs = copy.deepcopy(sampler.inputInfo)
      # "submit" the finished run
      jobHandler.addFinishedJob(newInp, metadata=kwargs)
      return None
      # NOTE: we return None here only because the Sampler's "counter" is not correctly passed
      # through if we add several samples at once through the restart. If we actually returned
      # a Realization object from the Sampler, this would not be a problem. - talbpaul
    return newInp

  def flushStep(self):
    """
      Reset Step attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flushStep()
    self._samplerInitDict = {}
    self.counter = 0
    self._outputCollectionLambda = None
