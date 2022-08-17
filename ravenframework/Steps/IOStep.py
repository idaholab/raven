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
  IOStep module
  This module contains the Step that is aimed to be perform
  IO processes
  Created on May 6, 2021
  @author: alfoa
  supercedes Steps.py from alfoa (2/16/2013)
"""
# External Modules----------------------------------------------------------------------------------
import os
import pickle
import cloudpickle
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .. import Files
from .. import Models
from .Step import Step
from ..OutStreams import OutStreamEntity
from ..DataObjects import DataObject
from ..Databases import Database
# Internal Modules End------------------------------------------------------------------------------

class IOStep(Step):
  """
    This step is used to extract or push information from/into a Database,
    or from a directory, or print out the data to an OutStream
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'STEP IOCOMBINED'
    self.fromDirectory = None
    self.actionType = None

  def __getOutputs(self, inDictionary):
    """
      Utility method to get all the instances marked as Output
      @ In, inDictionary, dict, dictionary of all instances
      @ Out, outputs, list, list of Output instances
    """
    outputs = []
    for out in inDictionary['Output']:
      if not isinstance(out, OutStreamEntity):
        outputs.append(out)
    return outputs

  def _localInitializeStep(self,inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    # check if #inputs == #outputs
    # collect the outputs without outstreams
    outputs = self.__getOutputs(inDictionary)
    databases = set()
    self.actionType = []
    errTemplate = 'In Step "{name}": When the Input is {inp}, this step accepts only {okay} as Outputs, ' +\
                  'but received "{received}" instead!'
    if len(inDictionary['Input']) != len(outputs) and len(outputs) > 0:
      self.raiseAnError(IOError,'In Step named ' + self.name + \
          ', the number of Inputs != number of Outputs, and there are Outputs. '+\
          f'Inputs: {len(inDictionary["Input"])} Outputs: {len(outputs)}')
    # determine transfer and if it is a valid combination
    for i in range(len(outputs)):
      # from Database to ...
      if isinstance(inDictionary['Input'][i], Database):
        # ... dataobject
        if isinstance(outputs[i], DataObject.DataObject):
          self.actionType.append('Database-dataObjects')
        # ... anything else
        else:
          self.raiseAnError(IOError, errTemplate.format(name=self.name,
                                                        inp='Database',
                                                        okay='DataObjects',
                                                        received=inDictionary['Output'][i].type))
      # from DataObject to ...
      elif  isinstance(inDictionary['Input'][i], DataObject.DataObject):
        # ... Database
        if isinstance(outputs[i], Database):
          self.actionType.append('dataObjects-Database')
        # ... anything else
        else:
          self.raiseAnError(IOError, errTemplate.format(name=self.name,
                                                       inp='DataObjects',
                                                       okay='Database',
                                                       received=inDictionary['Output'][i].type))
      # from ROM model to ...
      elif isinstance(inDictionary['Input'][i], (Models.ROM, Models.ExternalModel)):
        # ... file
        if isinstance(outputs[i],Files.File):
          if 'PYOMO' == outputs[i].getType().upper():
            self.actionType.append('MODEL-PYOMO')
          if 'FMU' == outputs[i].getType().upper():
            self.actionType.append('MODEL-FMU')
          else:
            self.actionType.append('MODEL-FILES')
        # ... data object
        elif isinstance(outputs[i], DataObject.DataObject):
          self.actionType.append('ROM-dataObjects')
        # ... anything else
        else:
          self.raiseAnError(IOError, errTemplate.format(name=self.name,
                                                        inp='ROM',
                                                        okay='Files or DataObjects',
                                                        received=inDictionary['Output'][i].type))
      # from File to ...
      elif isinstance(inDictionary['Input'][i],Files.File):
        # ... ROM
        if isinstance(outputs[i], (Models.ROM, Models.ExternalModel)):
          self.actionType.append('FILES-MODEL')
        # ... dataobject
        elif isinstance(outputs[i],DataObject.DataObject):
          self.actionType.append('FILES-dataObjects')
        # ... anything else
        else:
          self.raiseAnError(IOError, errTemplate.format(name=self.name,
                                                        inp='Files',
                                                        okay='MODEL',
                                                        received=inDictionary['Output'][i].type))
      # from anything else to anything else
      else:
        self.raiseAnError(IOError, f'In Step "{self.name}": This step accepts only Database, DataObjects, ROM, or Files as Input. Received "{inDictionary["Input"][i].name}" of type "{inDictionary["Input"][i].type}" instead!')
    # check actionType for fromDirectory
    if self.fromDirectory and len(self.actionType) == 0:
      self.raiseAnError(IOError, f'In Step named {self.name}. "fromDirectory" attribute provided but not conversion action is found (remove this atttribute for OutStream actions only"')
    #Initialize all the Database outputs.
    for i in range(len(outputs)):
      if isinstance(inDictionary['Output'][i], Database):
        if outputs[i].name not in databases:
          databases.add(outputs[i].name)
          outputs[i].initialize(self.name)
          self.raiseADebug(f'for the role Output the item of class {outputs[i].type} and name {outputs[i].name} has been initialized')

    # if have a fromDirectory and are a dataObjects-*, need to load data
    if self.fromDirectory:
      for i in range(len(inDictionary['Input'])):
        if self.actionType[i].startswith('dataObjects-'):
          inInput = inDictionary['Input'][i]
          filename = os.path.join(self.fromDirectory, inInput.name)
          inInput.load(filename, style='csv')

    # Initialize all the OutStreams
    for output in inDictionary['Output']:
      if isinstance(output, OutStreamEntity):
        output.initialize(inDictionary)
        self.raiseADebug(f'for the role Output the item of class {output.type} and name {output.name} has been initialized')
    # register metadata
    self._registerMetadata(inDictionary)

  def _localTakeAstepRun(self, inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    outputs = self.__getOutputs(inDictionary)
    for i in range(len(outputs)):
      if self.actionType[i] == 'Database-dataObjects':
        # inDictionary['Input'][i] is Database, outputs[i] is a DataObjects
        inDictionary['Input'][i].loadIntoData(outputs[i])
      elif self.actionType[i] == 'dataObjects-Database':
        # inDictionary['Input'][i] is a dataObjects, outputs[i] is Database
        outputs[i].saveDataToFile(inDictionary['Input'][i])

      elif self.actionType[i] == 'ROM-dataObjects':
        # inDictionary['Input'][i] is a ROM, outputs[i] is dataObject
        # print information from the ROM to the data set or associated XML.
        romModel = inDictionary['Input'][i]
        # get non-pointwise data (to place in XML metadata of data object)
        # TODO how can user ask for particular information?
        xml = romModel.writeXML(what='all')
        self.raiseADebug(f'Adding meta "{xml.getRoot().tag}" to output "{outputs[i].name}"')
        outputs[i].addMeta(romModel.name, node = xml)
        # get pointwise data (to place in main section of data object)
        romModel.writePointwiseData(outputs[i])

      elif self.actionType[i] == 'MODEL-FILES':
        # inDictionary['Input'][i] is a ROM, outputs[i] is Files
        # pickle the ROM
        # check the ROM is trained first
        if isinstance(inDictionary['Input'][i],Models.ROM) and not inDictionary['Input'][i].amITrained:
          self.raiseAnError(RuntimeError, f'Pickled rom "{inDictionary["Input"][i].name}" was not trained!  Train it before pickling and unpickling using a RomTrainer step.')
        fileobj = outputs[i]
        fileobj.open(mode='wb+')
        cloudpickle.dump(inDictionary['Input'][i], fileobj, protocol=pickle.HIGHEST_PROTOCOL)
        fileobj.flush()
        fileobj.close()

      elif self.actionType[i] == 'MODEL-PYOMO':
        outfile = open(outputs[i].getAbsFile(),"w")
        outfile.write(inDictionary['Input'][i].writePyomoGreyModel())
        outfile.close()

      elif self.actionType[i] == 'MODEL-FMU':
        #check the ROM is trained first (if ExternalModel no check it is performed)
        if isinstance(inDictionary['Input'][i],Models.ROM) and not inDictionary['Input'][i].amITrained:
          self.raiseAnError(RuntimeError, f'Pickled rom "{inDictionary["Input"][i].name}" was not trained!  Train it before pickling and unpickling using a RomTrainer step.')
        self.raiseAMessage(f'Exporting Model "{inDictionary["Input"][i].name}" as FMU named "{outputs[i].name}"')
        from ..utils.fmuExporter import FMUexporter
        fdir = inDictionary['jobHandler'].runInfoDict['FrameworkDir']
        fmuexec = FMUexporter(**{'model': inDictionary['Input'][i],'executeMethod': 'evaluate', 'workingDir': outputs[i].getPath(), 'frameworkDir': fdir, 'keepModule': True})
        fmuexec.buildFMU(outputs[i].getAbsFile())

      elif self.actionType[i] == 'FILES-MODEL':
        # inDictionary['Input'][i] is a Files, outputs[i] is ROM or ExternalModel
        # unpickle the ROM
        fileobj = inDictionary['Input'][i]
        unpickledObj = pickle.load(open(fileobj.getAbsFile(),'rb+'))
        if not isinstance(unpickledObj, (Models.ROM, Models.ExternalModel)):
          # DEBUGG
          # the following will iteratively check the size of objects being unpickled
          # this is quite useful for finding memory crashes due to parallelism
          # so I'm leaving it here for reference
          # print('CHECKING SIZE OF', unpickledObj)
          # target = unpickledObj# .supervisedEngine.supervisedContainer[0]._macroSteps[2025]._roms[0]
          # print('CHECKING SIZES')
          # from utils.Debugging import checkSizesWalk
          # checkSizesWalk(target, 1, str(type(target)), tol=2e4)
          # print('*'*80)
          # crashme
          # /DEBUGG
          self.raiseAnError(RuntimeError, f'Pickled object in "{fileobj}" is not a ROM.  Exiting ...')
        if isinstance(unpickledObj,Models.ROM) and not unpickledObj.amITrained:
          self.raiseAnError(RuntimeError, f'Pickled rom "{unpickledObj.name}" was not trained!  Train it before pickling and unpickling using a RomTrainer step.')
        # copy model (same for any internal model (Dummy model derived classes)
        outputs[i]._copyModel(unpickledObj)

      elif self.actionType[i] == 'FILES-dataObjects':
        # inDictionary['Input'][i] is a Files, outputs[i] is PointSet
        # load a CSV from file
        infile = inDictionary['Input'][i]
        options = {'fileToLoad':infile}
        outputs[i].load(inDictionary['Input'][i].getPath(),'csv',**options)

      else:
        # unrecognized, and somehow not caught by the step reader.
        self.raiseAnError(IOError, f"Unknown action type {self.actionType[i]}")

    for output in inDictionary['Output']:
      if isinstance(output, OutStreamEntity):
        output.addOutput()

  def _localGetInitParams(self):
    """
      Place here a specialization of the exporting of what in the step is added to the initial parameters
      the printing format of paramDict is key: paramDict[key]
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}

    return paramDict # no inputs

  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    if 'fromDirectory' in paramInput.parameterValues:
      self.fromDirectory = paramInput.parameterValues['fromDirectory']

  def flushStep(self):
    """
      Reset Step attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flushStep()
    self.actionType = None
