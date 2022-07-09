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
  This module contains the tools to export RAVEN models (e.g. ExternalModel, ROMs, etc.) as FMI/FMU
  Created on May 6, 2021
  @author: alfoa
"""
#External Modules------------------------------------------------------------------------------------
import tempfile
from pathlib import Path
from typing import Union
import cloudpickle
import pickle
import os
#  pythonfmu (this links to the one in framework/contrib)
from pythonfmu.fmi2slave import FMI2_MODEL_OPTIONS
from pythonfmu.builder import FmuBuilder
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..BaseClasses.MessageUser import MessageUser
#Internal Modules End--------------------------------------------------------------------------------

# create data type that is returned from pythonfmu
FilePath = Union[str, Path]

class FMUexporter(MessageUser):
  """
    FMU exporter for RAVEN
  """
  def __init__(self, **kwargs):
    """
      FMU builder constructor
      @ In, **kwargs, dict, kwarded dictionary with the builder options
      @ Out, None
    """
    #instanciate message user
    super().__init__()
    # grep options
    self._options = kwargs
    # keep the module
    self.keepModule = self._options.pop("keepModule", False)
    # check working dir
    self.workingDir = self._options.pop("workingDir", None)
    if self.workingDir is None and self.keepModule:
      self.raiseAnError(IOError, "No workingDir has been provided for FMU exporter!")

    self.model = self._options.pop("model", None)
    if self.model is None:
      self.raiseAnError(IOError, "No model has been provided for FMU exporter!")
    # temp folder
    if self.keepModule:
      self._temp = self.workingDir
    else:
      self._temp = tempfile.TemporaryDirectory(prefix="raven_fmu_")
    # we serialize the model and this model will be stored in the resource folder
    self.serializedModel = Path(self._temp) / (self.model.name + ".pk")
    # picklefile in temp directory
    with open(self.serializedModel, mode="wb+") as pk:
      cloudpickle.dump(self.model, pk, protocol=pickle.HIGHEST_PROTOCOL)
    self.executeMethod = self._options.pop("executeMethod", None)
    if self.executeMethod is None:
      self.raiseAnError(IOError, "No executeMethod has been provided for FMU exporter!")
    self.inputVars = self.model._getVariableList('input')
    self.outVars =  self.model._getVariableList('output')
    if not self.inputVars or not self.outVars:
      self.raiseAnError(IOError, "Model {} is not exportable as FMU since no info about inputs/outputs are available!".format(self.model.name))
    self.indexReturnDict = self._options.pop("indexReturnDict", None)
    if self.indexReturnDict is None:
      self.raiseAMessage("No indexReturnDict has been provided for FMU exporter! Default to 0!")
      self.indexReturnDict = 0
    self.ravenDir = os.path.dirname(self._options.pop("frameworkDir"))
    if self.ravenDir is None:
      self.raiseAnError(IOError, "No ravenDir has been provided for FMU exporter!")

  def createModelHandler(self):
    """
      This create a temporary file that represents the model to export
      @ In, None
      @ Out, createModelHandler, str, the module for the FMU creation
    """
    className = self.model.name.upper()
    filename = self.model.name + ".pk"

    return f"""
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
import sys
import pickle
import os
from pythonfmu.fmi2slave import Fmi2Type, Fmi2Slave, Fmi2Causality, Fmi2Variability, Integer, Real, Boolean, String

class {className}(Fmi2Slave):
  #
  #  RAVEN (raven.inl.gov) Model-based Python-driven simulator
  #
  author = "RAVEN Team"
  description = "RAVEN Model-based Python-driven simulator"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.inputVariables = {self.inputVars}
    self.outputVariables = {self.outVars}
    # set path to raven and the serialized model
    self.raven_path = r"{self.ravenDir}"
    # model_path is by default the path to this model that is exported as FMU (serialized). It is stored in the resource folder
    self.model_path = self.resources + "/" + "{filename}"
    #os.path.sep
    sys.path.append(self.model_path)
    # this flag activates the initialization at the begin of the solve
    self.initialized = False
    # register raven_path variables if needed to be changed
    self.register_variable(String("raven_path", causality=Fmi2Causality.parameter, variability=Fmi2Variability.tunable))
    # register input variables
    for var in self.inputVariables:
      # set var
      self.__dict__[var] = 0.0
      self.register_variable(Real(var, causality=Fmi2Causality.input))
    for var in self.outputVariables:
      # set var
      self.__dict__[var] = 0.0
      self.register_variable(Real(var, causality=Fmi2Causality.output))

  def setup_experiment(self, start_time: float):
    self.start_time = start_time
    if not self.initialized:
      sys.path.append(self.raven_path)
      # find the RAVEN framework
      if os.path.isdir(os.path.join(self.raven_path,"ravenframework")):
        # we import the Driver to load the RAVEN enviroment for the un-pickling
        try:
          import ravenframework.Driver
        except RuntimeError as ae:
          # we try to add the framework directory
          raise RuntimeError("Importing or RAVEN failed with error:" +str(ae))
      else:
        print("framework not found in",self.raven_path)
      # de-serialize the model
      print("model_path", self.model_path)
      self.model = pickle.load(open(self.model_path, mode='rb'))
      self.initialized = True

  def do_step(self, current_time: float, step_size: float) -> bool:
    request = dict()
    for var in self.inputVariables:
      request[var] = self.__dict__[var]
    request['current_time'] = current_time
    request['step_size'] = step_size

    return_var = self.model.{self.executeMethod}(request)
    outs = return_var if isinstance(return_var,dict) else return_var[{self.indexReturnDict}]

    for var in outs:
       self.__dict__[var] = outs[var]
    return True

"""
  def buildFMU(self, dest: FilePath = ".") -> Path:
    """
      Build the FMU
      @ In, dest, FilePath, destination of the built fmu
      @ Out, built, Path, the path of the built fmu
    """
    if not os.path.exists(self.serializedModel):
      self.raiseAnError(ValueError, "No such file {}".format(self.serializedModel))

    self._options["dest"] = os.path.dirname(dest)
    self._options["file_name"] = os.path.basename(dest)
    #Get files needed for running model
    self._options["project_files"] = {self.serializedModel}.union(self.model.getSerializationFiles())

    scriptFile = Path(self._temp) / (self.model.name + "_RAVENfmu.py")
    with open(scriptFile, "+w") as f:
      f.write(self.createModelHandler())
    self._options["script_file"] = scriptFile

    built = FmuBuilder.build_FMU(**self._options)
    if not self.keepModule:
      self._temp.cleanup()
    return  built
