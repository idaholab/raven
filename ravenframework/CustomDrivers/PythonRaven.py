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
  Class for using RAVEN as part of other Python workflows.
  Enables workflow loading, tampering, and running.
"""
import os
import sys

from . import DriverUtils

class Raven:
  """
    Class to enable running RAVEN as part of other Python workflows.
    Should provide utility functions to simplify the user's process.
  """
  framework = None

  # ********************
  # INITIALIZATION
  #
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    if self.framework is None:
      #Note: doSetup changes the sys.path, which can cause problems
      # if it is done at the start of this file which is done during import,
      # so it is only done now
      DriverUtils.doSetup()

      self.framework = DriverUtils.findFramework()


    self._simulation = None # RAVEN Simulation object (loaded workflow)
    self._xmlSource = None  # XML file from which workflow is loaded

  # ********************
  # API
  #
  def loadWorkflowFromFile(self, xmlFile):
    """
      Loads the target XML file as a workflow (simulation instance)
      @ In, xmlFile, string, target xml file to load (cwd?)
      @ Out, None
    """
    #Note: PythonRaven is imported as part of framework/__init__.py
    # so if Simulation is imported at the top, even an import of utils will pull
    # in almost all of Raven.
    from .. import Simulation
    from ..utils import TreeStructure as TS

    target = self._findFile(xmlFile)
    root = TS.parse(open(target, 'r')).getroot()
    targetDir = os.path.dirname(target)
    self._simulation = Simulation.Simulation(self.framework)
    self._simulation.XMLpreprocess(root, targetDir)
    self._simulation.XMLread(root, runInfoSkip={"DefaultInputFile"}, xmlFilename=target)
    self._simulation.initialize() # TODO separate method?

  def runWorkflow(self):
    """
      Runs the loaded workflow
      @ In, None
      @ Out, returnCode, int, value/error returned from RAVEN run
    """
    # FIXME reset the steps if necessary!
    returnCode = self._simulation.run()
    return returnCode

  def getEntity(self, kind, name):
    """
      Return an entity from RAVEN simulation
      @ In, kind, str, type of entity (e.g. DataObject, Sampler)
      @ In, name, str, identifier for entity (i.e. name of the entity)
      @ Out, entity, instance, RAVEN instance (None if not found)
    """
    # TODO is this the fastest way to get-and-check objects?
    kindGroup = self._simulation.entities.get(kind, None)
    if kindGroup is None:
      raise KeyError(f'Entity kind "{kind}" not recognized! Found: {list(self._simulation.entities.keys())}')
    entity = kindGroup.get(name, None)
    if entity is None:
      raise KeyError(f'No entity named "{name}" found among "{kind}" entities! Found: {list(self._simulation.entities[kind].keys())}')
    return entity


  # ********************
  # UTILITIES
  #
  def _findFile(self, wantFile):
    """
      Finds file on disk, trying a few options.
      @ In, wantFile, str, name and/or location of desired file
      @ Out, target, str, resolved absolute location of file (or None if not found)
    """
    target = None
    # option: user provided abs path
    fromAbs = os.path.abspath(wantFile)
    if os.path.isfile(fromAbs):
      target = fromAbs
    # option: user provided relative to current working directory
    if target is None:
      fromCWD = os.path.abspath(os.path.join(os.getcwd(), wantFile))
      if os.path.isfile(fromCWD):
        target = fromCWD
    # option: user provided relative to framework
    if target is None:
      fromFramework = os.path.abspath(os.path.join(self.framework, wantFile))
      if os.path.isfile(fromFramework):
        target = fromFramework
    # fail: no options yielded a valid file
    if target is None:
      msg = f'File "{wantFile}" was not found at any of the following absolute or relative locations:\n'
      msg += f'  absolute:  "{fromAbs}"\n'
      msg += f'  cwd:       "{fromCWD}"\n'
      msg += f'  framework: "{fromFramework}"\n'
      msg += 'Please check the path for the desired file.'
      raise IOError(msg)
    print(f'Target file found at "{target}"')
    return target
