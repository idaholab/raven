#!/usr/bin/env python
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
Created on Feb 20, 2013

@author: crisr

This is the Driver of RAVEN
"""

from __future__ import division, print_function, unicode_literals, absolute_import
# if in debug mode, activate deprication warnings
## TODO does this need to be done in all modules, or just this one?
import warnings

if not __debug__:
  warnings.filterwarnings("ignore")
else:
  warnings.simplefilter("default", DeprecationWarning)

import os
import sys
import time
import threading
import traceback
import xml.etree.ElementTree as ET

#warning: this needs to be before importing h5py
os.environ["MV2_ENABLE_AFFINITY"]="0"

frameworkDir = os.path.dirname(os.path.abspath(__file__))

# library handler is in scripts
sys.path.append(os.path.join(frameworkDir, '..', "scripts"))
import library_handler as LH
sys.path.pop() #remove scripts path for cleanliness

from utils import utils
import utils.TreeStructure as TS
utils.find_crow(frameworkDir)

if sys.version_info.major == 2:
  utils.add_path_recursively(os.path.join(frameworkDir,'contrib','pp'))
else:
  utils.add_path_recursively(os.path.join(frameworkDir,'contrib','pp3'))
utils.add_path(os.path.join(frameworkDir,'contrib','AMSC'))
utils.add_path(os.path.join(frameworkDir,'contrib'))
#Internal Modules
from Simulation import Simulation
from Application import __QtAvailable
from Interaction import Interaction
#Internal Modules


#------------------------------------------------------------- Driver
def printStatement():
  """
    Method to print the BEA header
    @ In, None
    @ Out, None
  """
  print("""
Copyright 2017 Battelle Energy Alliance, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
  """)

def printLogo():
  """
    Method to print a RAVEN logo
    @ In, None
    @ Out, None
  """
  print("""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      .---.        .------######       #####     ###   ###  ########  ###    ###
     /     \  __  /    --###  ###    ###  ###   ###   ###  ###       #####  ###
    / /     \(  )/    --###  ###    ###   ###  ###   ###  ######    ### ######
   //////   ' \/ `   --#######     #########  ###   ###  ###       ###  #####
  //// / // :    :   -###   ###   ###   ###    ######   ####      ###   ####
 // /   /  /`    '---###    ###  ###   ###      ###    ########  ###    ###
//          //..\\
===========UU====UU=============================================================
           '//||\\`
             ''``
    """)

def checkVersions():
  """
    Method to check if versions of modules are new enough. Will call sys.exit
    if they are not in the range specified.
    @ In, None
    @ Out, None
  """
  # if libraries are not to be checked, we're done here
  if not LH.checkVersions():
    return
  # otherwise, we check for incorrect libraries
  missing, notQA = LH.checkLibraries()
  if missing:
    print('ERROR: Some required Python libraries are missing but required to run RAVEN as configured:')
    for lib, version in missing:
      # report the missing library
      msg = '  -> MISSING: {}'.format(lib)
      # add the required version if applicable
      if version is not None:
        msg += ' version {}'.format(version)
      print(msg)
  if notQA:
    print('ERROR: Some required Python libraries have incorrect versions for running RAVEN as configured:')
    for lib, found, need in notQA:
      print('  -> WRONG VERSION: lib "{}" need "{}" but found "{}"'.format(lib, found, need))
  if missing or notQA:
    print('Try installing libraries using instructions on RAVEN repository wiki at ' +
           'https://github.com/idaholab/raven/wiki/Installing_RAVEN_Libraries.')
    sys.exit(-4)
  else:
    print('RAVEN Python dependencies located and checked.')
  # TODO give a warning for missing libs even if skipping check?
  # -> this is slow, so maybe not.


if __name__ == '__main__':
  """This is the main driver for the RAVEN framework"""
  # Retrieve the framework directory path and working dir

  printLogo()
  printStatement()

  checkVersions()

  verbosity      = 'all'
  interfaceCheck = False
  interactive = Interaction.No
  workingDir = os.getcwd()

  ## Remove duplicate command line options and preserve order so if they try
  ## conflicting options, the last one will take precedence.
  sys.argv = utils.removeDuplicates(sys.argv)

  itemsToRemove = []
  for item in sys.argv:
    # I don't think these do anything.  - talbpaul, 2017-10
    if item.lower() in ['silent','quiet','all']:
      verbosity = item.lower()
      itemsToRemove.append(item)
    elif item.lower() == 'interfacecheck':
      interfaceCheck = True
      itemsToRemove.append(item)
    elif item.lower() == 'interactive':
      if __QtAvailable:
        interactive = Interaction.Yes
      else:
        print('Qt is not available, disabling interactive mode.\n')
      itemsToRemove.append(item)
    elif item.lower() == 'interactivecheck':
      if __QtAvailable:
        interactive = Interaction.Test
      else:
        print('Qt is not available, disabling interactive check.\n')
      itemsToRemove.append(item)

  ## Now outside of the loop iterating on the object we want to modify, we are
  ## safe to remove each of the items
  for item in itemsToRemove:
    sys.argv.remove(item)

  if interfaceCheck:
    os.environ['RAVENinterfaceCheck'] = 'True'
    print('Interface CHECK activated!\n')
  else:
    os.environ['RAVENinterfaceCheck'] = 'False'

  simulation = Simulation(frameworkDir, verbosity=verbosity, interactive=interactive)

  #If a configuration file exists, read it in
  configFile = os.path.join(os.path.expanduser("~"),".raven","default_runinfo.xml")
  if os.path.exists(configFile):
    tree = ET.parse(configFile)
    root = tree.getroot()

    if root.tag == 'Simulation' and [x.tag for x in root] == ["RunInfo"]:
      simulation.XMLread(root,runInfoSkip=set(["totNumCoresUsed"]),xmlFilename=configFile)
    else:
      e=IOError('DRIVER',str(configFile)+' should only have Simulation and inside it RunInfo')
      print('\nERROR! In Driver,',e,'\n')
      sys.exit(1)

  # Find the XML input file
  if len(sys.argv) == 1:
    #NOTE: This can be overriden at the command line:
    # python Driver.py anotherFile.xml
    # or in the configuration file by DefaultInputFile
    inputFiles = [simulation.getDefaultInputFile()]
  else:
    inputFiles = sys.argv[1:]

  for i in range(len(inputFiles)):
    if not os.path.isabs(inputFiles[i]):
      inputFiles[i] = os.path.join(workingDir,inputFiles[i])

  simulation.setInputFiles(inputFiles)
  #Parse the input
  #For future developers of this block, assure that useful, informative exceptions
  #  are still thrown while parsing the XML tree.  Otherwise any error made by
  #  the developer or user might be obfuscated.
  for inputFile in inputFiles:
    try:
      tree = TS.parse(open(inputFile,'r'))
    except TS.InputParsingError as e:
      print('\nInput Parsing error!',e,'\n')
      sys.exit(1)

    #except?  riseanIOError('not possible to parse (xml based) the input file '+inputFile)
    if verbosity=='debug':
      print('DRIVER','opened file '+inputFile)

    root = tree.getroot()
    if root.tag != 'Simulation':
      e=IOError('The outermost block of the input file '+inputFile+' it is not Simulation')
      print('\nInput XML Error!',e,'\n')
      sys.exit(1)

    # call the function to load the external xml files into the input tree
    cwd = os.path.dirname(os.path.abspath(inputFile))
    simulation.XMLpreprocess(root,cwd)
    #generate all the components of the simulation
    #Call the function to read and construct each single module of the simulation
    simulation.XMLread(root,runInfoSkip=set(["DefaultInputFile"]),xmlFilename=inputFile)

  def raven():
    """
      A worker function that allows the computation of the main RAVEN execution
      to be offloaded to another thread, freeing the main thread for UI
      interaction (Qt requires UI to be handled on the main thread of execution)
    """
    simulation.initialize()
    simulation.run()

    ## If there is an associated UI application, then we can quit it now that
    ## we are done, the main thread does not know when this done presumably
    ## because this thread still is technically running as long as the app,
    ## which both threads can see, has not called quit. Otherwise, we could do
    ## this after the while loop below.
    if simulation.app is not None:
      simulation.app.quit()

  if simulation.app is not None:
    try:
      ## Create the thread that will run RAVEN, and make sure that it will die if
      ## the main thread dies by making it a daemon, then start it up
      ravenThread = threading.Thread(target=raven)
      ravenThread.daemon = True
      ravenThread.start()

      ## If there is an associated application, then we can start it up now as
      ## well. It will listen for UI update requests from the ravenThread.
      if simulation.app is not None:
        simulation.app.exec_()

      ## This makes sure that the main thread waits for RAVEN to complete before
      ## exiting, however join will block the main thread until ravenThread is
      ## complete, thus ignoring any kill signals until after it has completed
      # ravenThread.join()

      waitTime = 0.1 ## in seconds

      ## So, in order to live wait for ravenThread, we need a spinlock that will
      ## allow us to accept keyboard input.
      while ravenThread.isAlive():
        ## Use one of these two alternatives, effectively they should be the same
        ## not sure if there is any advantage to one over the other
        time.sleep(waitTime)
        # ravenThread.join(waitTime)

    except KeyboardInterrupt:
      if ravenThread.isAlive():
        traceback.print_stack(sys._current_frames()[ravenThread.ident])
      print ('\n\n! Received keyboard interrupt, exiting RAVEN.\n\n')
    except SystemExit:
      if ravenThread.isAlive():
        traceback.print_stack(sys._current_frames()[ravenThread.ident])
      print ('\n\n! Exit called, exiting RAVEN.\n\n')
  else:
    raven()
