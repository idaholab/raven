#!/usr/bin/env python
"""
Created on Feb 20, 2013

@author: crisr

This is the Driver of RAVEN
"""
#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3

#External Modules--------------------begin
import xml.etree.ElementTree as ET
import os
import sys
import threading
import time
import traceback
#External Modules--------------------end

#warning: this needs to be before importing h5py
os.environ["MV2_ENABLE_AFFINITY"]="0"

frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(frameworkDir,'utils'))
import utils
utils.find_crow(frameworkDir)
utils.add_path_recursively(os.path.join(frameworkDir,'contrib'))
#Internal Modules
from Simulation import Simulation
from Application import __PySideAvailable
#Internal Modules

#------------------------------------------------------------- Driver
def printStatement():
  """
    Method to print the BEA header
    @ In, None
    @ Out, None
  """
  print("""
  NOTICE: This computer software was prepared by Battelle Energy
  Alliance, LLC, hereinafter the Contractor, under Contract
  No. DE-AC07-05ID14517 with the United States (U.S.)  Department of
  Energy (DOE). All rights in the computer software are reserved by
  DOE on behalf of the United States Government and, if applicable,
  the Contractor as provided in the Contract. You are authorized to
  use this computer software for Governmental purposes but it is not
  to be released or distributed to the public. NEITHER THE UNITED
  STATES GOVERNMENT, NOR DOE, NOR THE CONTRACTOR MAKE ANY WARRANTY,
  EXPRESSED OR IMPLIED, OR ASSUMES ANY LIABILITY OR RESPONSIBILITY FOR
  THE USE, ACCURACY, COMPLETENESS, OR USEFULNESS OR ANY INFORMATION,
  APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE
  WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS. This notice, including
  this sentence, must appear on any copies of this computer software.
  """)

def checkVersions():
  """
    Method to check if versions of modules are new enough. Will call sys.exit
    if they are not in the range specified.
    @ In, None
    @ Out, None
  """
  sys.path.append(os.path.join(os.path.dirname(frameworkDir),"scripts","TestHarness","testers"))
  import RavenUtils
  sys.path.pop() #remove testers path
  missing,outOfRange,notQA = RavenUtils.checkForMissingModules()
  if len(missing) + len(outOfRange) > 0 and RavenUtils.checkVersions():
    print("ERROR: too old, too new, or missing raven libraries, not running:")
    for error in missing + outOfRange + notQA:
      print(error)
    sys.exit(-4)
  else:
    if len(missing) + len(outOfRange) > 0:
      print("WARNING: not using tested versions of the libraries:")
      for warning in notQA + missing + outOfRange:
        print(warning)

if __name__ == '__main__':
  """This is the main driver for the RAVEN framework"""
  # Retrieve the framework directory path and working dir
  printStatement()

  checkVersions()

  verbosity      = 'all'
  interfaceCheck = False
  interactive = False
  workingDir = os.getcwd()

  ## Remove duplicate command line options and preserve order so if they try
  ## conflicting options, the last one will take precedence.
  sys.argv = utils.removeDuplicates(sys.argv)

  itemsToRemove = []
  for item in sys.argv:
    if item.lower() in ['silent','quiet','all']:
      verbosity = item.lower()
      itemsToRemove.append(item)
    elif item.lower() == 'interfacecheck':
      interfaceCheck = True
      itemsToRemove.append(item)
    elif item.lower() == 'interactive':
      if __PySideAvailable:
        interactive = True
      else:
        print('\nPySide is not installed, disabling interactive mode.\n')
      itemsToRemove.append(item)

  ## Now outside of the loop iterating on the object we want to modify, we are
  ## safe to remove each of the items
  for item in itemsToRemove:
    sys.argv.remove(item)

  if interfaceCheck:
    os.environ['RAVENinterfaceCheck'] = 'True'
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
      tree = ET.parse(inputFile)
    except ET.ParseError as e:
      print('\nXML Parsing error!',e,'\n')
      sys.exit(1)

    #except?  riseanIOError('not possible to parse (xml based) the input file '+inputFile)
    if verbosity=='debug':
      print('DRIVER','opened file '+inputFile)

    root = tree.getroot()
    if root.tag != 'Simulation':
      e=IOError('The outermost block of the input file '+inputFile+' it is not Simulation')
      print('\nInput XML Error!',e,'\n')
      sys.exit(1)

    # call the function to load the external xml files into the ET
    simulation.XMLpreprocess(root,xmlFileName=inputFile)
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
