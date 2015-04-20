'''
Created on Feb 20, 2013

@author: crisr
'''
#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3

#External Modules--------------------begin
import xml.etree.ElementTree as ET
import os
import sys
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
#Internal Modules

#------------------------------------------------------------- Driver
def printStatement():
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

if __name__ == '__main__':
  '''This is the main driver for the RAVEN framework'''
  # Retrieve the framework directory path and working dir
  printStatement()
  verbosity      = 'all'
  interfaceCheck = False
  workingDir = os.getcwd()
  for item in sys.argv:
    if   item.lower() == 'silent':
      verbosity = 'silent'
      sys.argv.pop(sys.argv.index(item))
    elif item.lower() == 'quiet':
      verbosity = 'quiet'
      sys.argv.pop(sys.argv.index(item))
    elif item.lower() == 'all':
      verbosity = 'all'
      sys.argv.pop(sys.argv.index(item))
    elif item.lower() == 'debug':
      debug = True
      sys.argv.pop(sys.argv.index(item))
    elif item.lower() == 'interfacecheck':
      interfaceCheck = True
      sys.argv.pop(sys.argv.index(item))
  if interfaceCheck: os.environ['RAVENinterfaceCheck'] = 'True'
  else             : os.environ['RAVENinterfaceCheck'] = 'False'
  simulation = Simulation(frameworkDir,verbosity=verbosity)
  #If a configuration file exists, read it in
  configFile = os.path.join(os.path.expanduser("~"),".raven","default_runinfo.xml")
  if os.path.exists(configFile):
    tree = ET.parse(configFile)
    root = tree.getroot()
    if root.tag == 'Simulation' and [x.tag for x in root] == ["RunInfo"]:
      simulation.XMLread(root,runInfoSkip=set(["totNumCoresUsed"]),xmlFilename=configFile)
    else:
      utils.raiseAWarning('DRIVER',str(configFile)+' should only have Simulation and inside it RunInfo')

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
  #!!!!!!!!!!!!   Please do not put the parsing in a try statement... we need to make the parser able to print errors out
  for inputFile in inputFiles:
    tree = ET.parse(inputFile)
    #except?  raisea IOError('not possible to parse (xml based) the input file '+inputFile)
    if debug: utils.raiseAMessage('DRIVER','opened file '+inputFile)
    root = tree.getroot()
    if root.tag != 'Simulation': utils.raiseAnError(IOError,'DRIVER','The outermost block of the input file '+inputFile+' it is not Simulation')
    #generate all the components of the simulation

    #Call the function to read and construct each single module of the simulation
    simulation.XMLread(root,runInfoSkip=set(["DefaultInputFile"]),xmlFilename=inputFile)
  # Initialize the simulation
  simulation.initialize()
  # Run the simulation
  simulation.run()

