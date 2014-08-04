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
from utils import returnPrintTag
frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
ravenDir = os.path.dirname(frameworkDir)

#Add the module directory to the search path.
pmoduleDir = os.path.join(os.path.dirname(ravenDir),"crow","python_modules")
sys.path.append(pmoduleDir)
if not os.path.exists(pmoduleDir): raise IOError('The directory "python_modules" has not been found. It location is supposed to be ../python_modules')

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

os.environ["MV2_ENABLE_AFFINITY"]="0"

if __name__ == '__main__':
  '''This is the main driver for the RAVEN framework'''
  # Retrieve the framework directory path and working dir
  printStatement()

  workingDir = os.getcwd()
  if 'debug=True' in sys.argv: debug=True
  else                       : debug=False

  simulation = Simulation(frameworkDir,debug=debug)
  #If a configuration file exists, read it in
  configFile = os.path.join(os.path.expanduser("~"),".raven","default_runinfo.xml")
  if os.path.exists(configFile):
    tree = ET.parse(configFile)
    root = tree.getroot()
    if root.tag == 'Simulation' and [x.tag for x in root] == ["RunInfo"]:
      simulation.XMLread(root,runInfoSkip=set(["totNumCoresUsed"]))
    else:
      print(returnPrintTag('DRIVER') + ' -> WARNING:',configFile,' should only have Simulation and inside it RunInfo')

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
    #except?  raise IOError('not possible to parse (xml based) the input file '+inputFile)
    if debug: print('opened file '+inputFile)
    root = tree.getroot()
    if root.tag != 'Simulation': raise IOError (returnPrintTag('DRIVER') +' The outermost block of the input file '+inputFile+' it is not Simulation')
    #generate all the components of the simulation

    #Call the function to read and construct each single module of the simulation
    simulation.XMLread(root,runInfoSkip=set(["DefaultInputFile"]))
  # Initialize the simulation
  simulation.initialize()
  # Run the simulation
  simulation.run()

