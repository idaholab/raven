'''
Created on Feb 20, 2013

@author: crisr
'''
#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3

#External Modules
import xml.etree.ElementTree as ET
import os
import sys
#External Modules

#Internal Modules
from Simulation import Simulation
#Internal Modules

#-------------------------------------------------------------Test Driver
debug = True

if __name__ == '__main__':
  '''This is the main driver for the RAVEN framework'''
  # Retrieve the framework directory path and working dir
  frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
  workingDir = os.getcwd()

  simulation = Simulation(frameworkDir,debug=debug)
  #If a configuration file exists, read it in
  configFile = os.path.join(os.path.expanduser("~"),".raven","default_runinfo.xml")
  if os.path.exists(configFile):
    tree = ET.parse(configFile)
    root = tree.getroot()
    if root.tag == 'Simulation' and [x.tag for x in root] == ["RunInfo"]:
      simulation.XMLread(root,runInfoSkip=set(["totNumCoresUsed"]))
    else:
      print('WARNING:',configFile,' should only have Simulation and inside it RunInfo')

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
    #except:  raise IOError('not possible to parse (xml based) the input file '+inputFile)
    if debug: print('opened file '+inputFile)
    root = tree.getroot()
    if root.tag != 'Simulation': raise IOError ('The outermost block of the input file '+inputFile+' it is not Simulation')
    #generate all the components of the simulation
  
    #Call the function to read and construct each single module of the simulation 
    simulation.XMLread(root,runInfoSkip=set(["DefaultInputFile"]))
  # Initialize the simulation 
  simulation.initialize()
  # Run the simulation 
  simulation.run()
  
  

  
  
  
  
  
  
  


