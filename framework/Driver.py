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
  # open the XML input
  if len(sys.argv) == 1:
    #NOTE: This can be overriden at the command line:
    # python Driver.py anotherFile.xml
    inputFile = os.path.join(workingDir,'test.xml')
  else:
    inputFile = sys.argv[1]
    if not os.path.isabs(inputFile): inputFile = os.path.join(workingDir,inputFile)

  #Parse the input
  #!!!!!!!!!!!!   Please do not put the parsing in a try statement... we need to make the parser able to print errors out 
  tree = ET.parse(inputFile)
  #except:  raise IOError('not possible to parse (xml based) the input file '+inputFile)
  if debug: print('opened file '+inputFile)
  root = tree.getroot()
  #generate all the components of the simulation
  simulation = Simulation(inputFile, frameworkDir,debug=debug)
  #Call the function to read and construct each single module of the simulation 
  simulation.XMLread(root)
  # Initialize the simulation 
  simulation.initialize()
  # Run the simulation 
  simulation.run()


