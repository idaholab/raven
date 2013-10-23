'''
Created on Feb 20, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import os
from Simulation import Simulation
import sys
#import matplotlib.pyplot as plt #FIXME

debug = False

if __name__ == '__main__':
  '''This is the main driver for the RAVEN framework'''
  # Retrieve the framework directory path and working dir
  frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
  workingDir = os.getcwd()
  # open the XML input
  if len(sys.argv) == 1:
    #inputFile = '/Users/alfoa/projects/trunk/raven/StochPoly_framework/SCtest.xml'
    inputFile = os.path.join(workingDir,'test_externalModule.xml')
  else:
    inputFile = sys.argv[1]
    if not os.path.isabs(inputFile): inputFile = os.path.join(workingDir,inputFile)

  #Parse the input
  #Please do not put the parsing in a try statement... we need to make the parser able to print errors out 
  tree = ET.parse(inputFile)
  #except:  raise IOError('not possible to parse (xml based) the input file '+inputFile)
  if debug: print('opened file '+inputFile)
  root = tree.getroot()
  #generate all the components of the simulation
  simulation = Simulation(inputFile, frameworkDir,debug=debug)
  #Call the function to read and construct each single module of the simulation 
  simulation.XMLread(root)
  # Run the simulation 
  simulation.run()


