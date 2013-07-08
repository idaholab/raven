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
import matplotlib.pyplot as plt
import subprocess

debug = True

if __name__ == '__main__':
  script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  #open the XML
  try:
    if len(sys.argv) == 1:
      inputFile = 'test.xml' 
    else:
      inputFile = sys.argv[1]
  except:
    raise IOError ('input file not provided')
  workingDir = os.getcwd()
  if not os.path.isabs(inputFile):
    inputFile = os.path.join(workingDir,inputFile)
  if not os.path.exists(inputFile):
    print('file not found '+inputFile)
  try:
    tree = ET.parse(inputFile)
    if debug: print('opened file '+inputFile)
  except:
    raise IOError ('not able to parse ' + inputFile)
  root = tree.getroot()
  #generate all the components of the simulation
  simulation = Simulation(inputFile, script_dir)
  simulation.XMLread(root)
  in_pbs = "PBS_NODEFILE" in os.environ
  if simulation.runInfoDict['mode'] == 'pbs' and not in_pbs:
    batchSize = simulation.runInfoDict['batchSize']
    command = ["qsub","-l","select="+str(batchSize)+":ncpus=1",
               "-l","walltime="+simulation.runInfoDict["expectedTime"],
               "-l","place=free","-v",
               'COMMAND="python Driver.py '+inputFile+'"',
               "./save_and_command2.sh"]
    os.chdir(workingDir)
    print(os.getcwd(),command)
    subprocess.call(command)
  else:
    simulation.run()
  
  plt.show()

