'''
Created on Feb 20, 2013

@author: crisr
'''
import sys
import xml.etree.ElementTree as ET
from Simulation import Simulation


if __name__ == '__main__':
    #open the XML
    try:
      inputFile = sys.argv[1]
    except:
      raise IOError ('incorrect input file provided ' + sys.argv[1])
    try:
      tree = ET.parse(inputFile)
    except:
      raise IOError ('not able to parse' + inputFile)
    #generate all the components of the simulation
    simulation = Simulation()
    root = tree.getroot()
    for simType in root:
      simulation.add(simType)
      
      
      