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
Created on Jul 26, 2013

@author: andrea
"""
import xml.etree.ElementTree as ET
import sys
import time
from BatemanClass import *

def readInputXML(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    input = {}
    for element in root:
      if element.tag != "nuclides":
        input[element.tag] = [float(elm) for elm in element.text.split()]
        if element.tag == "totalTime": input[element.tag] = input[element.tag][0]
      else:
        input[element.tag] = {}
        for child in element:
          input[element.tag][child.tag] ={}
          for childChild in child:
            try   : input[element.tag][child.tag][childChild.tag] = float(childChild.text)
            except: input[element.tag][child.tag][childChild.tag] = childChild.text
    return input

if __name__ == '__main__':
    startTime = time.time()
    if len(sys.argv) == 1: inputFileName = "Input.xml"
    else                 : inputFileName = sys.argv[1]
    if len(sys.argv) < 3 : outputFileName = "results.csv"
    else                 : outputFileName = sys.argv[2]
    logFile = open(outputFileName.split(".")[0]+".out", 'w')
    headerLicense = "Copyright 2017 Battelle Energy Alliance, LLC \n "
    headerLicense+= "\n"
    headerLicense+= "Licensed under the Apache License, Version 2.0 (the 'License'); \n"
    headerLicense+= "you may not use this file except in compliance with the License. \n"
    headerLicense+= "You may obtain a copy of the License at \n"
    headerLicense+= "\n"
    headerLicense+= "http://www.apache.org/licenses/LICENSE-2.0 \n"
    headerLicense+= "\n"
    headerLicense+= "Unless required by applicable law or agreed to in writing, software \n"
    headerLicense+= "distributed under the License is distributed on an 'AS IS' BASIS, \n"
    headerLicense+= "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    headerLicense+= "See the License for the specific language governing permissions and\n"
    headerLicense+= "limitations under the License.\n"
    headerLicense+= "\n"
    headerLicense+= "-------------------------------------------------------------------------\n"
    headerLicense+= "\n"
    headerLicense+= "Input is   : "+inputFileName+ "\n"
    headerLicense+= "Outputs are: "+outputFileName+ " | "+outputFileName.split(".")[0]+".out"+"\n"
    logFile.write(headerLicense)
    logFile.write("\n")
    logFile.write("-------------------------------------------------------------------------\n")
    print(headerLicense)
    print("-------------------------------------------------------------------------\n")
    print("Reading input")
    print("-------------------------------------------------------------------------\n")
    initializationDict = readInputXML(inputFileName)
    test = BatemanClass(initializationDict)
    print("Running code")
    print("-------------------------------------------------------------------------\n")
    test.runDpl()
    if outputFileName.endswith(".csv"): outputFileName = outputFileName
    else                              : outputFileName = outputFileName + ".csv"
    print("Writing results")
    print("-------------------------------------------------------------------------\n")
    outputFileCsv = open(outputFileName, 'w')
    test.printResults(outputFileCsv)
    outputFileCsv.close()
    logFile.write("RESULTS:\n")
    test.printResults(logFile," ")
    totalTime = time.time() - startTime
    logFile.write("CPU TIME: " +'{:6s}'.format(str(totalTime))+" s")
    logFile.write("-------------------------------------------------------------------------\n")
    logFile.write("SUCCESS")
    logFile.close()
    print("End calculation")
    print("\nCPU TIME: " +'{:6s}'.format(str(totalTime))+" s")
    print("-------------------------------------------------------------------------\n")



