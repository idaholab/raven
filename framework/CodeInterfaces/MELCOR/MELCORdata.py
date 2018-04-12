# Copyright 2017 University of Rome La Sapienza and Battelle Energy Alliance, LLC
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
  Created on April 18, 2017
  @author: Matteo Donorio (University of Rome La Sapienza),
           Fabio Gianneti (University of Rome La Sapienza),
           Andrea Alfonsi (INL)

  Modified on January 24, 2018
  @author: Violet Olson
           Thomas Riley (Oregon State University)
           Robert Shannon (Oregon State University)
  Change Summary: Added Control Function parsing
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
import re
import copy

class MELCORdata:
  """
    class that parses output of MELCOR 2.1 output file and reads in trip, minor block and write a csv file
    For now, Only the data associated to control volumes and control functions are parsed and output
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, FileObject, the file to parse
      @ Out, None
    """
    self.lines      = open(filen,"r").readlines()
    timeBlocks      = self.getTimeBlocks()
    self.timeParams = {}
    volForEachTime  = self.returnVolumeHybro(timeBlocks)
    self.timeParams.update(volForEachTime)
    self.functions  = self.returnControlFunctions(timeBlocks)

  def getTimeBlocks(self):
    """
     This method returns a dictionary of lists of type {"time":[lines Of Output for that time]}
     @ In, None
     @ Out, timeBlock, dict, {"time":[lines Of Output for that time]}
    """
    lineNum = []
    timeBlock = {}
    for lineNumber, line in enumerate(self.lines):
      if line.strip().startswith("1*"):
        lineNum.append([lineNumber,self.lines[lineNumber+1].split("=")[1].split( )[0]])
    for cnt,info in enumerate(lineNum):
      endLineCnt = lineNum[cnt+1][0]-1 if cnt < len(lineNum)-1 else len(self.lines)-1
      timeBlock[info[1]] = self.lines[info[0]+1:endLineCnt]
    return timeBlock

  def returnControlFunctions(self, timeBlock):
    """
      CONTROL FUNCTIONS EDIT
      @ In, timeBlock, dict, {"time":[lines Of Output for that time]}
      @ Out, functionValuesForEachTime, dict, {"time":{"functionName":"functionValue"}}
    """
    functionValuesForEachTime = {}
    timeOneRegex_name = re.compile("^\s*CONTROL\s+FUNCTION\s+(?P<name>[^\(]*)\s+(\(.*\))?\s*IS\s+.+\s+TYPE.*$")
    timeOneRegex_value = re.compile("^\s*VALUE\s+=\s+(?P<value>[^\s]*)")
    startRegex = re.compile("\s*CONTROL\s*FUNCTION\s*NUMBER\s*CURRENT\s*VALUE")
    regex = re.compile("^\s*(?P<name>( ?([0-9a-zA-Z-]+))*)\s+([0-9]+)\s*(?P<value>((([0-9.-]+)E(\+|-)[0-9][0-9])|((T|F))))\s*.*$")
    for time,listOfLines in timeBlock.items():
      functionValues = {}
      start = -1
      for lineNumber, line in enumerate(listOfLines):
        if re.search(startRegex, line):
          start = lineNumber + 1
          break
        elif re.search(timeOneRegex_name, line):
          start = -2
          break
      if start > 0:
        for lineNumber, line in enumerate(listOfLines[start:]):
          if line.startswith(" END OF EDIT FOR CF"):
            break
          match = re.match(regex, line)
          if match is not None:
            functionValues[match.groupdict()["name"]] = match.groupdict()["value"]
      elif start == -2:
        for lineNumber, line in enumerate(listOfLines):
          fcnName = re.match(timeOneRegex_name, line)
          if fcnName is not None:
            fcnValue = re.match(timeOneRegex_value, listOfLines[lineNumber+1])
            if fcnValue is not None:
              functionValues[fcnName.groupdict()["name"]] = fcnValue.groupdict()["value"]
      functionValuesForEachTime[time] = functionValues
    return functionValuesForEachTime

  def returnVolumeHybro(self,timeBlock):
    """
      CONTROL VOLUME HYDRODYNAMICS EDIT
      @ In, timeBlock, dict, {"time":[lines Of Output for that time]}
    """
    volForEachTime = {}
    for time,listOfLines in timeBlock.items():
      results = {}
      for cnt, line in enumerate(listOfLines):
        if line.strip().startswith("VOLUME"):
          headers  = line.strip().split()[1:len(line.strip().split())-1]
          for lineLine in listOfLines[cnt + 2:]:
            if len(lineLine.strip()) < 1:
              break
            valueSplit   = lineLine.strip().split()
            volumeNumber = lineLine.strip().split()[0]
            if not volumeNumber.isdigit():
              break
            valueSplit = valueSplit[1:len(valueSplit)]
            for paramCnt,header in enumerate(headers):
              parameter = "volume_"+str(volumeNumber)+"_"+header.strip()
              try:
                testFloat = float(valueSplit[paramCnt])
                results[parameter] = valueSplit[paramCnt]
              except ValueError:
                # in this way, the "strings" are not placed in the resulting csv
                pass
      volForEachTime[time] = copy.deepcopy(results)
    return volForEachTime

  def writeCsv(self,filen):
    """
      Output the parsed results into a CSV file
      @ In, filen, str, the file name of the CSV file
      @ Out, None
    """
    IOcsvfile=open(filen,'w+')
    getHeaders = self.timeParams.values()[0].keys()
    CFHeaders = self.functions.values()[0].keys()
    header = ','.join(getHeaders + CFHeaders)
    header = "time,"+header+"\n"
    IOcsvfile.write(header)
    for time in self.timeParams.keys():
      stringToWrite = str(time)
      for value in self.timeParams[time].values():
        stringToWrite+=","+str(value)
      for value in self.functions[time].values():
        stringToWrite+=","+str(value)
      stringToWrite+="\n"
      IOcsvfile.write(stringToWrite)
    IOcsvfile.close()
