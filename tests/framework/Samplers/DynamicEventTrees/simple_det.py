#!/usr/bin/env python
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
Program that reads in input file, and runs it for a bit.
If start_time + timesteps * time_delta < end_time, it will
write out an xml file with branch info, otherwise it will stop.

"""


#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

import sys,os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

print("Arguments:",sys.argv)

if len(sys.argv) < 3 or sys.argv[1] != "-i":
  print(sys.argv[0]," -i input")
  sys.exit(-1)


inputFilename = sys.argv[2]
inputFile = open(inputFilename,"r")
root = os.path.splitext(inputFilename)[0]
head,tail = os.path.split(root)
prefix = "out~"
outFile = open(os.path.join(head,prefix+tail+".csv"),"w")

lines = inputFile.readlines()
lastLine = lines.index("[]\n")

def get_params(line):
  start_match = re.search("\[\./([a-zA-Z_]+)\]",line)
  if start_match is not None:
    return "[./]",start_match.group(1)
  if " = " not in line:
    return None, None
  equalsIndex = line.index("=")
  name = line[:equalsIndex].strip()
  if line.rstrip().endswith('"'):
    params = line[equalsIndex + 1:].strip().strip('"').split()
    return name,params
  else:
    return name,line[equalsIndex + 1:].strip()

triggerLow = []
triggerHigh = []

for name, params in [get_params(l) for l in lines]:
  print(name,params)
  if name == "names":
    nameList = params
    names = ",".join(["time","x"]+params)
    nameIndexs = {}
    for i in range(len(params)):
      nameIndexs[params[i]] = i
  elif name == "start":
    start = list(map(float, params))
  elif name == "dt":
    dt = list(map(float, params))
  elif name == "dx":
    dx = list(map(float, params))
  elif name == "start_time":
    startTime = float(params)
  elif name == "end_time":
    endTime = float(params)
  elif name == "time_delta":
    timeDelta = float(params)
  elif name == "x_start":
    xStart = float(params)
  elif name == "timesteps":
    timesteps = int(params)
  elif name == "x_dt":
    xDt = float(params)
  elif name == "end_dx":
    endDx = list(map(float, params))
  elif name == "end_probability":
    endProbability = list(map(float, params))
  elif name == "trigger_name":
    triggerName = params.strip()
  elif name == "trigger_low":
    triggerLow = list(map(float, params))
  elif name == "trigger_high":
    triggerHigh = list(map(float, params))
  elif name == "[./]":
    header = params
  elif name == "value":
    temp_value = float(params)
    if header == "x":
      xStart = temp_value - startTime * xDt
    elif header in nameIndexs:
      j = nameIndexs[header]
      start[j] = temp_value - startTime * dt[j]

print(names,file=outFile)

#timeDelta = (endTime - startTime)/timesteps

if len(triggerLow) > 0 and len(triggerHigh) > 0:
  hasTrigger = True
else:
  hasTrigger = False

triggerVariable = ""
triggerValue = 0.0

for i in range(0,timesteps+1):
  shouldBreak = False
  time = startTime + timeDelta * i
  x = xStart + xDt * time
  print(time, end=",", file=outFile)
  print(x, end="", file=outFile)
  for j in range(0,min(len(start),len(dx),len(dt))):
    print(end=",", file=outFile)
    value = start[j] + time*dt[j] + x*dx[j]
    print(value, end="", file=outFile)
    if hasTrigger:
      if triggerLow[j] < value < triggerHigh[j]:
        shouldBreak = True
        triggerVariable = nameList[j]
        triggerValue = value
  print(file=outFile)
  if shouldBreak:
    break

#print([x + endDxi for endDxi in endDx],endProbability)

outFile.close()

if len(triggerVariable) == 0:
  triggerVariable = "x"
  triggerValue = x

if time < endTime:
  root = ET.Element('Branch_info')
  root.set("end_time", str(time))
  triggerNode=ET.SubElement(root,"Distribution_trigger")
  triggerNode.set("name", triggerName)
  var = ET.SubElement(triggerNode,'Variable')
  var.text=triggerVariable
  var.set('type', "auxiliary")
  var.set('old_value', str(triggerValue))
  var.set('actual_value', " ".join([str(v) for v in [triggerValue + endDxi for endDxi in endDx]]))
  var.set('probability', " ".join([str(p) for p in endProbability]))

  xmlFile = open(os.path.join(head,prefix+tail+"_actual_branch_info.xml"),"w")
  xmlFile.write(minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="\t"))
  xmlFile.close()


