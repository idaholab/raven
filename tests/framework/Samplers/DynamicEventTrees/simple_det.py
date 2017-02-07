#!/usr/bin/env python
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
  if " = " not in line:
    return None, None
  equalsIndex = line.index("=")
  name = line[:equalsIndex].strip()
  if line.rstrip().endswith('"'):
    params = line[equalsIndex + 1:].strip().strip('"').split()
    return name,params
  else:
    return name,line[equalsIndex + 1:].strip()

for name, params in [get_params(l) for l in lines]:
  print(name,params)
  if name == "names":
    names = ",".join(["time","x"]+params)
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
  elif name == "value":
    temp_value = float(params)
    xStart = temp_value - startTime * xDt

print(names,file=outFile)

#timeDelta = (endTime - startTime)/timesteps

for i in range(0,timesteps+1):
  time = startTime + timeDelta * i
  x = xStart + xDt * time
  print(time, end=",", file=outFile)
  print(x, end="", file=outFile)
  for j in range(0,min(len(start),len(dx),len(dt))):
    print(end=",", file=outFile)
    print(start[j] + time*dt[j] + x*dx[j], end="", file=outFile)
  print(file=outFile)

#print([x + endDxi for endDxi in endDx],endProbability)

outFile.close()

if time < endTime:
  root = ET.Element('Branch_info')
  root.set("end_time", str(time))
  triggerNode=ET.SubElement(root,"Distribution_trigger")
  triggerNode.set("name", triggerName)
  var = ET.SubElement(triggerNode,'Variable')
  var.text="x"
  var.set('type', "auxiliary")
  var.set('old_value', str(x))
  var.set('actual_value', " ".join([str(v) for v in [x + endDxi for endDxi in endDx]]))
  var.set('probability', " ".join([str(p) for p in endProbability]))

  xmlFile = open(os.path.join(head,prefix+tail+"_actual_branch_info.xml"),"w")
  xmlFile.write(minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="\t"))
  xmlFile.close()


