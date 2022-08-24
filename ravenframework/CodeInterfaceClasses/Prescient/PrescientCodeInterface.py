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
Created on 2020-Sept-2

This is a CodeInterface for the Presient code.

"""

import os
import re
import warnings
try:
  import pkg_resources
  prescient = pkg_resources.get_distribution("prescient")
  prescientLocation = prescient.location
except Exception as inst:
  prescientLocation = None
  prescientException = inst

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase

class Prescient(CodeInterfaceBase):
  """
    This class is used to run the Prescient production cost modeling
    platform.
    https://github.com/grid-parity-exchange/Prescient
    It can perterb Prescient inputs and read the data in the
    bus_detail.csv and the hourly_summary.csv
  """

  def generateCommand(self, inputFiles, exe, clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code).  Ones of type 'PrescientRunnerInput' will be passed to the runner.py as command line arguments
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< fileargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is a list of commands to run the code (string), returnCommand[1] is the name of the output root
    """
    if prescientLocation is None:
      warnings.warn(f"Finding Prescient failed with {prescientException}")

    runnerInput = []
    for inp in inputFiles:
      if inp.getType() == 'PrescientRunnerInput':
        runnerInput.append(("parallel", "runner.py "+inp.getAbsFile()))

    return (runnerInput, os.path.join(inputFiles[0].getPath(), "output"))

  def createNewInput(self, inputs, oinputs, samplerType, **Kwargs):
    """
      This method is used to generate an input based on the information passed in.
      @ In, inputs, list,  list of current input files (input files from last this method call)
      @ In, oinputs, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    if prescientLocation is None:
      warnings.warn(f"Finding Prescient failed with {prescientException}")
    self._outputDirectory = None
    for singleInput in inputs:
      if singleInput.getType() == 'PrescientRunnerInput':
        #print("Need to modify", singleInput, "to fix", prescientLocation)
        newLines = []
        for line in open(singleInput.getAbsFile(),"r").readlines():
          if line.lstrip().startswith("--model-directory=") and prescientLocation is not None:
            items = line.split("=",1)[1].split("|")
            newPath = prescientLocation
            started = False
            for item in items:
              if item == "prescient":
                started = True
              if started:
                newPath = os.path.join(newPath, item)
            line = "--model-directory="+newPath
          elif line.lstrip().startswith("--output-directory="):
            self._outputDirectory = line.split("=",1)[1].rstrip()
          newLines.append(line)
        with open(singleInput.getAbsFile(),"w") as newFile:
          for line in newLines:
            newFile.write(line)
      elif singleInput.getType() == 'PrescientInput':
        #print("SampledVars", Kwargs["SampledVars"])
        #print("Modifying", singleInput)
        data = open(singleInput.getAbsFile(),"r").read()
        data = self.__processData(data, Kwargs["SampledVars"])
        open(singleInput.getAbsFile(),"w").write(data)
      else:
        raise IOError("Unknown Prescient input type: " + singleInput.getType() +
                      " for " + singleInput.getAbsFile())
    return inputs

  def __processData(self, data, samples):
    """
      Processes the input data and does some simple arithmetic
      This is used on the input files to allow more flexible perturbations
      This allow arithmetic like var*2.0+1.0
      @ In, data, string, the string to process
      @ In, samples, dict, the dictionary of variable values
      @ Out, retval, string, the string with values replaced
    """
    #"this is a $(a)$ and $(a+2)$ and $(a-2)$ and $(b_var)$ and $(a*-2.0)$"
    #"and $(a*2+1)$"
    splited = re.split("\\$\\(([a-z0-9._*+-]*)\\)\\$", data)
    retval = ""
    for i, value in enumerate(splited):
      if i % 2 == 1:
        name, mult, add = re.match("([a-z_][a-z0-9_]*)(\\*-?[0-9.]+)?([+-]?[0-9.]+)?", value).groups()
        num = samples[name]
        if mult is not None:
          num *= float(mult[1:])
        if add is not None:
          num += float(add)
        retval += str(num)
      else:
        retval += value
    return retval

  def _readBusData(self, filename):
    """
      Reads the electricity bus data into a dictionary
      @ In, filename, string, the bus_detail.csv file
      @ Out, (retDict,busList,datalist,hasMinute), (dictionary,list,list,bool),
        dictionary of each time, list of all the busses found,
        the data that each bus has, and if the bus has minute by minute data
    """
    with open(filename, "r") as inFile:
      first = True
      retDict = {}
      busSet = set()
      dataList = []
      for line in inFile.readlines():
        line = line.strip()
        if first:
          first = False
          if line.startswith("Date,Hour,Minute,Bus,"):
            hasMinute = True
          elif line.startswith("Date,Hour,Bus,"):
            #line sorta looks like: "Date,Hour,Bus,Shortfall,Overgeneration,LMP,LMP DA"
            hasMinute = False
          else:
            raise IOError("Unexpected first line of bus detail in Prescient Code Interface:" + line)
          dataList = [s.replace(" ","_") for s in line.split(",")[3:]]
          continue
        splited = line.split(",")
        if hasMinute:
          date, hour, minute, bus = splited[:4]
          rest = splited[4:]
          key = (date,hour,minute)
        else:
          date, hour, bus = splited[:3]
          rest = splited[3:]
          key = (date,hour)
        busSet.add(bus)
        timeDict = retDict.get(key,{})
        timeDict[bus] = rest
        retDict[key] = timeDict
      busList = list(busSet)
      busList.sort()
      return retDict, busList, dataList, hasMinute

  def finalizeCodeOutput(self, command, codeLogFile, subDirectory):
    """
      Convert csv information to RAVEN's prefered formats
      Joins together two different csv files and also reorders it a bit.
      @ In, command, ignored
      @ In, codeLogFile, ignored
      @ In, subDirectory, string, the subdirectory where the information is.
      @ Out, directory, string, the base name of the csv file
    """
    #print("finalizeCodeOutput", command, codeLogFile, subDirectory)
    toRead = "hourly_summary" #"Daily_summary"
    if self._outputDirectory is not None:
      directory = os.path.join(subDirectory, self._outputDirectory)
    else:
      directory = subDirectory
    readFile = os.path.join(directory, toRead)
    if toRead.lower().startswith("hourly"):
      busData, busList, busDataList, hasMinute = self._readBusData(os.path.join(directory, "bus_detail.csv"))
      outDict = {}
      inFile = open(readFile+".csv","r")
      hasNetDemand = False
      firstLine = inFile.readline()
      date, hour, rest = firstLine.split(",", maxsplit=2)
      restSplit = rest.rstrip().split(",")
      hourKey = hour.strip()
      timeKey = date.rstrip()+"_"+hourKey
      outDict[timeKey] = []
      outDict[hour] = []
      otherKeys = restSplit
      for key in otherKeys:
        outDict[key] = []
      if "RenewablesUsed" in outDict and "Demand" in outDict:
        hasNetDemand = True
        outDict["NetDemand"] = []
      first = False
      for bus in busList:
        for dataName in busDataList:
          outDict[bus+"_"+dataName] = []

      for line in inFile.readlines():
        date, hour, rest = line.split(",", maxsplit=2)
        restSplit = rest.rstrip().split(",")
        timeValue = date.rstrip()+"_"+hour.lstrip()
        outDict[timeKey].append(timeValue)
        outDict[hourKey].append(float(hour))
        for key,value in zip(otherKeys,restSplit):
          outDict[key].append(float(value))
        if hasNetDemand:
          #Calculate the demand - renewables used to get net demand
          netDemand = outDict["Demand"][-1]  - outDict["RenewablesUsed"][-1]
          outDict["NetDemand"].append(netDemand)
        for bus in busList:
          if hasMinute:
            busTimeKey = (date, hour, '0')
          else:
            busTimeKey = (date, hour)
          for dataName, data in zip(busDataList,busData[busTimeKey][bus]):
            outDict[bus+"_"+dataName].append(float(data) if len(data) > 0 else float("NaN"))
    return outDict

  def addDefaultExtension(self):
    """
      Possible input extensions found in the input files.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['txt', 'dat'])

