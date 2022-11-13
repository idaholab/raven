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
Created on August 31, 2020

@author: Andrea Alfonsi, Congjian Wang

comments: Interface for AccelerateCFD
"""
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import glob
import math
from sklearn import neighbors
from .OpenFoamPP import fieldParser
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..Generic.GenericCodeInterface import GenericParser
from ravenframework.utils import mathUtils

class AcceleratedCFD(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to AcceleratedCFD
  """
  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    self.locations = {}
    for child in xmlNode:
      if child.tag == 'outputLocations':
        coords = child.find("coordinates")
        if coords is None:
          raise IOError("Sub-node <coordinates> not found in node <outputLocations>!")
        coordinates = coords.text.split()
        cds = []
        for coord in coordinates:
          if not coord.startswith("(") or not coord.endswith(")"):
            raise IOError("<coordinates> must be inputted with the following format (x1,y1,z1) ")
          xyz = [c.strip() for c in coord.replace(")","").replace("(","").split(",")]
          if len(xyz) != 3:
            raise IOError("<coordinates> must be inputted with the following format (x1,y1,z1) (x2,y2,z2) etc ")
          cds.append(xyz)
        self.locations['inputCoords'] = np.atleast_2d(np.asarray(cds, dtype=object))

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    coordsLocal = {}
    self.romName = None
    self.romType = None
    self.locations["coords"] = np.zeros(self.locations["inputCoords"].shape)
    map = {'x':0,'y':1,'z':2}
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getType().startswith("mesh"):
        coord = inputFile.getType().split("-")[-1].strip()
        if coord not in ['x', 'y', 'z']:
          raise IOError('Mesh type not == to x, y or z. Got: ' + coord)
        coordsLocal[coord] = self.readFoamFile(inputFile.getAbsFile())
        for i in range(len(self.locations["inputCoords"])):
          if self.locations["inputCoords"][i, map[coord]] in ['min','max','middle']:
            if self.locations["inputCoords"][i, map[coord]] == 'min':
              operator = np.min
            elif self.locations["inputCoords"][i, map[coord]] == 'max':
              operator = np.max
            else:
              operator = np.average
            v = operator(coordsLocal[coord][1])
          else:
            v = float(self.locations["inputCoords"][i, map[coord]])
          idx, _ = mathUtils.numpyNearestMatch(coordsLocal[coord][1],v)
          self.locations["coords"][i, map[coord]] = coordsLocal[coord][1][idx]
      if inputFile.getType().lower() == 'input':
        with open(inputFile.getAbsFile(), "r") as inputObj:
          xml = inputObj.read()
        xml = '<root>\n' + xml + '\n</root>'
        tree = ET.ElementTree(ET.fromstring(xml))
        root = tree.getroot()
        xmlFind = lambda str: root.findall(str)[0].text
        self.fomPath = xmlFind('./fullOrderModel/fomPath')
        self.fomName = xmlFind('./fullOrderModel/librarySolution/fomDirectoryName')
        if '~' in self.fomPath:
          self.fomPath = os.path.expanduser(self.fomPath)
        if not os.path.isabs(self.fomPath):
          workingDir = runInfo['WorkingDir']
          self.fomPath = os.path.join(workingDir, self.fomPath)
        self.fomDataFolder = os.path.join(self.fomPath,self.fomName,"postProcessing","probe","0","U")
        self.romName = xmlFind('./rom/romName')
        self.romType = xmlFind('./rom/romType')
        # read data from FOM
        datai = pd.read_csv(self.fomDataFolder,skiprows=3,header=None,sep=r'\s+').iloc[:,[0,1,2]].replace('[()]','',regex=True).astype(float)
        datai = datai.rename(columns={0:'t',1:'ux',2:'uy'})
        dataList = [datai]
        self.dataFom = pd.concat(dataList,keys=['fom'])

    if not len(coordsLocal):
      raise IOError('Mesh type files must be inputed (mesh-x, mesh-y, mesh-z). Got None!')
    if self.romName is None or self.romType is None:
      raise IOError('<romName> or <romType> not found in input file!')
    # find nearest
    neigh = neighbors.KNeighborsRegressor(n_neighbors=1)
    s = len(coordsLocal[coord][1])
    X = np.asarray([coordsLocal[coord][1] for coord in ['x', 'y', 'z']]).T
    del coordsLocal
    y = np.asarray (range( s ))
    neigh.fit(X, y)
    self.locations["loc"] = neigh.predict(self.locations["coords"])

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (lenght of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    # find the input file (check that one input is provided)
    inputToPerturb = self.findInps(inputFiles,"input")
    # create output file root
    outputfile = 'out~' + inputToPerturb[0].getBase()
    # create command
    # the input file name is hardcoded in AccelerateCFD (podInputs.xml)
    executeCommand = [("parallel", "podPrecompute -parallel"), ("parallel", "podROM -i podInputs.xml -parallel"), ("parallel", "podFlowReconstruct -parallel"), ("parallel", "podPostProcess aVelocityFOM -parallel"),  ("serial", "cd rom/"+self.romType+"_"+self.romName+"/"+"system"),   ("serial", "rm  controlDict"), ("serial", "cp controlDict.2 controlDict"), ("serial", "cd ../"), ("parallel", "postProcess -parallel"), ("serial", "cd ../..")]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def getInputExtension(self):
    """
      Return a tuple of possible file extensions for a simulation initialization file (e.g., input.i).
      @ In, None
      @ Out, validExtensions, tuple, tuple of valid extensions
    """
    return ('')

  def findInps(self,inputFiles, inputType):
    """
      Locates the input files required by AcellerateCFD Interface
      @ In, inputFiles, list, list of Files objects
      @ In, inputType, str, inputType to find (e.g. mesh-x, input, etc)
      @ Out, podDictInput, list, list containing AcellerateCFD required input files
    """
    podDictInput = []
    for inputFile in inputFiles:
      if inputFile.getType().strip().lower() == inputType.lower():
        podDictInput.append(inputFile)
    if len(podDictInput) == 0:
      raise IOError('no "'+inputType+'" type file has been found!')
    return podDictInput

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generate a new AccelerateCFD input file (txt format) from the original, changing parameters
      as specified in Kwargs['SampledVars'].
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    if 'dynamiceventtree' in str(samplerType).lower():
      raise IOError("Dynamic Event Tree-based samplers not supported by AccelerateCFD interface yet!")
    currentInputsToPerturb = self.findInps(currentInputFiles,"input")
    originalInputs         = self.findInps(oriInputFiles,"input")
    parser = GenericParser.GenericParser(currentInputsToPerturb)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputsToPerturb,originalInputs)
    return currentInputFiles

  def readFoamFile(self, filename):
    """
      This method is aimed to read a Open Foam file for accelerated CFD
      @ In, filename, str, the file name
      @ Out, content, dict, the open foam output content
    """
    with open(filename, "r") as foam:
      lines = foam.readlines()
      settings = {}
      for row, line in enumerate(lines):
        if line.strip().startswith("FoamFile"):
          info = lines[row+2:row+7]
          for var in info:
            inf, val = [v.strip().replace(";","").replace('"','') for v in var.split()]
            settings[inf] = val
        if line.strip().startswith("dimensions"):
          settings["dimensions"] = [int(v.replace("[","").replace("]","")) for v in line.split()[-1].replace(";","").split()]
        if len(settings) > 1:
          del lines
          break
    field = fieldParser.parseInternalField(filename)
    return settings, field

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    resultFolder = os.path.join(workingDir,"rom",self.romType +"_"+ self.romName, "processor0")
    resultingDirs = [os.path.join(resultFolder, o) for o in os.listdir(resultFolder)
                        if os.path.isdir(os.path.join(resultFolder,o))]
    guard = None
    for subDir in resultingDirs:
      time = subDir.split(os.path.sep)[-1]
      try:
        guard = float(time)
        break
      except ValueError:
        pass
    if guard is not None:
      failure = not (os.path.exists(os.path.join(subDir, "Urom")) or os.path.join(subDir, "srom"))
    return failure

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Called by RAVEN to modify output files (if needed) so that they are in a proper form.
      In this case, the default .mat output needs to be converted to .csv output, which is the
      format that RAVEN can communicate with.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    # open output file
    # resulting folder of processor*
    resultFolders = glob.glob(os.path.join(workingDir,"rom",self.romType +"_"+ self.romName,"processor*"))
    numberOfProcs = len(resultFolders)
    resultFolders.sort()
    resultFolder =  resultFolders[0]
    results = {}
    # resulting folder of ./processor*/time/
    resultingDirs = [os.path.join(resultFolder, o) for o in os.listdir(resultFolder)
                        if os.path.isdir(os.path.join(resultFolder,o))]
    resultingDirs.sort()
    timeList = []
    tsDirs =  resultingDirs
    # read time
    for ts in resultingDirs:
      try:
        time = float(ts.split(os.path.sep)[-1])
        timeList.append(time)
      except ValueError:
        tsDirs.pop(tsDirs.index(ts))
    resultingDirs =  tsDirs
    timeList.sort()
    results["time"] = np.zeros(len(timeList))
    for ts in resultingDirs:
      refDict =  ts
      fieldVector, fieldScalar = None, None
      for proc in range(numberOfProcs):
        # read field vector from ./processor*/time/Urom
        whereToRead =  refDict.replace("processor0", "processor"+str(proc))
        if os.path.exists(os.path.join(whereToRead, "Urom")):
          settingsVector, fieldVect = self.readFoamFile(os.path.join(whereToRead, "Urom"))
          if fieldVector is None:
            fieldVector = fieldVect
          else:
            fieldVector = np.concatenate((fieldVector, fieldVect), axis=0)
        # read field scalar from ./processor*/time/srom
        if os.path.exists(os.path.join(whereToRead, "srom")):
          settingsScalar, fieldScal = self.readFoamFile(os.path.join(whereToRead, "srom"))
          if fieldVector is None:
            fieldScalar =  fieldScal
          else:
            fieldScalar = np.concatenate((fieldScalar, fieldScal), axis=0)
      time =  float(settingsVector['location'])
      indx, _ = mathUtils.numpyNearestMatch(timeList, time)
      results["time"][indx] = time
      # read values of coordinates given by user from fieldVector
      if fieldVector is not None:
        for i in range(len(self.locations["loc"])):
          cord = settingsVector['class'] + "-"
          cord += str(tuple(self.locations['inputCoords'][i].tolist()))
          cord = cord.replace(" ", "").replace("'", "").replace("(", "").replace(")", "").replace(",", "_").replace(".","_")
          vals = fieldVector[int(self.locations["loc"][i])]
          for j, coord in enumerate(['x', 'y', 'z']):
            variableName = cord + "-" + coord
            val = vals[j]
            if variableName not in results:
              results[variableName] = np.zeros(len(timeList))
            results[variableName][indx] = val
      # read values of scalar variables from fieldScalar
      if fieldScalar is not None:
        for i in range(len(self.locations["loc"])):
          cord = settingsScalar['class'] + "-"
          cord += str(tuple(self.locations['inputCoords'][i].tolist()))
          cord = cord.replace(" ", "").replace("'", "").replace("(", "").replace(")", "").replace(",", "_").replace(".","_")
          variableName = cord
          val = fieldScalar[int(self.locations["loc"][i])]
          if variableName not in results:
            results[variableName] = np.zeros(len(timeList))
          results[variableName][indx] = val
      # process post-processing file for rom
      ppRomFile = os.path.join(workingDir,"rom",self.romType +"_"+ self.romName,"postProcessing","probe","0","Urom")
      if os.path.exists(ppRomFile):
        datai = pd.read_csv(ppRomFile,skiprows=3,header=None,sep=r'\s+').iloc[:,[0,1,2]].replace('[()]','',regex=True).astype(float)
        datai = datai.rename(columns={0:'t',1:'ux',2:'uy'})
        dataList = [datai]
        romData = pd.concat(dataList,keys=['rom'])
        # averageVelocity
        ufomux = self.dataFom.loc['fom'].mean()['ux']
        uromux = romData.loc['rom'].mean()['ux']
        ufomuy = self.dataFom.loc['fom'].mean()['uy']
        uromuy = romData.loc['rom'].mean()['uy']
        # standardDeviation
        ufomuxStd = self.dataFom.loc['fom'].std()['ux']
        uromuxStd = romData.loc['rom'].std()['ux']
        ufomuyStd = self.dataFom.loc['fom'].std()['uy']
        uromuyStd = romData.loc['rom'].std()['uy']
        uerr = math.sqrt((ufomux-uromux)**2 + (ufomuy-uromuy)**2)
        results["ufomux"] = np.asarray([ufomux]*len(timeList))
        results["uromux"] = np.asarray([uromux]*len(timeList))
        results["ufomuy"] = np.asarray([ufomuy]*len(timeList))
        results["uromuy"] = np.asarray([uromuy]*len(timeList))
        results["uerr"] = np.asarray([uerr]*len(timeList))
      else:
        print("WARNING: postprocessing file for rom "+self.romType +"_"+ self.romName+" not found! Path: "+ ppRomFile)
    return results
