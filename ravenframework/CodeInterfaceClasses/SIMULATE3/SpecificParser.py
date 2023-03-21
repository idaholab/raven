"""
Created on Aug. 10th, 2022
@author: khnguy22
comment: Specific parser for Simulate3 interface
"""

import os
from xml.etree import ElementTree as ET

class DataParser():
  """
    Parse the data in RAVEN input in to SIMULATE input
    and be saved for the perturbed input generator in Perturbed parser
    Note: this data parser is only called once
  """

  def __init__(self, inputFile):
    """
      Constructor.
      @ In, inputFiles, string, xml Simulate3 parameters file
      @ Out, None
    """
    self.inputFile = inputFile
    self.getParameters()

  def getParameters(self):
    """
      Get required parameters from xml file for generating
      Simulate3 input later.
      @ In, None
      @ Out, None
    """
    fullFile = os.path.join(self.inputFile)
    dorm = ET.parse(fullFile)
    root = dorm.getroot()
    self.batchNumber = int(root.find('batch').text)
    self.restartFile = root.find('restart_file').text.strip()
    self.loadPoint = root.find('load_point').text.strip()
    self.coreWidth = root.find('core_width').text.strip()
    self.axialNodes = root.find('axial_nodes').text.strip()
    self.csLib = root.find('cs_lib').text.strip()
    self.flow = root.find('flow').text.strip()
    self.power = root.find('power').text.strip()
    self.pressure = root.find('pressure').text.strip()
    self.inletTemperature = root.find('inlet_temperature').text.strip()
    self.depletion = root.find('depletion').text.strip()
    self.mapSize = root.find('map_size').text.strip()
    self.symmetry = root.find('symmetry').text.strip()
    self.numberAssemblies = int(root.find('number_assemblies').text)
    self.reflectorFlag = root.find('reflector').text.strip()
    self.faDict = []
    for fa in root.iter('FA'):
      self.faDict.append(fa.attrib)

class PerturbedPaser():
  """
    Parse value in the perturbed xml file replaces the nominal values by the perturbed values.
  """

  def __init__(self, inputFile, workingDir, inputName, perturbDict):
    """
    Constructor.
      @ In, inputFiles, string, xml Simulate3 varibles that will be perturbed file
      @ In, workingDir, string, absolute path to the working directory
      @ In, inputName, string, inputname for simulate3 input file
      @ In, perturbDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.inputFile = inputFile
    self.perturbDict = perturbDict
    self.workingDir = workingDir
    self.inputName = inputName
    # get perturbed value and create new xml file
    self.replaceOldFile()

  def replaceOldFile(self):
    """
    Replace orignal xml file with perturbed variables
    @ In, None
    @ Out, None
    """
    perturbedID = []
    perturbedVal = []

    for key, value in self.perturbDict.items():
      id_ = ''.join([n for n in key if n.isdigit()])
      try:
        id_ = int(id_)
      except:
        ValueError('There is no id indication in variable from RAVEN, please check!')
      perturbedID.append(int(id_))
      perturbedVal.append(int(value))
    fullFile = os.path.join(self.inputFile)
    dorm = ET.parse(fullFile)
    root = dorm.getroot()
    self.data = root
    for child in self.data:
      idx = perturbedID.index(int(child.attrib['id']))
      child.attrib['FAid'] = str(perturbedVal[idx])
    writedata = ET.tostring(root,encoding='utf8', method='xml')
    newfile = open(self.inputFile, "w")
    newfile.write(writedata.decode())
    newfile.close()

  def generateSim3Input(self, parameter):
    """
    Generate new input for SIMULATE3 to run
    @ In, parameter, DataParser Ojbect instance, data in DataParser class
    @ Out, None
    """
    file_ = open(f"{self.workingDir}/{self.inputName}",'w')
    if parameter.batchNumber <= 1:
      file_.write(f"'RES' '../../{parameter.restartFile}' {parameter.loadPoint}/\n")
    else:
      file_.write(f"'DIM.PWR' {parameter.coreWidth}/\n")
      file_.write(f"'DIM.CAL' {parameter.axialNodes} 2 2/\n")
      file_.write("'DIM.DEP' 'EXP' 'PIN' 'HTMO' 'HBOR' 'HTF' 'XEN' 'SAM' 'EBP'/     \n")
    file_.write("\n")
    loadingPattern = getMap(parameter, [int(child.attrib['FAid']) for child in self.data])
    file_.write(loadingPattern) # later, from get map
    file_.write("\n")
    if parameter.batchNumber <=1:
      pass
    else:
      raise ValueError("Larger batch number is not available for this version")
    file_.write(f"'LIB' '../../{parameter.csLib}' \n")
    file_.write(f"'COR.OPE' {parameter.power}, {parameter.flow}, {parameter.pressure}/\n")
    file_.write("'COR.TIN' {}/ \n".format(parameter.inletTemperature))
    file_.write("\n")
    if parameter.batchNumber >= 2:
      file_.write(f"'BAT.LAB' {parameter.batchNumber} 'CYC-{parameter.batchNumber}' /\n")
      file_.write(f"'DEP.CYC' 'CYCLE{parameter.batchNumber}' 0.0 {parameter.batchNumber}/\n")
    file_.write(f"'DEP.STA' 'AVE' 0.0 0.5 1 2 -1 {parameter.depletion} /\n")
    file_.write("'ITE.SRC' 'SET' 'EOLEXP' , , 0.02, , , , , , 'MINBOR' 10., , , , , 4, 4, , , /\n")
    file_.write("'STA'/\n")
    file_.write("'END'/\n")
    file_.close()

# Outside functions
def findType(faID,faDict):
  """
    Get type of FA ID
    @ In, faID, int/str, the id for FA
    @ In, faDict, list, list of FA xml input attributes
    @ Out, faType, list, list of FA types
  """
  faType = [id['type'] for id in faDict if id['FAid']==str(faID)][0]
  return faType

def getMap(parameter, locationList):
  """
    Genrate Loading Pattern
    @ In, parameter, DataParser Object Instance, Instance store the parameter data
    @ In, locationList, list, Location list from PerturbedPaser class
    @ Out, loadingPattern, str, Loading Pattern
  """
  maxType = max([id['type'] for id in parameter.faDict])
  numberSpaces = len(str(maxType)) + 1
  problemMap = getCoreMap(parameter.mapSize, parameter.symmetry,
                           parameter.numberAssemblies, parameter.reflectorFlag)
  rowCount = 1
  loadingPattern = ""
  faDict = parameter.faDict
  for row in range(25):    #max core 25x25
    if row in problemMap:
      loadingPattern += f"'FUE.TYP'  {rowCount},"
      for col in range(25):
        if col in problemMap[row]:
          if not problemMap[row][col]:
            if isinstance(problemMap[row][col], int):
              geneNumber = problemMap[row][col]
              gene = locationList[geneNumber]
              value = findType(gene,faDict)
              str_ = f"{value}"
              loadingPattern += f"{str_.rjust(numberSpaces)}"
            else:
              loadingPattern += f"{'0'.rjust(numberSpaces)}"
          else:
            geneNumber = problemMap[row][col]
            gene = locationList[geneNumber]
            value = findType(gene,faDict)
            str_ = f"{value}"
            loadingPattern += f"{str_.rjust(numberSpaces)}"
      loadingPattern += "/\n"
      rowCount += 1
  loadingPattern += "\n"

  return loadingPattern

def getCoreMap(mapSize, symmetry, numberAssemblies, reflectorFlag):
  """
    Get core map depending on symmetry, number of assemblies and reflector
    @ In, mapSize, string, Mapsize (full, quarter, octant)
    @ In, symmetry, string, symmetry key
    @ In, numberAssemblies, int, # of assemblies
    @ In, reflectorFlag, Boolean, reflectorFlag indicate usage of reflector
    @ Out, dictionatry (matrix), Coremap
  """
  if mapSize.lower() == "full_core" or mapSize.lower() == "full":
      mapKey = "FULL"
      allowedSymmetries = ("OCTANT","QUARTER_ROTATIONAL","QUARTER_MIRROR")
      if symmetry.upper() in allowedSymmetries:
          symmetryKey = symmetry.upper()
      else:
          ValueError(f"Unrecognized problem symmetry. Recognized symmetries are {allowedSymmetries}")
  elif mapSize.lower() == "quarter" or mapSize.lower() == "quarter_core":
      mapKey = "QUARTER"
      allowedSymmetries = ("OCTANT","QUARTER")
      if symmetry.upper() in allowedSymmetries:
          symmetryKey = symmetry.upper()
      else:
          ValueError(f"Unrecognized problem symmetry. Recognized symmetries are {allowedSymmetries}")
  else:
      ValueError("Unrecognized core map key.")
  if reflectorFlag.lower() not in ["yes","no","true",'false']:
      ValueError(f"Unrecognized keyword for reflector block! ")
  elif reflectorFlag.lower() == "true" or reflectorFlag.lower() == "yes":
      reflKey="WITH_REFLECTOR"
  else:
      reflKey="WITHOUT_REFLECTOR"

  return coreMaps[mapKey][symmetryKey][numberAssemblies][reflKey]

### coreMaps value
coreMaps = {}
coreMaps['FULL'] = {}
coreMaps['FULL']['OCTANT'] = {}
coreMaps['FULL']['OCTANT'][157] = {}
coreMaps['FULL']['OCTANT'][157]['WITH_REFLECTOR'] = { 0:{0:None,1:None,2:None,3:None,4:None,5:None,6:34,7:33,8:32,9:33,10:34,11:None,12:None,13:None,14:None, 15:None,16:None},
                                                       1:{0:None,1:None,2:None,3:None,4:31  ,5:30,  6:29,7:28,8:27,9:28,10:29,11:30,  12:31,  13:None,14:None, 15:None,16:None},
                                                       2:{0:None,1:None,2:None,3:26  ,4:25  ,5:24,  6:23,7:22,8:21,9:22,10:23,11:24,  12:25,  13:26,  14:None, 15:None,16:None},
                                                       3:{0:None,1:None,2:26  ,3:20  ,4:19  ,5:18,  6:17,7:16,8:15,9:16,10:17,11:18,  12:19,  13:20,  14:26  , 15:None,16:None},
                                                       4:{0:None,1:31,  2:25  ,3:19  ,4:14  ,5:13,  6:12,7:11,8:10,9:11,10:12,11:13,  12:14,  13:19,  14:25  , 15:31,  16:None},
                                                       5:{0:None,1:30,  2:24  ,3:18  ,4:13  ,5:9,   6:8, 7:7, 8:6, 9:7, 10:8, 11:9,   12:13,  13:18,  14:24  , 15:30,  16:None},
                                                       6:{0:34,  1:29,  2:23  ,3:17  ,4:12  ,5:8,   6:5, 7:4, 8:3, 9:4, 10:5, 11:8,   12:12,  13:17,  14:23  , 15:29,  16:34},
                                                       7:{0:33,  1:28,  2:22  ,3:16  ,4:11  ,5:7,   6:4, 7:2, 8:1, 9:2, 10:4, 11:7,   12:11,  13:16,  14:22  , 15:28,  16:33},
                                                       8:{0:32,  1:27,  2:21  ,3:15  ,4:10  ,5:6,   6:3, 7:1, 8:0, 9:1, 10:3, 11:6,   12:10,  13:15,  14:21  , 15:27,  16:32},
                                                       9:{0:33,  1:28,  2:22  ,3:16  ,4:11  ,5:7,   6:4, 7:2, 8:1, 9:2, 10:4, 11:7,   12:11,  13:16,  14:22  , 15:28,  16:33},
                                                      10:{0:34,  1:29,  2:23  ,3:17  ,4:12  ,5:8,   6:5, 7:4, 8:3, 9:4, 10:5, 11:8,   12:12,  13:17,  14:23  , 15:29,  16:34},
                                                      11:{0:None,1:30,  2:24  ,3:18  ,4:13  ,5:9,   6:8, 7:7 ,8:6, 9:7, 10:8, 11:9,   12:13,  13:18,  14:24  , 15:30,  16:None},
                                                      12:{0:None,1:31,  2:25  ,3:19  ,4:14  ,5:13,  6:12,7:11,8:10,9:11,10:12,11:13,  12:14,  13:19,  14:25  , 15:31,  16:None},
                                                      13:{0:None,1:None,2:26  ,3:20  ,4:19  ,5:18,  6:17,7:16,8:15,9:16,10:17,11:18,  12:19,  13:20,  14:26  , 15:None,16:None},
                                                      14:{0:None,1:None,2:None,3:26  ,4:25  ,5:24,  6:23,7:22,8:21,9:22,10:23,11:24,  12:25,  13:26,  14:None, 15:None,16:None},
                                                      15:{0:None,1:None,2:None,3:None,4:31  ,5:30,  6:29,7:28,8:27,9:28,10:29,11:30,  12:31,  13:None,14:None, 15:None,16:None},
                                                      16:{0:None,1:None,2:None,3:None,4:None,5:None,6:34,7:33,8:32,9:33,10:34,11:None,12:None,13:None,14:None, 15:None,16:None}}
coreMaps['FULL']['OCTANT'][157]['WITHOUT_REFLECTOR'] = { 0:{0:None,1:None,2:None,3:None,4:None,5:None,6:25,7:24,8:25,9:None,10:None,11:None,12:None,13:None,14:None},
                                                          1:{0:None,1:None,2:None,3:None,4:23,  5:22,  6:21,7:20,8:21,9:22,  10:23,  11:None,12:None,13:None,14:None},
                                                          2:{0:None,1:None,2:None,3:19,  4:18,  5:17,  6:16,7:15,8:16,9:17,  10:18,  11:19,  12:None,13:None,14:None},
                                                          3:{0:None,1:None,2:19,  3:14,  4:13,  5:12,  6:11,7:10,8:11,9:12,  10:13,  11:14,  12:19,  13:None,14:None},
                                                          4:{0:None,1:23,  2:18,  3:13,  4:9,   5:8,   6:7, 7:6, 8:7, 9:8,   10:9,   11:13,  12:18,  13:23,  14:None},
                                                          5:{0:None,1:22,  2:17,  3:12,  4:8,   5:5,   6:4, 7:3, 8:4, 9:5,   10:8,   11:12,  12:17,  13:22,  14:None},
                                                          6:{0:25,  1:21,  2:16,  3:11,  4:7,   5:4,   6:2, 7:1, 8:2, 9:4,   10:7,   11:11,  12:16,  13:21,  14:25},
                                                          7:{0:24,  1:20,  2:15,  3:10,  4:6,   5:3,   6:1, 7:0, 8:1, 9:3,   10:6,   11:10,  12:15,  13:20,  14:24},
                                                          8:{0:25,  1:21,  2:16,  3:11,  4:7,   5:4,   6:2, 7:1, 8:2, 9:4,   10:7,   11:11,  12:16,  13:21,  14:25},
                                                          9:{0:None,1:22,  2:17,  3:12,  4:8,   5:5,   6:4, 7:3, 8:4, 9:5,   10:8,   11:12,  12:17,  13:22,  14:None},
                                                         10:{0:None,1:23,  2:18,  3:13,  4:9,   5:8,   6:7, 7:6, 8:7,9:8,    10:9 ,  11:13,  12:18,  13:23,  14:None},
                                                         11:{0:None,1:None,2:19,  3:14,  4:13,  5:12,  6:11,7:10,8:11,9:12,  10:13,  11:14,  12:19,  13:None,14:None},
                                                         12:{0:None,1:None,2:None,3:19,  4:18,  5:17,  6:16,7:15,8:16,9:17,  10:18,  11:19,  12:None,13:None,14:None},
                                                         13:{0:None,1:None,2:None,3:None,4:23,  5:22,  6:21,7:20,8:21,9:22,  10:23,  11:None,12:None,13:None,14:None},
                                                         14:{0:None,1:None,2:None,3:None,4:None,5:None,6:25,7:24,8:25,9:None,10:None,11:None,12:None,13:None,14:None}}
coreMaps['FULL']['OCTANT'][193] = {}
coreMaps['FULL']['OCTANT'][193]['WITH_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:39,5:38,6:37,7:36,8:35,9:36,10:37,11:38,12:39,13:None,14:None,15:None,16:None},
                                                      1:{0:None,1:None,2:34,  3:33,  4:32,5:31,6:30,7:29,8:28,9:29,10:30,11:31,12:32,13:33,  14:34,  15:None,16:None},
                                                      2:{0:None,1:34,  2:27,  3:26,  4:25,5:24,6:23,7:22,8:21,9:22,10:23,11:24,12:25,13:26,  14:27,  15:34,  16:None},
                                                      3:{0:None,1:33,  2:26,  3:20,  4:19,5:18,6:17,7:16,8:15,9:16,10:17,11:18,12:19,13:20,  14:26,  15:33,  16:None},
                                                      4:{0:39,  1:32,  2:25,  3:19,  4:14,5:13,6:12,7:11,8:10,9:11,10:12,11:13,12:14,13:19,  14:25,  15:32,  16:39},
                                                      5:{0:38,  1:31,  2:24,  3:18,  4:13,5:9, 6:8, 7:7, 8:6, 9:7, 10:8, 11:9, 12:13,13:18,  14:24,  15:31,  16:38},
                                                      6:{0:37,  1:30,  2:23,  3:17,  4:12,5:8, 6:5, 7:4, 8:3, 9:4, 10:5, 11:8, 12:12,13:17,  14:23,  15:30,  16:37},
                                                      7:{0:36,  1:29,  2:22,  3:16,  4:11,5:7, 6:4, 7:2, 8:1, 9:2, 10:4, 11:7, 12:11,13:16,  14:22,  15:29,  16:36},
                                                      8:{0:35,  1:28,  2:21,  3:15,  4:10,5:6, 6:3, 7:1, 8:0, 9:1, 10:3, 11:6, 12:10,13:15,  14:21,  15:28,  16:35},       #Edited input count
                                                      9:{0:36,  1:29,  2:22,  3:16,  4:11,5:7, 6:4, 7:2, 8:1, 9:2, 10:4, 11:7, 12:11,13:16,  14:22,  15:29,  16:36},
                                                     10:{0:37,  1:30,  2:23,  3:17,  4:12,5:8, 6:5, 7:4, 8:3, 9:4, 10:5, 11:8, 12:12,13:17,  14:23,  15:30,  16:37},
                                                     11:{0:38,  1:31,  2:24,  3:18,  4:13,5:9, 6:8, 7:7, 8:6, 9:7, 10:8, 11:9, 12:13,13:18,  14:24,  15:31,  16:38},
                                                     12:{0:39,  1:32,  2:25,  3:19,  4:14,5:13,6:12,7:11,8:10,9:11,10:12,11:13,12:14,13:19,  14:25,  15:32,  16:39},
                                                     13:{0:None,1:33,  2:26,  3:20,  4:19,5:18,6:17,7:16,8:15,9:16,10:17,11:18,12:19,13:20,  14:26,  15:33,  16:None},
                                                     14:{0:None,1:34,  2:27,  3:26,  4:25,5:24,6:23,7:22,8:21,9:22,10:23,11:24,12:25,13:26,  14:27,  15:34,  16:None},
                                                     15:{0:None,1:None,2:34,  3:33,  4:32,5:31,6:30,7:29,8:28,9:29,10:30,11:31,12:32,13:33,  14:34,  15:None,16:None},
                                                     16:{0:None,1:None,2:None,3:None,4:39,5:38,6:37,7:36,8:35,9:36,10:37,11:38,12:39,13:None,14:None,15:None,16:None}}

coreMaps['FULL']['OCTANT'][193]['WITHOUT_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:30,5:29, 6:28,7:27,8:28,9:29, 10:30,11:None,12:None,13:None,14:None},
                                                         1:{0:None,1:None,2:26,  3:25,  4:24,5:23, 6:22,7:21,8:22,9:23, 10:24,11:25,  12:26,  13:None,14:None},
                                                         2:{0:None,1:26,  2:20,  3:19,  4:18,5:17, 6:16,7:15,8:16,9:17, 10:18,11:19,  12:20,  13:26,  14:None},
                                                         3:{0:None,1:25,  2:19,  3:14,  4:13,5:12, 6:11,7:10,8:11,9:12, 10:13,11:14,  12:19,  13:25,  14:None},
                                                         4:{0:30,  1:24,  2:18,  3:13,  4:9, 5:8,  6:7, 7:6, 8:7, 9:8,  10:9, 11:13,  12:18,  13:24,  14:30},
                                                         5:{0:29,  1:23,  2:17,  3:12,  4:8, 5:5,  6:4, 7:3, 8:4, 9:5,  10:8, 11:12,  12:17,  13:23,  14:29},
                                                         6:{0:28,  1:22,  2:16,  3:11,  4:7, 5:4,  6:2, 7:1, 8:2, 9:4,  10:7, 11:11,  12:16,  13:22,  14:28},
                                                         7:{0:27,  1:21,  2:15,  3:10,  4:6, 5:3,  6:1, 7:0, 8:1, 9:3,  10:6, 11:10,  12:15,  13:21,  14:27},
                                                         8:{0:28,  1:22,  2:16,  3:11,  4:7, 5:4,  6:2, 7:1, 8:2, 9:4,  10:7, 11:11,  12:16,  13:22,  14:28},
                                                         9:{0:29,  1:23,  2:17,  3:12,  4:8, 5:5,  6:4, 7:3, 8:4, 9:5,  10:8, 11:12,  12:17,  13:23,  14:29},
                                                        10:{0:30,  1:24,  2:18,  3:13,  4:9, 5:8,  6:7, 7:6, 8:7, 9:8,  10:9, 11:13,  12:18,  13:24,  14:30},
                                                        11:{0:None,1:25,  2:19,  3:14,  4:13,5:12, 6:11,7:10,8:11,9:12, 10:13,11:14,  12:19,  13:25,  14:None},
                                                        12:{0:None,1:26,  2:20,  3:19,  4:18,5:17, 6:16,7:15,8:16,9:17, 10:18,11:19,  12:20,  13:26,  14:None},
                                                        13:{0:None,1:None,2:26,  3:25,  4:24,5:23, 6:22,7:21,8:22,9:23, 10:24,11:25,  12:26,  13:None,14:None},
                                                        14:{0:None,1:None,2:None,3:None,4:30,5:29, 6:28,7:27,8:28,9:29, 10:30,11:None,12:None,13:None,14:None}}
coreMaps['FULL']['OCTANT'][241] = {}
coreMaps['FULL']['OCTANT'][241]['WITH_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:None,5:47, 6:46, 7:45, 8:44, 9:43, 10:44, 11:45, 12:46, 13:47, 14:None,15:None,16:None,17:None,18:None},
                                                      1:{0:None,1:None,2:None,3:42,  4:41,  5:40, 6:39, 7:38, 8:37, 9:36, 10:37, 11:38, 12:39, 13:40, 14:41,  15:42,  16:None,17:None,18:None},
                                                      2:{0:None,1:None,2:35,  3:34,  4:33,  5:32, 6:31, 7:30, 8:29, 9:28, 10:29, 11:30, 12:31, 13:32, 14:33,  15:34,  16:35,  17:None,18:None},
                                                      3:{0:None,1:42,  2:34,  3:27,  4:26,  5:25, 6:24, 7:23, 8:22, 9:21, 10:22, 11:23, 12:24, 13:25, 14:26,  15:27,  16:34,  17:42,  18:None},
                                                      4:{0:None,1:41,  2:33,  3:26,  4:20,  5:19, 6:18, 7:17, 8:16, 9:15, 10:16, 11:17, 12:18, 13:19, 14:20,  15:26,  16:33,  17:41,  18:None},
                                                      5:{0:47,  1:40,  2:32,  3:25,  4:19,  5:14, 6:13, 7:12, 8:11, 9:10, 10:11, 11:12, 12:13, 13:14, 14:19,  15:25,  16:32,  17:40,  18:47},
                                                      6:{0:46,  1:39,  2:31,  3:24,  4:18,  5:13, 6: 9, 7: 8, 8: 7, 9: 6, 10: 7, 11: 8, 12: 9, 13:13, 14:18,  15:24,  16:31,  17:39,  18:46},
                                                      7:{0:45,  1:38,  2:30,  3:23,  4:17,  5:12, 6: 8, 7: 5, 8: 4, 9: 3, 10: 4, 11: 5, 12: 8, 13:12, 14:17,  15:23,  16:30,  17:38,  18:45},
                                                      8:{0:44,  1:37,  2:29,  3:22,  4:16,  5:11, 6: 7, 7: 4, 8: 2, 9: 1, 10: 2, 11: 4, 12: 7, 13:11, 14:16,  15:22,  16:29,  17:37,  18:44},
                                                      9:{0:43,  1:36,  2:28,  3:21,  4:15,  5:10, 6: 6, 7: 3, 8: 1, 9: 0, 10: 1, 11: 3, 12: 6, 13:10, 14:15,  15:21,  16:28,  17:36,  18:43},
                                                     10:{0:44,  1:37,  2:29,  3:22,  4:16,  5:11, 6: 7, 7: 4, 8: 2, 9: 1, 10: 2, 11: 4, 12: 7, 13:11, 14:16,  15:22,  16:29,  17:37,  18:44},
                                                     11:{0:45,  1:38,  2:30,  3:23,  4:17,  5:12, 6: 8, 7: 5, 8: 4, 9: 3, 10: 4, 11: 5, 12: 8, 13:12, 14:17,  15:23,  16:30,  17:38,  18:45},
                                                     12:{0:46,  1:39,  2:31,  3:24,  4:18,  5:13, 6: 9, 7: 8, 8: 7, 9: 6, 10: 7, 11: 8, 12: 9, 13:13, 14:18,  15:24,  16:31,  17:39,  18:46},
                                                     13:{0:47,  1:40,  2:32,  3:25,  4:19,  5:14, 6:13, 7:12, 8:11, 9:10, 10:11, 11:12, 12:13, 13:14, 14:19,  15:25,  16:32,  17:40,  18:47},
                                                     14:{0:None,1:41,  2:33,  3:26,  4:20,  5:19, 6:18, 7:17, 8:16, 9:15, 10:16, 11:17, 12:18, 13:19, 14:20,  15:26,  16:33,  17:41,  18:None},
                                                     15:{0:None,1:42,  2:34,  3:27,  4:26,  5:25, 6:24, 7:23, 8:22, 9:21, 10:22, 11:23, 12:24, 13:25, 14:26,  15:27,  16:34,  17:42,  18:None},
                                                     16:{0:None,1:None,2:35,  3:34,  4:33,  5:32, 6:31, 7:30, 8:29, 9:28, 10:29, 11:30, 12:31, 13:32, 14:33,  15:34,  16:35,  17:None,18:None},
                                                     17:{0:None,1:None,2:None,3:42,  4:41,  5:40, 6:39, 7:38, 8:37, 9:36, 10:37, 11:38, 12:39, 13:40, 14:41,  15:42,  16:None,17:None,18:None},
                                                     18:{0:None,1:None,2:None,3:None,4:None,5:47, 6:46, 7:45, 8:44, 9:43, 10:44, 11:45, 12:46, 13:47, 14:None,15:None,16:None,17:None,18:None}}
coreMaps['FULL']['OCTANT'][241]['WITHOUT_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:None,5:37, 6:36,  7:35,  8:34, 9:35, 10:36, 11:37, 12:None,13:None,14:None,15:None,16:None},
                                                         1:{0:None,1:None,2:None,3:33,  4:32,  5:31, 6:30,  7:29,  8:28, 9:29, 10:30, 11:31, 12:32,  13:33,  14:None,15:None,16:None},
                                                         2:{0:None,1:None,2:27,  3:26,  4:25,  5:24, 6:23,  7:22,  8:21, 9:22, 10:23, 11:24, 12:25,  13:26,  14:27,  15:None,16:None},
                                                         3:{0:None,1:33,  2:26,  3:20,  4:19,  5:18, 6:17,  7:16,  8:15, 9:16, 10:17, 11:18, 12:19,  13:20,  14:26,  15:33,  16:None},
                                                         4:{0:None,1:32,  2:25,  3:19,  4:14,  5:13, 6:12,  7:11,  8:10, 9:11, 10:12, 11:13, 12:14,  13:19,  14:25,  15:32,  16:None},
                                                         5:{0:37  ,1:31,  2:24,  3:18,  4:13,  5:9,  6:8,   7:7,   8:6,  9:7,  10:8,  11:9,  12:13,  13:18,  14:24,  15:31,  16:37  },
                                                         6:{0:36  ,1:30,  2:23,  3:17,  4:12,  5:8,  6:5,   7:4,   8:3,  9:4,  10:5,  11:8,  12:12,  13:17,  14:23,  15:30,  16:36  },
                                                         7:{0:35  ,1:29,  2:22,  3:16,  4:11,  5:7,  6:4,   7:2,   8:1,  9:2,  10:4,  11:7,  12:11,  13:16,  14:22,  15:29,  16:35  },
                                                         8:{0:34  ,1:28,  2:21,  3:15,  4:10,  5:6,  6:3,   7:1,   8:0,  9:1,  10:3,  11:6,  12:10,  13:15,  14:21,  15:28,  16:34  },
                                                         9:{0:35  ,1:29,  2:22,  3:16,  4:11,  5:7,  6:4,   7:2,   8:1,  9:2,  10:4,  11:7,  12:11,  13:16,  14:22,  15:29,  16:35  },
                                                        10:{0:36  ,1:30,  2:23,  3:17,  4:12,  5:8,  6:5,   7:4,   8:3,  9:4,  10:5,  11:8,  12:12,  13:17,  14:23,  15:30,  16:36  },
                                                        11:{0:37  ,1:31,  2:24,  3:18,  4:13,  5:9,  6:8,   7:7,   8:6,  9:7,  10:8,  11:9,  12:13,  13:18,  14:24,  15:31,  16:37  },
                                                        12:{0:None,1:32,  2:25,  3:19,  4:14,  5:13, 6:12,  7:11,  8:10, 9:11, 10:12, 11:13, 12:14,  13:19,  14:25,  15:32,  16:None},
                                                        13:{0:None,1:33,  2:26,  3:20,  4:19,  5:18, 6:17,  7:16,  8:15, 9:16, 10:17, 11:18, 12:19,  13:20,  14:26,  15:33,  16:None},
                                                        14:{0:None,1:None,2:27,  3:26,  4:25,  5:24, 6:23,  7:22,  8:21, 9:22, 10:23, 11:24, 12:25,  13:26,  14:27,  15:None,16:None},
                                                        15:{0:None,1:None,2:None,3:33,  4:32,  5:31, 6:30,  7:29,  8:28, 9:29, 10:30, 11:31, 12:32,  13:33,  14:None,15:None,16:None},
                                                        16:{0:None,1:None,2:None,3:None,4:None,5:37, 6:36,  7:35,  8:34, 9:35, 10:36, 11:37, 12:None,13:None,14:None,15:None,16:None}}
coreMaps['FULL']['QUARTER_MIRROR'] = {}
coreMaps['FULL']['QUARTER_MIRROR'][157] = {}
coreMaps['FULL']['QUARTER_MIRROR'][157]['WITH_REFLECTOR'] =  {0:{0:None,1:None,2:None,3:None,4:None,5:None,6:55, 7:54, 8:53,9:54, 10:55,11:None,12:None,13:None,14:None,15:None,16:None},
                                                               1:{0:None,1:None,2:None,3:None,4:52,  5:51,  6:50, 7:49, 8:48,9:49, 10:50,11:51,  12:52,  13:None,14:None,15:None,16:None},
                                                               2:{0:None,1:None,2:None,3:47,  4:46,  5:45,  6:44, 7:43, 8:42,9:43, 10:44,11:45,  12:46,  13:47,  14:None,15:None,16:None},
                                                               3:{0:None,1:None,2:41,  3:40,  4:39,  5:38,  6:37, 7:36, 8:35,9:36, 10:37,11:38,  12:39,  13:40,  14:41,  15:None,16:None},
                                                               4:{0:None,1:34,  2:33,  3:32,  4:31,  5:30,  6:29, 7:28, 8:27,9:28, 10:29,11:30,  12:31,  13:32,  14:33,  15:34,  16:None},
                                                               5:{0:None,1:26,  2:25,  3:24,  4:23,  5:22,  6:21, 7:20, 8:19,9:20, 10:21,11:22,  12:23,  13:24,  14:25,  15:26,  16:None},
                                                               6:{0:18,  1:17,  2:16,  3:15,  4:14,  5:13,  6:12, 7:11, 8:10,9:11, 10:12,11:13,  12:14,  13:15,  14:16,  15:17,  16:18},
                                                               7:{0:9,   1:8,   2:7,   3:6,   4:5,   5:4,   6:3,  7:2,  8:1, 9:2,  10:3, 11:4,   12:5,   13:6,   14:7,   15:8,   16:9},
                                                               8:{0:53,  1:48,  2:42,  3:35,  4:27,  5:19,  6:10, 7:1,  8:0, 9:1,  10:10,11:19,  12:27,  13:35,  14:42,  15:48,  16:53},
                                                               9:{0:9,   1:8,   2:7,   3:6,   4:5,   5:4,   6:3,  7:2,  8:1, 9:2,  10:3, 11:4,   12:5,   13:6,   14:7,   15:8,   16:9},
                                                              10:{0:18,  1:17,  2:16,  3:15,  4:14,  5:13,  6:12, 7:11, 8:10,9:11, 10:12,11:13,  12:14,  13:15,  14:16,  15:17,  16:18},
                                                              11:{0:None,1:26,  2:25,  3:24,  4:23,  5:22,  6:21, 7:20, 8:19,9:20, 10:21,11:22,  12:23,  13:24,  14:25,  15:26,  16:None},
                                                              12:{0:None,1:34,  2:33,  3:32,  4:31,  5:30,  6:29, 7:28, 8:27,9:28, 10:29,11:30,  12:31,  13:32,  14:33,  15:34,  16:None},
                                                              13:{0:None,1:None,2:41,  3:40,  4:39,  5:38,  6:37, 7:36, 8:35,9:36, 10:37,11:38,  12:39,  13:40,  14:41,  15:None,16:None},
                                                              14:{0:None,1:None,2:None,3:47,  4:46,  5:45,  6:44, 7:43, 8:42,9:43, 10:44,11:45,  12:46,  13:47,  14:None,15:None,16:None},
                                                              15:{0:None,1:None,2:None,3:None,4:52,  5:51,  6:50, 7:49, 8:48,9:49, 10:50,11:51,  12:52,  13:None,14:None,15:None,16:None},
                                                              16:{0:None,1:None,2:None,3:None,4:None,5:None,6:55, 7:54, 8:53,9:54, 10:55,11:None,12:None,13:None,14:None,15:None,16:None}}
coreMaps['FULL']['QUARTER_MIRROR'][157]['WITHOUT_REFLECTOR'] = {0:{0:None, 1:None,  2:None,  3:None,  4:None,5:None, 6:39, 7:38, 8:39, 9:None,10:None,11:None,12:None,13:None,14:None},
                                                                 1:{0:None, 1:None,  2:None,  3:None,  4:37,  5:36,   6:35, 7:34, 8:35, 9:36,  10:37,  11:None,12:None,13:None,14:None},
                                                                 2:{0:None, 1:None,  2:None,  3:33,    4:32,  5:31,   6:30, 7:29, 8:30, 9:31,  10:32,  11:33,  12:None,13:None,14:None},
                                                                 3:{0:None, 1:None,  2:28,    3:27,    4:26,  5:25,   6:24, 7:23, 8:24, 9:25,  10:26,  11:27,  12:28,  13:None,14:None},
                                                                 4:{0:None, 1:22,    2:21,    3:20,    4:19,  5:18,   6:17, 7:16, 8:17, 9:18,  10:19,  11:20,  12:21,  13:22,  14:None},
                                                                 5:{0:None, 1:15,    2:14,    3:13,    4:12,  5:11,   6:10, 7:9,  8:10, 9:11,  10:12,  11:13,  12:14,  13:15,  14:None},
                                                                 6:{0:8,    1:7,     2:6,     3:5,     4:4,   5:3,    6:2,  7:1,  8: 2, 9: 3,  10: 4,  11:5,   12:6,   13:7,   14:8},
                                                                 7:{0:38,   1:34,    2:29,    3:23,    4:16,  5:9,    6:1,  7:0,  8: 1, 9: 9,  10:16,  11:23,  12:29,  13:34,  14:38},
                                                                 8:{0:8,    1:7,     2:6,     3:5,     4:4,   5:3,    6:2,  7:1,  8: 2, 9: 3,  10: 4,  11:5,   12:6,   13:7,   14:8},
                                                                 9:{0:None, 1:15,    2:14,    3:13,    4:12,  5:11,   6:10, 7:9,  8:10, 9:11,  10:12,  11:13,  12:14,  13:15,  14:None},
                                                                10:{0:None, 1:22,    2:21,    3:20,    4:19,  5:18,   6:17, 7:16, 8:17, 9:18,  10:19,  11:20,  12:21,  13:22,  14:None},
                                                                11:{0:None, 1:None,  2:28,    3:27,    4:26,  5:25,   6:24, 7:23, 8:24, 9:25,  10:26,  11:27,  12:28,  13:None,14:None},
                                                                12:{0:None, 1:None,  2:None,  3:33,    4:32,  5:31,   6:30, 7:29, 8:30, 9:31,  10:32,  11:33,  12:None,13:None,14:None},
                                                                13:{0:None, 1:None,  2:None,  3:None,  4:37,  5:36,   6:35, 7:34, 8:35, 9:36,  10:37,  11:None,12:None,13:None,14:None},
                                                                14:{0:None, 1:None,  2:None,  3:None,  4:None,5:None, 6:39, 7:38, 8:39, 9:None,10:None,11:None,12:None,13:None,14:None}}
coreMaps['FULL']['QUARTER_MIRROR'][193] = {}
coreMaps['FULL']['QUARTER_MIRROR'][193]['WITH_REFLECTOR'] = {}
coreMaps['FULL']['QUARTER_MIRROR'][193]['WITHOUT_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:46,5:45,6:44,  7:43,  8:44,9:45, 10:46,11:None,12:None,13:None,14:None},
                                                                 1:{0:None,1:None,2:None,3:42,  4:41,5:40,6:39,  7:38,  8:39,9:40, 10:41,11:42,  12:None,13:None,14:None},
                                                                 2:{0:None,1:None,2:37,  3:36,  4:35,5:34,6:33,  7:32,  8:33,9:34, 10:35,11:36,  12:37,  13:None,14:None},
                                                                 3:{0:None,1:31,  2:30,  3:29,  4:28,5:27,6:26,  7:25,  8:26,9:27, 10:28,11:29,  12:30,  13:31,  14:None},
                                                                 4:{0:24,  1:23,  2:22,  3:21,  4:20,5:19,6:18,  7:17,  8:18,9:19, 10:20,11:21,  12:22,  13:23,  14:24},
                                                                 5:{0:16,  1:15,  2:14,  3:13,  4:12,5:11,6:10,  7:9,   8:10,9:11, 10:12,11:13,  12:14,  13:15,  14:16},
                                                                 6:{0:8,   1:7,   2:6,   3:5,   4:4, 5:3, 6:2,   7:1,   8:2, 9:3,  10:4, 11:5,   12:6,   13:7,   14:8},
                                                                 7:{0:43,  1:38,  2:32,  3:25,  4:17,5:9, 6:1,   7:0,   8:1, 9:9,  10:17,11:25,  12:32,  13:38,  14:43},
                                                                 8:{0:8,   1:7,   2:6,   3:5,   4:4, 5:3, 6:2,   7:1,   8:2, 9:3,  10:4, 11:5,   12:6,   13:7,   14:8},
                                                                 9:{0:16,  1:15,  2:14,  3:13,  4:12,5:11,6:10,  7:9,   8:10,9:11, 10:12,11:13,  12:14,  13:15,  14:16},
                                                                10:{0:24,  1:23,  2:22,  3:21,  4:20,5:19,6:18,  7:17,  8:18,9:19, 10:20,11:21,  12:22,  13:23,  14:24},
                                                                11:{0:None,1:31,  2:30,  3:29,  4:28,5:27,6:26,  7:25,  8:26,9:27, 10:28,11:29,  12:30,  13:31,  14:None},
                                                                12:{0:None,1:None,2:37,  3:36,  4:35,5:34,6:33,  7:32,  8:33,9:34, 10:35,11:36,  12:37,  13:None,14:None},
                                                                13:{0:None,1:None,2:None,3:42,  4:41,5:40,6:39,  7:38,  8:39,9:40, 10:41,11:42,  12:None,13:None,14:None},
                                                                14:{0:None,1:None,2:None,3:None,4:46,5:45,6:44,  7:43,  8:44,9:45, 10:46,11:None,12:None,13:None,14:None}}

coreMaps['FULL']['QUARTER_MIRROR'][241] = {}
coreMaps['FULL']['QUARTER_MIRROR'][241]['WITH_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:None,5:78, 6:77, 7:76, 8:75, 9:74, 10:75, 11:76, 12:77, 13:78, 14:None,15:None,16:None,17:None,18:None},
                                                              1:{ 0:None,1:None,2:None,3:73,  4:72,  5:71, 6:70, 7:69, 8:68, 9:67, 10:68, 11:69, 12:70, 13:71, 14:72,  15:73,  16:None,17:None,18:None},
                                                              2:{ 0:None,1:None,2:66,  3:65,  4:64,  5:63, 6:62, 7:61, 8:60, 9:59, 10:60, 11:61, 12:62, 13:63, 14:64,  15:65,  16:66,  17:None,18:None},
                                                              3:{ 0:None,1:58,  2:57,  3:56,  4:55,  5:54, 6:53, 7:52, 8:51, 9:50, 10:51, 11:52, 12:53, 13:54, 14:55,  15:56,  16:57,  17:58,  18:None},
                                                              4:{ 0:None,1:49,  2:48,  3:47,  4:46,  5:45, 6:44, 7:43, 8:42, 9:41, 10:42, 11:43, 12:44, 13:45, 14:46,  15:47,  16:48,  17:49,  18:None},
                                                              5:{ 0:40,  1:39,  2:38,  3:37,  4:36,  5:35, 6:34, 7:33, 8:32, 9:31, 10:32, 11:33, 12:34, 13:35, 14:36,  15:37,  16:38,  17:39,  18:40},
                                                              6:{ 0:30,  1:29,  2:28,  3:27,  4:26,  5:25, 6:24, 7:23, 8:22, 9:21, 10:22, 11:23, 12:24, 13:25, 14:26,  15:27,  16:28,  17:29,  18:30},
                                                              7:{ 0:20,  1:19,  2:18,  3:17,  4:16,  5:15, 6:14, 7:13, 8:12, 9:11, 10:12, 11:13, 12:14, 13:15, 14:16,  15:17,  16:18,  17:19,  18:20},
                                                              8:{ 0:10,  1: 9,  2: 8,  3: 7,  4: 6,  5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6,  15: 7,  16: 8,  17: 9,  18:10},
                                                              9:{ 0:74,  1:67,  2:59,  3:50,  4:41,  5:31, 6:21, 7:11, 8: 1, 9: 0, 10: 1, 11:11, 12:21, 13:31, 14:41,  15:50,  16:59,  17:67,  18:74},
                                                             10:{ 0:10,  1: 9,  2: 8,  3: 7,  4: 6,  5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6,  15: 7,  16: 8,  17: 9,  18:10},
                                                             11:{ 0:20,  1:19,  2:18,  3:17,  4:16,  5:15, 6:14, 7:13, 8:12, 9:11, 10:12, 11:13, 12:14, 13:15, 14:16,  15:17,  16:18,  17:19,  18:20},
                                                             12:{ 0:30,  1:29,  2:28,  3:27,  4:26,  5:25, 6:24, 7:23, 8:22, 9:21, 10:22, 11:23, 12:24, 13:25, 14:26,  15:27,  16:28,  17:29,  18:30},
                                                             13:{ 0:40,  1:39,  2:38,  3:37,  4:36,  5:35, 6:34, 7:33, 8:32, 9:31, 10:32, 11:33, 12:34, 13:35, 14:36,  15:37,  16:38,  17:39,  18:40},
                                                             14:{ 0:None,1:49,  2:48,  3:47,  4:46,  5:45, 6:44, 7:43, 8:42, 9:41, 10:42, 11:43, 12:44, 13:45, 14:46,  15:47,  16:48,  17:49,  18:None},
                                                             15:{ 0:None,1:58,  2:57,  3:56,  4:55,  5:54, 6:53, 7:52, 8:51, 9:50, 10:51, 11:52, 12:53, 13:54, 14:55,  15:56,  16:57,  17:58,  18:None},
                                                             16:{ 0:None,1:None,2:66,  3:65,  4:64,  5:63, 6:62, 7:61, 8:60, 9:59, 10:60, 11:61, 12:62, 13:63, 14:64,  15:65,  16:66,  17:None,18:None},
                                                             17:{ 0:None,1:None,2:None,3:73,  4:72,  5:71, 6:70, 7:69, 8:68, 9:67, 10:68, 11:69, 12:70, 13:71, 14:72,  15:73,  16:None,17:None,18:None},
                                                             18:{ 0:None,1:None,2:None,3:None,4:None,5:78, 6:77, 7:76, 8:75, 9:74, 10:75, 11:76, 12:77, 13:78, 14:None,15:None,16:None,17:None,18:None}}
coreMaps['FULL']['QUARTER_MIRROR'][241]['WITHOUT_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:None,5:60, 6:59, 7:58, 8:57, 9:58, 10:59, 11:60, 12:None, 13:None, 14:None, 15:None, 16:None},
                                                                 1:{0:None,1:None,2:None,3:56,  4:55,  5:54, 6:53, 7:52, 8:51, 9:52, 10:53, 11:54, 12:55,   13:56,   14:None, 15:None, 16:None},
                                                                 2:{0:None,1:None,2:50,  3:49,  4:48,  5:47, 6:46, 7:45, 8:44, 9:45, 10:46, 11:47, 12:48,   13:49,   14:50,   15:None, 16:None},
                                                                 3:{0:None,1:43,  2:42,  3:41,  4:40,  5:39, 6:38, 7:37, 8:36, 9:37, 10:38, 11:39, 12:40,   13:41,   14:42,   15:43,   16:None},
                                                                 4:{0:None,1:35,  2:34,  3:33,  4:32,  5:31, 6:30, 7:29, 8:28, 9:29, 10:30, 11:31, 12:32,   13:33,   14:34,   15:35,   16:None},
                                                                 5:{0:27,  1:26,  2:25,  3:24,  4:23,  5:22, 6:21, 7:20, 8:19, 9:20, 10:21, 11:22, 12:23,   13:24,   14:25,   15:26,   16:27  },
                                                                 6:{0:18,  1:17,  2:16,  3:15,  4:14,  5:13, 6:12, 7:11, 8:10, 9:11, 10:12, 11:13, 12:14,   13:15,   14:16,   15:17,   16:18  },
                                                                 7:{0: 9,  1: 8,  2: 7,  3: 6,  4: 5,  5: 4, 6: 3, 7: 2, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5,   13: 6,   14: 7,   15: 8,   16: 9  },
                                                                 8:{0:57,  1:51,  2:44,  3:36,  4:28,  5:19, 6:10, 7: 1, 8: 0, 9: 1, 10:10, 11:19, 12:28,   13:36,   14:44,   15:51,   16:57  },
                                                                 9:{0: 9,  1: 8,  2: 7,  3: 6,  4: 5,  5: 4, 6: 3, 7: 2, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5,   13: 6,   14: 7,   15: 8,   16: 9  },
                                                                10:{0:18,  1:17,  2:16,  3:15,  4:14,  5:13, 6:12, 7:11, 8:10, 9:11, 10:12, 11:13, 12:14,   13:15,   14:16,   15:17,   16:18  },
                                                                11:{0:27,  1:26,  2:25,  3:24,  4:23,  5:22, 6:21, 7:20, 8:19, 9:20, 10:21, 11:22, 12:23,   13:24,   14:25,   15:26,   16:27  },
                                                                12:{0:None,1:35,  2:34,  3:33,  4:32,  5:31, 6:30, 7:29, 8:28, 9:29, 10:30, 11:31, 12:32,   13:33,   14:34,   15:35,   16:None},
                                                                13:{0:None,1:43,  2:42,  3:41,  4:40,  5:39, 6:38, 7:37, 8:36, 9:37, 10:38, 11:39, 12:40,   13:41,   14:42,   15:43,   16:None},
                                                                14:{0:None,1:None,2:50,  3:49,  4:48,  5:47, 6:46, 7:45, 8:44, 9:45, 10:46, 11:47, 12:48,   13:49,   14:50,   15:None, 16:None},
                                                                15:{0:None,1:None,2:None,3:56,  4:55,  5:54, 6:53, 7:52, 8:51, 9:52, 10:53, 11:54, 12:55,   13:56,   14:None, 15:None, 16:None},
                                                                16:{0:None,1:None,2:None,3:None,4:None,5:60, 6:59, 7:58, 8:57, 9:58, 10:59, 11:60, 12:None, 13:None, 14:None, 15:None, 16:None}}
coreMaps['FULL']['QUARTER_ROTATIONAL'] = {}
coreMaps['FULL']['QUARTER_ROTATIONAL'][157] = {}
coreMaps['FULL']['QUARTER_ROTATIONAL'][157]['WITH_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:None,5:None,6:55, 7:54,8:53,9:9, 10:18,11:None,12:None,13:None,14:None,15:None,16:None},
                                                                  1:{0:None,1:None,2:None,3:None,4:52,  5:51,  6:50, 7:49,8:48,9:8, 10:17,11:26,  12:34,  13:None,14:None,15:None,16:None},
                                                                  2:{0:None,1:None,2:None,3:47,  4:46,  5:45,  6:44, 7:43,8:42,9:7, 10:16,11:25,  12:33,  13:41,  14:None,15:None,16:None},
                                                                  3:{0:None,1:None,2:41,  3:40,  4:39,  5:38,  6:37, 7:36,8:35,9:6, 10:15,11:24,  12:32,  13:40,  14:47,  15:None,16:None},
                                                                  4:{0:None,1:34,  2:33,  3:32,  4:31,  5:30,  6:29, 7:28,8:27,9:5, 10:14,11:23,  12:31,  13:39,  14:46,  15:52,  16:None},
                                                                  5:{0:None,1:26,  2:25,  3:24,  4:23,  5:22,  6:21, 7:20,8:19,9:4, 10:13,11:22,  12:30,  13:38,  14:45,  15:51,  16:None},
                                                                  6:{0:18,  1:17,  2:16,  3:15,  4:14,  5:13,  6:12, 7:11,8:10,9:3, 10:12,11:21,  12:29,  13:37,  14:44,  15:50,  16:55},
                                                                  7:{0:9,   1:8,   2:7,   3:6,   4:5,   5:4,   6:3,  7:2, 8:1, 9:2, 10:11,11:20,  12:28,  13:36,  14:43,  15:49,  16:54},
                                                                  8:{0:53,  1:48,  2:42,  3:35,  4:27,  5:19,  6:10, 7:1, 8:0, 9:1, 10:10,11:19,  12:27,  13:35,  14:42,  15:48,  16:53},
                                                                  9:{0:54,  1:49,  2:43,  3:36,  4:28,  5:20,  6:11, 7:2, 8:1, 9:2, 10:3, 11:4,   12:5,   13:6,   14:7,   15:8,   16:9},
                                                                 10:{0:55,  1:50,  2:44,  3:37,  4:29,  5:21,  6:12, 7:3, 8:10,9:11,10:12,11:13,  12:14,  13:15,  14:16,  15:17,  16:18},
                                                                 11:{0:None,1:51,  2:45,  3:38,  4:30,  5:22,  6:13, 7:4, 8:19,9:20,10:21,11:22,  12:23,  13:24,  14:25,  15:26,  16:None},
                                                                 12:{0:None,1:52,  2:46,  3:39,  4:31,  5:23,  6:14, 7:5, 8:27,9:28,10:29,11:30,  12:31,  13:32,  14:33,  15:34,  16:None},
                                                                 13:{0:None,1:None,2:47,  3:40,  4:32,  5:24,  6:15, 7:6, 8:35,9:36,10:37,11:38,  12:39,  13:40,  14:41,  15:None,16:None},
                                                                 14:{0:None,1:None,2:None,3:41,  4:33,  5:25,  6:16, 7:7, 8:42,9:43,10:44,11:45,  12:46,  13:47,  14:None,15:None,16:None},
                                                                 15:{0:None,1:None,2:None,3:None,4:34,  5:26,  6:17, 7:8, 8:48,9:49,10:50,11:51,  12:52,  13:None,14:None,15:None,16:None},
                                                                 16:{0:None,1:None,2:None,3:None,4:None,5:None,6:18, 7:9, 8:53,9:54,10:55,11:None,12:None,13:None,14:None,15:None,16:None}}
coreMaps['FULL']['QUARTER_ROTATIONAL'][157]['WITHOUT_REFLECTOR'] = {0 :{0:None, 1:None,  2:None,  3:None,  4:None,  5:None,  6:39,  7:38,  8: 8,  9:None,  10:None  , 11:None, 12:None, 13:None, 14:None},
                                                                     1 :{0:None, 1:None,  2:None,  3:None,  4:37,    5:36,    6:35,  7:34,  8: 7,  9:15,    10:22,     11:None, 12:None, 13:None, 14:None},
                                                                     2 :{0:None, 1:None,  2:None,  3:33,    4:32,    5:31,    6:30,  7:29,  8: 6,  9:14,    10:21,     11:28,   12:None, 13:None, 14:None},
                                                                     3 :{0:None, 1:None,  2:28,    3:27,    4:26,    5:25,    6:24,  7:23,  8: 5,  9:13,    10:20,     11:27,   12:33,   13:None, 14:None},
                                                                     4 :{0:None, 1:22,    2:21,    3:20,    4:19,    5:18,    6:17,  7:16,  8: 4,  9:12,    10:19,     11:26,   12:32,   13:37,   14:None},
                                                                     5 :{0:None, 1:15,    2:14,    3:13,    4:12,    5:11,    6:10,  7: 9,  8: 3,  9:11,    10:18,     11:25,   12:31,   13:36,   14:None},
                                                                     6 :{0: 8,   1: 7,    2: 6,    3: 5,    4: 4,    5: 3,    6:2,   7: 1,  8: 2,  9:10,    10:17,     11:24,   12:30,   13:35,   14:39},
                                                                     7 :{0:38,   1:34,    2:29,    3:23,    4:16,    5: 9,    6:1,   7: 0,  8: 1,  9: 9,    10:16,     11:23,   12:29,   13:34,   14:38},
                                                                     8 :{0:39,   1:35,    2:30,    3:24,    4:17,    5:10,    6:2,   7: 1,  8: 2,  9: 3,    10: 4,     11: 5,   12: 6,   13: 7,   14: 8},
                                                                     9 :{0:None, 1:36,    2:31,    3:25,    4:18,    5:11,    6:4,   7:3,   8:4,   9:5,     10:8,      11:12,   12:17,   13:22,   14:None},
                                                                     10:{0:None, 1:37,    2:32,    3:26,    4:19,    5:12,    6:4,   7:16,  8:17,  9:18,    10:19,     11:20,   12:21,   13:22,   14:None},
                                                                     11:{0:None, 1:None,  2:33,    3:27,    4:20,    5:13,    6:5,   7:23,  8:24,  9:25,    10:26,     11:27,   12:28,   13:None, 14:None},
                                                                     12:{0:None, 1:None,  2:None,  3:28,    4:21,    5:14,    6:6,   7:29,  8:30,  9:31,    10:32,     11:33,   12:None, 13:None, 14:None},
                                                                     13:{0:None, 1:None,  2:None,  3:None,  4:22,    5:15,    6:7,   7:34,  8:35,  9:36,    10:37,     11:None, 12:None, 13:None, 14:None},
                                                                     14:{0:None, 1:None,  2:None,  3:None,  4:None,  5:None,  6:8,   7:38,  8:39,  9:None,  10:None  , 11:None, 12:None, 13:None, 14:None}}
coreMaps['FULL']['QUARTER_ROTATIONAL'][193] = {}

coreMaps['FULL']['QUARTER_ROTATIONAL'][193]['WITH_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:None,5:78,6:77,7:76,8:75,9:74,10:75,11:76,12:77,13:78,14:None,15:None,16:None,17:None,18:None},
                                                                  1:{0:None,1:None,2:None,3:73,  4:72,  5:71,6:70,7:69,8:68,9:67,10:68,11:69,12:70,13:71,14:72,  15:73,  16:None,17:None,18:None},
                                                                  2:{0:None,1:None,2:66,  3:65,  4:64,  5:63,6:62,7:61,8:8, 9:59,10:8, 11:61,12:62,13:63,14:64,  15:65,  16:66,  17:None,18:None},
                                                                  3:{0:None,1:58,  2:57,  3:56,  4:55,  5:54,6:53,7:52,8:51,9:50,10:51,11:52,12:53,13:54,14:55,  15:56,  16:57,  17:58,  18:None},
                                                                  4:{0:None,1:49,  2:48,  3:47,  4:46,  5:45,6:44,7:43,8:42,9:41,10:42,11:43,12:44,13:45,14:46,  15:47,  16:48,  17:49,  18:None},
                                                                  5:{0:40,  1:39,  2:38,  3:37,  4:36,  5:35,6:34,7:33,8:32,9:31,10:32,11:33,12:34,13:35,14:36,  15:37,  16:38,  17:39,  18:40  },
                                                                  6:{0:30,  1:29,  2:28,  3:27,  4:26,  5:25,6:24,7:23,8:22,9:21,10:22,11:23,12:24,13:25,14:26,  15:27,  16:28,  17:29,  18:30  },
                                                                  7:{0:20,  1:19,  2:18,  3:17,  4:16,  5:15,6:14,7:13,8:12,9:11,10:12,11:13,12:14,13:15,14:16,  15:17,  16:18,  17:19,  18:20  },
                                                                  8:{0:10,  1:9 ,  2:8 ,  3:7,   4:6 ,  5:5 ,6:4 ,7:3, 8:2 ,9:1, 10:2 ,11:3 ,12:4 ,13:5 ,14:6 ,  15:7 ,  16:8 ,  17:9 ,  18:10  },
                                                                  9:{0:74,  1:67,  2:59,  3:50,  4:41,  5:31,6:21,7:11,8:1 ,9:0, 10:1 ,11:11,12:21,13:31,14:41,  15:50,  16:59,  17:67,  18:74  },
                                                                 10:{0:10,  1:9 ,  2:8 ,  3:51,  4:6 ,  5:5 ,6:4 ,7:3 ,8:2 ,9:1, 10:2 ,11:3 ,12:4 ,13:5 ,14:6 ,  15:7 ,  16:8 ,  17:9 ,  18:10  },
                                                                 11:{0:20,  1:19,  2:18,  3:52,  4:16,  5:15,6:14,7:13,8:12,9:11,10:12,11:13,12:14,13:15,14:16,  15:17,  16:18,  17:19,  18:20  },
                                                                 12:{0:30,  1:29,  2:28,  3:53,  4:26,  5:25,6:24,7:23,8:22,9:21,10:22,11:23,12:24,13:25,14:26,  15:27,  16:28,  17:29,  18:30  },
                                                                 13:{0:40,  1:39,  2:38,  3:54,  4:36,  5:35,6:34,7:33,8:32,9:31,10:32,11:33,12:34,13:35,14:36,  15:37,  16:38,  17:39,  18:40  },
                                                                 14:{0:None,1:49,  2:48,  3:55,  4:46,  5:45,6:44,7:43,8:42,9:41,10:42,11:43,12:44,13:45,14:46,  15:47,  16:48,  17:49,  18:None},
                                                                 15:{0:None,1:58,  2:57,  3:56,  4:55,  5:54,6:53,7:52,8:51,9:50,10:51,11:52,12:53,13:54,14:55,  15:56,  16:57,  17:58,  18:None},
                                                                 16:{0:None,1:None,2:66,  3:57,  4:64,  5:63,6:62,7:61,8:60,9:59,10:60,11:61,12:62,13:63,14:64,  15:65,  16:66,  17:None,18:None},
                                                                 17:{0:None,1:None,2:None,3:58,  4:72,  5:71,6:70,7:69,8:68,9:67,10:68,11:69,12:70,13:71,14:72,  15:73,  16:None,17:None,18:None},
                                                                 18:{0:None,1:None,2:None,3:None,4:None,5:78,6:77,7:76,8:75,9:74,10:75,11:76,12:77,13:78,14:None,15:None,16:None,17:None,18:None}}

coreMaps['FULL']['QUARTER_ROTATIONAL'][193]['WITHOUT_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:48,  5:47, 6:46, 7:45, 8:8,  9:16,  10:24, 11:None,12:None,13:None,14:None},
                                                                     1:{0:None,1:None,2:44,  3:43,  4:42,  5:41, 6:40, 7:39, 8:7,  9:15,  10:23, 11:31,  12:38,  13:None,14:None},
                                                                     2:{0:None,1:38,  2:37,  3:36,  4:35,  5:34, 6:33, 7:32, 8:6,  9:14,  10:22, 11:30,  12:37,  13:44,  14:None},
                                                                     3:{0:None,1:31,  2:30,  3:29,  4:28,  5:27, 6:26, 7:25, 8:5,  9:13,  10:21, 11:29,  12:36,  13:43,  14:None},
                                                                     4:{0:24,  1:23,  2:22,  3:21,  4:20,  5:19, 6:18, 7:17, 8:4,  9:12,  10:20, 11:28,  12:35,  13:42,  14:48},
                                                                     5:{0:16,  1:15,  2:14,  3:13,  4:12,  5:11, 6:10, 7:9,  8:3,  9:11,  10:19, 11:27,  12:34,  13:41,  14:47},
                                                                     6:{0:8,   1:7,   2:6,   3:5,   4:4,   5:3,  6:2,  7:1,  8:2,  9:10,  10:18, 11:26,  12:33,  13:40,  14:46},
                                                                     7:{0:45,  1:39,  2:32,  3:25,  4:17,  5:9,  6:1,  7:0,  8:1,  9:9,   10:17, 11:25,  12:32,  13:39,  14:45},
                                                                     8:{0:46,  1:40,  2:33,  3:26,  4:18,  5:10, 6:2,  7:1,  8:2,  9:3,   10:4,  11:5,   12:6,   13:7,   14:8},
                                                                     9:{0:47,  1:41,  2:34,  3:27,  4:19,  5:11, 6:3,  7:9,  8:10, 9:11,  10:12, 11:13,  12:14,  13:15,  14:16},
                                                                    10:{0:48,  1:42,  2:35,  3:28,  4:20,  5:12, 6:4,  7:17, 8:18, 9:19,  10:20, 11:21,  12:22,  13:23,  14:24},
                                                                    11:{0:None,1:43,  2:36,  3:29,  4:21,  5:13, 6:5,  7:25, 8:26, 9:27,  10:28, 11:29,  12:30,  13:31,  14:None},
                                                                    12:{0:None,1:44,  2:37,  3:30,  4:22,  5:14, 6:6,  7:32, 8:33, 9:34,  10:35, 11:36,  12:37,  13:38,  14:None},
                                                                    13:{0:None,1:None,2:38,  3:31,  4:23,  5:15, 6:7,  7:39, 8:40, 9:41,  10:42, 11:43,  12:44,  13:None,14:None},
                                                                    14:{0:None,1:None,2:None,3:None,4:24,  5:16, 6:8,  7:45, 8:46, 9:47,  10:48, 11:None,12:None,13:None,14:None}}
coreMaps['FULL']['QUARTER_ROTATIONAL'][241] = {}
coreMaps['FULL']['QUARTER_ROTATIONAL'][241]['WITH_REFLECTOR'] = {0:{ 0:None,1:None,2:None,3:None,4:None,5:78, 6:77, 7:76, 8:75, 9:74, 10:10, 11:20, 12:30, 13:40, 14:None,15:None,16:None,17:None,18:None},
                                                                  1:{ 0:None,1:None,2:None,3:73,  4:72,  5:71, 6:70, 7:69, 8:68, 9:67, 10: 9, 11:19, 12:29, 13:39, 14:49,  15:58,  16:None,17:None,18:None},
                                                                  2:{ 0:None,1:None,2:66,  3:65,  4:64,  5:63, 6:62, 7:61, 8:60, 9:59, 10: 8, 11:18, 12:28, 13:38, 14:48,  15:57,  16:66,  17:None,18:None},
                                                                  3:{ 0:None,1:58,  2:57,  3:56,  4:55,  5:54, 6:53, 7:52, 8:51, 9:50, 10: 7, 11:17, 12:27, 13:37, 14:47,  15:56,  16:65,  17:73,  18:None},
                                                                  4:{ 0:None,1:49,  2:48,  3:47,  4:46,  5:45, 6:44, 7:43, 8:42, 9:41, 10: 6, 11:16, 12:26, 13:36, 14:46,  15:55,  16:64,  17:72,  18:None},
                                                                  5:{ 0:40,  1:39,  2:38,  3:37,  4:36,  5:35, 6:34, 7:33, 8:32, 9:31, 10: 5, 11:15, 12:25, 13:35, 14:45,  15:54,  16:63,  17:71,  18:78},
                                                                  6:{ 0:30,  1:29,  2:28,  3:27,  4:26,  5:25, 6:24, 7:23, 8:22, 9:21, 10: 4, 11:14, 12:24, 13:34, 14:44,  15:53,  16:62,  17:70,  18:77},
                                                                  7:{ 0:20,  1:19,  2:18,  3:17,  4:16,  5:15, 6:14, 7:13, 8:12, 9:11, 10: 3, 11:13, 12:23, 13:33, 14:43,  15:52,  16:61,  17:69,  18:76},
                                                                  8:{ 0:10,  1: 9,  2: 8,  3: 7,  4: 6,  5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 2, 11:12, 12:22, 13:32, 14:42,  15:51,  16:60,  17:68,  18:75},
                                                                  9:{ 0:74,  1:67,  2:59,  3:50,  4:41,  5:31, 6:21, 7:11, 8: 1, 9: 0, 10: 1, 11:11, 12:21, 13:31, 14:41,  15:50,  16:59,  17:67,  18:74},
                                                                 10:{ 0:75,  1:68,  2:60,  3:51,  4:42,  5:32, 6:22, 7:12, 8: 2, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6,  15: 7,  16: 8,  17: 9,  18:10},
                                                                 11:{ 0:76,  1:69,  2:61,  3:52,  4:43,  5:33, 6:23, 7:13, 8: 3, 9:11, 10:12, 11:13, 12:14, 13:15, 14:16,  15:17,  16:18,  17:19,  18:20},
                                                                 12:{ 0:77,  1:70,  2:62,  3:53,  4:44,  5:34, 6:24, 7:14, 8: 4, 9:21, 10:22, 11:23, 12:24, 13:25, 14:26,  15:27,  16:28,  17:29,  18:30},
                                                                 13:{ 0:78,  1:71,  2:63,  3:54,  4:45,  5:35, 6:25, 7:15, 8: 5, 9:31, 10:32, 11:33, 12:34, 13:35, 14:36,  15:37,  16:38,  17:39,  18:40},
                                                                 14:{ 0:None,1:72,  2:64,  3:55,  4:46,  5:36, 6:26, 7:16, 8: 6, 9:41, 10:42, 11:43, 12:44, 13:45, 14:46,  15:47,  16:48,  17:49,  18:None},
                                                                 15:{ 0:None,1:73,  2:65,  3:56,  4:47,  5:37, 6:27, 7:17, 8: 7, 9:50, 10:51, 11:52, 12:53, 13:54, 14:55,  15:56,  16:57,  17:58,  18:None},
                                                                 16:{ 0:None,1:None,2:66,  3:57,  4:48,  5:38, 6:28, 7:18, 8: 8, 9:59, 10:60, 11:61, 12:62, 13:63, 14:64,  15:65,  16:66,  17:None,18:None},
                                                                 17:{ 0:None,1:None,2:None,3:58,  4:49,  5:39, 6:29, 7:19, 8: 9, 9:67, 10:68, 11:69, 12:70, 13:71, 14:72,  15:73,  16:None,17:None,18:None},
                                                                 18:{ 0:None,1:None,2:None,3:None,4:None,5:40, 6:30, 7:20, 8:10, 9:74, 10:75, 11:76, 12:77, 13:78, 14:None,15:None,16:None,17:None,18:None}}
coreMaps['FULL']['QUARTER_ROTATIONAL'][241]['WITHOUT_REFLECTOR'] = {0:{0:None,1:None,2:None,3:None,4:None,5:60, 6:59, 7:58, 8:57, 9: 9, 10:18, 11:27, 12:None, 13:None, 14:None, 15:None, 16:None},
                                                                     1:{0:None,1:None,2:None,3:56,  4:55,  5:54, 6:53, 7:52, 8:51, 9: 8, 10:17, 11:26, 12:35,   13:43,   14:None, 15:None, 16:None},
                                                                     2:{0:None,1:None,2:50,  3:49,  4:48,  5:47, 6:46, 7:45, 8:44, 9: 7, 10:16, 11:25, 12:34,   13:42,   14:50,   15:None, 16:None},
                                                                     3:{0:None,1:43,  2:42,  3:41,  4:40,  5:39, 6:38, 7:37, 8:36, 9: 6, 10:15, 11:24, 12:33,   13:41,   14:49,   15:56,   16:None},
                                                                     4:{0:None,1:35,  2:34,  3:33,  4:32,  5:31, 6:30, 7:29, 8:28, 9: 5, 10:14, 11:23, 12:32,   13:40,   14:48,   15:55,   16:None},
                                                                     5:{0:27,  1:26,  2:25,  3:24,  4:23,  5:22, 6:21, 7:20, 8:19, 9: 4, 10:13, 11:22, 12:31,   13:39,   14:47,   15:54,   16:60,  },
                                                                     6:{0:18,  1:17,  2:16,  3:15,  4:14,  5:13, 6:12, 7:11, 8:10, 9: 3, 10:12, 11:21, 12:30,   13:38,   14:46,   15:53,   16:59,  },
                                                                     7:{0: 9,  1: 8,  2: 7,  3: 6,  4: 5,  5: 4, 6: 3, 7: 2, 8: 1, 9: 2, 10:11, 11:20, 12:29,   13:37,   14:45,   15:52,   16:58,  },
                                                                     8:{0:57,  1:51,  2:44,  3:36,  4:28,  5:19, 6:10, 7: 1, 8: 0, 9: 1, 10:10, 11:19, 12:28,   13:36,   14:44,   15:51,   16:57,  },
                                                                     9:{0:58,  1:52,  2:45,  3:37,  4:29,  5:20, 6:11, 7: 2, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5,   13: 6,   14: 7,   15: 8,   16: 9,  },
                                                                    10:{0:59,  1:53,  2:46,  3:38,  4:30,  5:21, 6:12, 7: 3, 8:10, 9:11, 10:12, 11:13, 12:14,   13:15,   14:16,   15:17,   16:18,  },
                                                                    11:{0:60,  1:54,  2:47,  3:39,  4:31,  5:22, 6:13, 7: 4, 8:19, 9:20, 10:21, 11:22, 12:23,   13:24,   14:25,   15:26,   16:27,  },
                                                                    12:{0:None,1:55,  2:48,  3:40,  4:32,  5:23, 6:14, 7: 5, 8:28, 9:29, 10:30, 11:31, 12:32,   13:33,   14:34,   15:35,   16:None},
                                                                    13:{0:None,1:56,  2:49,  3:41,  4:33,  5:24, 6:15, 7: 6, 8:36, 9:37, 10:38, 11:39, 12:40,   13:41,   14:42,   15:43,   16:None},
                                                                    14:{0:None,1:None,2:50,  3:42,  4:34,  5:25, 6:16, 7: 7, 8:44, 9:45, 10:46, 11:47, 12:48,   13:49,   14:50,   15:None, 16:None},
                                                                    15:{0:None,1:None,2:None,3:43,  4:35,  5:26, 6:17, 7: 8, 8:51, 9:52, 10:53, 11:54, 12:55,   13:56,   14:None, 15:None, 16:None},
                                                                    16:{0:None,1:None,2:None,3:None,4:None,5:27, 6:18, 7: 9, 8:57, 9:58, 10:59, 11:60, 12:None, 13:None, 14:None, 15:None, 16:None}}
coreMaps['QUARTER'] = {}
coreMaps['QUARTER']['QUARTER'] = {}
coreMaps['QUARTER']['QUARTER'][157] = {}
coreMaps['QUARTER']['QUARTER'][157]['WITH_REFLECTOR'] = {8:{8:0, 9:1,  10:10,11:19,  12:27,  13:35,  14:42,  15:48,  16:53},
                                                          9:{8:1, 9:2,  10:3, 11:4,   12:5,   13:6,   14:7,   15:8,   16:9},
                                                         10:{8:10,9:11, 10:12,11:13,  12:14,  13:15,  14:16,  15:17,  16:18},
                                                         11:{8:19,9:20, 10:21,11:22,  12:23,  13:24,  14:25,  15:26,  16:None},
                                                         12:{8:27,9:28, 10:29,11:30,  12:31,  13:32,  14:33,  15:34,  16:None},
                                                         13:{8:35,9:36, 10:37,11:38,  12:39,  13:40,  14:41,  15:None,16:None},
                                                         14:{8:42,9:43, 10:44,11:45,  12:46,  13:47,  14:None,15:None,16:None},
                                                         15:{8:48,9:49, 10:50,11:51,  12:52,  13:None,14:None,15:None,16:None},
                                                         16:{8:53,9:54, 10:55,11:None,12:None,13:None,14:None,15:None,16:None}}
coreMaps['QUARTER']['QUARTER'][157]['WITHOUT_REFLECTOR'] = {7 :{7: 0,  8: 1,  9: 9,  10:16,  11:23, 12:29, 13:34, 14:38},
                                                             8 :{7: 1,  8: 2,  9: 3,  10: 4,  11: 5, 12: 6, 13: 7, 14: 8},
                                                             9 :{7: 9,  8:10,  9:11,  10:12,  11:13, 12:14, 13:15, 14:None},
                                                             10:{7:16,  8:17,  9:18,  10:19,  11:20, 12:21, 13:22, 14:None},
                                                             11:{7:23,  8:24,  9:25,  10:26,  11:27, 12:28,    13:None,14:None},
                                                             12:{7:29,  8:30,  9:31,  10:32,  11:33, 12:None,  13:None,14:None},
                                                             13:{7:34,  8:35,  9:36,  10:37,  11:None,12:None, 13:None,14:None},
                                                             14:{7:38,  8:39,  9:None,10:None,11:None,12:None, 13:None,14:None}}
coreMaps['QUARTER']['QUARTER'][193] = {}
coreMaps['QUARTER']['QUARTER'][193]['WITH_REFLECTOR'] ={8:{8:0, 9:1, 10:10,11:19,12:28,13:37,  14:43,  15:51,  16:58},       #Edited input count
                                                         9:{8:1, 9:2, 10:3, 11:4, 12:5, 13:6,   14:7,   15:8,   16:9},
                                                        10:{8:10,9:11,10:12,11:13,12:14,13:15,  14:16,  15:17,  16:18},
                                                        11:{8:19,9:20,10:21,11:22,12:23,13:24,  14:25,  15:26,  16:27},
                                                        12:{8:28,9:29,10:30,11:31,12:32,13:33,  14:34,  15:35,  16:36},
                                                        13:{8:37,9:38,10:39,11:38,12:39,13:40,  14:41,  15:42,  16:None},
                                                        14:{8:43,9:44,10:45,11:46,12:47,13:48,  14:49,  15:50,  16:None},
                                                        15:{8:51,9:52,10:53,11:54,12:55,13:56,  14:57,  15:None,16:None},
                                                        16:{8:58,9:59,10:60,11:61,12:62,13:None,14:None,15:None,16:None}}
coreMaps['QUARTER']['QUARTER'][193]['WITHOUT_REFLECTOR'] = {7 :{7:0,   8:1, 9:9,  10:17,11:25,  12:32,  13:38,  14:43},
                                                             8 :{7:1,   8:2, 9:3,  10:4, 11:5,   12:6,   13:7,   14:8},
                                                             9 :{7:9,   8:10,9:11, 10:12,11:13,  12:14,  13:15,  14:16},
                                                             10:{7:17,  8:18,9:19, 10:20,11:21,  12:22,  13:23,  14:24},
                                                             11:{7:25,  8:26,9:27, 10:28,11:29,  12:30,  13:31,  14:None},
                                                             12:{7:32,  8:33,9:34, 10:35,11:36,  12:37,  13:None,14:None},
                                                             13:{7:38,  8:39,9:40, 10:41,11:42,  12:None,13:None,14:None},
                                                             14:{7:43,  8:44,9:45, 10:46,11:None,12:None,13:None,14:None}}
coreMaps['QUARTER']['QUARTER'][241] = {}
coreMaps['QUARTER']['QUARTER'][241]['WITH_REFLECTOR'] = {9:{9: 0, 10: 1, 11:11, 12:21, 13:31, 14:41,  15:50,  16:59,  17:67,  18:74},
                                                         10:{9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6,  15: 7,  16: 8,  17: 9,  18:10},
                                                         11:{9:11, 10:12, 11:13, 12:14, 13:15, 14:16,  15:17,  16:18,  17:19,  18:20},
                                                         12:{9:21, 10:22, 11:23, 12:24, 13:25, 14:26,  15:27,  16:28,  17:29,  18:30},
                                                         13:{9:31, 10:32, 11:33, 12:34, 13:35, 14:36,  15:37,  16:38,  17:39,  18:40},
                                                         14:{9:41, 10:42, 11:43, 12:44, 13:45, 14:46,  15:47,  16:48,  17:49,  18:None},
                                                         15:{9:50, 10:51, 11:52, 12:53, 13:54, 14:55,  15:56,  16:57,  17:58,  18:None},
                                                         16:{9:59, 10:60, 11:61, 12:62, 13:63, 14:64,  15:65,  16:66,  17:None,18:None},
                                                         17:{9:67, 10:68, 11:69, 12:70, 13:71, 14:72,  15:73,  16:None,17:None,18:None},
                                                         18:{9:74, 10:75, 11:76, 12:77, 13:78, 14:None,15:None,16:None,17:None,18:None}}

coreMaps['QUARTER']['QUARTER'][241]['WITHOUT_REFLECTOR'] = {8:{8: 0, 9: 1, 10:10, 11:19, 12:28,   13:36,   14:44,   15:51,   16:57  },
                                                             9:{8: 1, 9: 2, 10: 3, 11: 4, 12: 5,   13: 6,   14: 7,   15: 8,   16: 9  },
                                                            10:{8:10, 9:11, 10:12, 11:13, 12:14,   13:15,   14:16,   15:17,   16:18  },
                                                            11:{8:19, 9:20, 10:21, 11:22, 12:23,   13:24,   14:25,   15:26,   16:27  },
                                                            12:{8:28, 9:29, 10:30, 11:31, 12:32,   13:33,   14:34,   15:35,   16:None},
                                                            13:{8:36, 9:37, 10:38, 11:39, 12:40,   13:41,   14:42,   15:43,   16:None},
                                                            14:{8:44, 9:45, 10:46, 11:47, 12:48,   13:49,   14:50,   15:None, 16:None},
                                                            15:{8:51, 9:52, 10:53, 11:54, 12:55,   13:56,   14:None, 15:None, 16:None},
                                                            16:{8:57, 9:58, 10:59, 11:60, 12:None, 13:None, 14:None, 15:None, 16:None}}
coreMaps['QUARTER']['OCTANT'] = {}
coreMaps['QUARTER']['OCTANT'][157] = {}
coreMaps['QUARTER']['OCTANT'][157]['WITH_REFLECTOR'] = {8:{8:0, 9:1, 10:3, 11:6,   12:10,  13:15,  14:21  , 15:27,  16:32},
                                                         9:{8:1, 9:2, 10:4, 11:7,   12:11,  13:16,  14:22  , 15:28,  16:33},
                                                        10:{8:3, 9:4, 10:5, 11:8,   12:12,  13:17,  14:23  , 15:29,  16:34},
                                                        11:{8:6, 9:7, 10:8, 11:9,   12:13,  13:18,  14:24  , 15:30,  16:None},
                                                        12:{8:10,9:11,10:12,11:13,  12:14,  13:19,  14:25  , 15:31,  16:None},
                                                        13:{8:15,9:16,10:17,11:18,  12:19,  13:20,  14:26  , 15:None,16:None},
                                                        14:{8:21,9:22,10:23,11:24,  12:25,  13:26,  14:None, 15:None,16:None},
                                                        15:{8:27,9:28,10:29,11:30,  12:31,  13:None,14:None, 15:None,16:None},
                                                        16:{8:32,9:33,10:34,11:None,12:None,13:None,14:None, 15:None,16:None}}
coreMaps['QUARTER']['OCTANT'][157]['WITHOUT_REFLECTOR'] = {7:{7:0,   8:1,  9:3,    10:6,    11:10,   12:15,   13:20,   14:24},
                                                            8:{7:1,   8:2,  9:4,    10:7,    11:11,   12:16,   13:21,   14:25},
                                                            9:{7:3,   8:4,  9:5,    10:8,    11:12,   12:17,   13:22,   14:None},
                                                            10:{7:6, 8:7,  9:8,    10:9,    11:13,   12:18,   13:23,   14:None},
                                                            11:{7:10, 8:11, 9:12,   10:13,   11:14,   12:19,   13:None, 14:None},
                                                            12:{7:15, 8:16, 9:17,   10:18,   11:19,   12:None, 13:None, 14:None},
                                                            13:{7:20, 8:21, 9:22,   10:23,   11:None, 12:None, 13:None, 14:None},
                                                            14:{7:24, 8:25, 9:None, 10:None, 11:None, 12:None, 13:None, 14:None}}
coreMaps['QUARTER']['OCTANT'][193] = {}
coreMaps['QUARTER']['OCTANT'][193]['WITH_REFLECTOR'] = {8:{8:0, 9:1, 10:3, 11:6, 12:10,13:15,  14:21,  15:28,  16:35},       #Edited input count
                                                         9:{8:1, 9:2, 10:4, 11:7, 12:11,13:16,  14:22,  15:29,  16:36},
                                                        10:{8:3, 9:4, 10:5, 11:8, 12:12,13:17,  14:23,  15:30,  16:37},
                                                        11:{8:6, 9:7, 10:8, 11:9, 12:13,13:18,  14:24,  15:31,  16:38},
                                                        12:{8:10,9:11,10:12,11:13,12:14,13:19,  14:25,  15:32,  16:39},
                                                        13:{8:15,9:16,10:17,11:18,12:19,13:20,  14:26,  15:33,  16:None},
                                                        14:{8:21,9:22,10:23,11:24,12:25,13:26,  14:27,  15:34,  16:None},
                                                        15:{8:28,9:29,10:30,11:31,12:32,13:33,  14:34,  15:None,16:None},
                                                        16:{8:35,9:36,10:37,11:38,12:39,13:None,14:None,15:None,16:None}}
coreMaps['QUARTER']['OCTANT'][193]['WITHOUT_REFLECTOR'] = {7:{7:0, 8:1, 9:3,  10:6, 11:10,  12:15,  13:21,  14:26},
                                                            8:{7:1, 8:2, 9:4,  10:7, 11:11,  12:16,  13:22,  14:27},
                                                            9:{7:3, 8:4, 9:5,  10:8, 11:12,  12:17,  13:23,  14:28},
                                                           10:{7:6, 8:7, 9:8,  10:9, 11:13,  12:18,  13:24,  14:29},
                                                           11:{7:10,8:11,9:12, 10:13,11:14,  12:19,  13:25,  14:None},
                                                           12:{7:15,8:16,9:17, 10:18,11:19,  12:20,  13:None,14:None},
                                                           13:{7:21,8:22,9:23, 10:24,11:25,  12:None,13:None,14:None},
                                                           14:{7:26,8:27,9:28, 10:29,11:None,12:None,13:None,14:None}}
coreMaps['QUARTER']['OCTANT'][241] = {}
coreMaps['QUARTER']['OCTANT'][241]['WITH_REFLECTOR'] = {9:{9: 0, 10: 1, 11: 3, 12: 6, 13:10, 14:15,  15:21,  16:28,  17:36,  18:43},
                                                        10:{9: 1, 10: 2, 11: 4, 12: 7, 13:11, 14:16,  15:22,  16:29,  17:37,  18:44},
                                                        11:{9: 3, 10: 4, 11: 5, 12: 8, 13:12, 14:17,  15:23,  16:30,  17:38,  18:45},
                                                        12:{9: 6, 10: 7, 11: 8, 12: 9, 13:13, 14:18,  15:24,  16:31,  17:39,  18:46},
                                                        13:{9:10, 10:11, 11:12, 12:13, 13:14, 14:19,  15:25,  16:32,  17:40,  18:47},
                                                        14:{9:15, 10:16, 11:17, 12:18, 13:19, 14:20,  15:26,  16:33,  17:41,  18:None},
                                                        15:{9:21, 10:22, 11:23, 12:24, 13:25, 14:26,  15:27,  16:34,  17:42,  18:None},
                                                        16:{9:28, 10:29, 11:30, 12:31, 13:32, 14:33,  15:34,  16:35,  17:None,18:None},
                                                        17:{9:36, 10:37, 11:38, 12:39, 13:40, 14:41,  15:42,  16:None,17:None,18:None},
                                                        18:{9:43, 10:44, 11:45, 12:46, 13:47, 14:None,15:None,16:None,17:None,18:None}}
coreMaps['QUARTER']['OCTANT'][241]['WITHOUT_REFLECTOR'] = {8:{8:0,  9:1,  10:3,  11:6,  12:10,  13:15,  14:21,  15:28,  16:34  },
                                                            9:{8:1,  9:2,  10:4,  11:7,  12:11,  13:16,  14:22,  15:29,  16:35  },
                                                           10:{8:3,  9:4,  10:5,  11:8,  12:12,  13:17,  14:23,  15:30,  16:36  },
                                                           11:{8:6,  9:7,  10:8,  11:9,  12:13,  13:18,  14:24,  15:31,  16:37  },
                                                           12:{8:10, 9:11, 10:12, 11:13, 12:14,  13:19,  14:25,  15:32,  16:None},
                                                           13:{8:15, 9:16, 10:17, 11:18, 12:19,  13:20,  14:26,  15:33,  16:None},
                                                           14:{8:21, 9:22, 10:23, 11:24, 12:25,  13:26,  14:27,  15:None,16:None},
                                                           15:{8:28, 9:29, 10:30, 11:31, 12:32,  13:33,  14:None,15:None,16:None},
                                                           16:{8:34, 9:35, 10:36, 11:37, 12:None,13:None,14:None,15:None,16:None}}
