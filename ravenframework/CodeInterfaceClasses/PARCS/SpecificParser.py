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
Created on Aug. 10th, 2022
@author: khnguy22
comment: Specific parser for PRACS interface
"""
import os
from xml.etree import ElementTree as ET

class DataParser():
  """
    Parse the data in RAVEN input in to PARCS input
    and be saved for the perturbed input generator in Perturbed parser
    Note: this data parser is only called once
  """
  def __init__(self, inputFile):
    """
      Constructor.
      @ In, inputFile, string, xml PARCS parameters file
      @ Out, None
    """
    self.inputFile = inputFile
    self.getParameters()

  def getParameters(self):
    """
      Get required parameters from xml file for generating
      Constructor.
      @ In, None
      @ Out, None
    """
    fullFile = os.path.join(self.inputFile)
    dorm = ET.parse(fullFile)
    root = dorm.getroot()
    self.THFlag = root.find('THFlag').text.strip()
    self.power = root.find('power').text.strip()
    self.initialBoron = root.find('initialBoron').text.strip()
    self.coreType = root.find('coretype').text.strip()
    self.xsDir = root.find('XSdir').text.strip()
    self.depDir = root.find('Depdir').text.strip()
    self.depHistory = root.find('DepHistory').text.strip()
    self.NFA = root.find('NFA').text.strip()
    self.NAxial = root.find('NAxial').text.strip()
    self.geometry = root.find('Geometry').text.strip()
    self.faPitch = root.find('FA_Pitch').text.strip()
    self.faPower = root.find('FA_Power').text.strip()
    self.gridX = root.find('grid_x').text.strip()
    self.gridY = root.find('grid_y').text.strip()
    self.gridZ = root.find('grid_z').text.strip()
    self.neutmeshX = root.find('neutmesh_x').text.strip()
    self.neutmeshY = root.find('neutmesh_y').text.strip()
    self.BC = root.find('BC').text.strip()
    self.faDict = []
    for fa in root.iter('FA'):
      self.faDict.append(fa.attrib)
    self.xsDict =[]
    for xs in root.iter('XS'):
      self.xsDict.append(xs.attrib)

class PerturbedPaser():
  """
    Parse value in the perturbed xml file replaces the nominal values by the perturbed values.
  """
  def __init__(self, inputFile, workingDir, inputName, perturbDict):
    """
    Constructor.
      @ In, inputFile, string, xml PARCS varibles that will be perturbed file
      @ In, workingDir, string, absolute path to the working directory
      @ In, inputName, string, inputname for PARCS input file
      @ In, perturbDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.inputFile = inputFile #original perturb input file
    self.perturbDict = perturbDict #get from RAVEN
    self.workingDir = workingDir
    self.inputName = inputName
    # get perturbed value and create new xml file
    self.replaceOldFile()

  def replaceOldFile(self):
    """
      Replace orignal xml file with perturbed variables
      @ In, perturbed variables dictionary
      @ workingDir, absolute path to the working directory
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
    #sorting
    perturbedValSorted =  [val for _,val in sorted(zip(perturbedID,perturbedVal))]
    perturbedVal =  perturbedValSorted
    perturbedID = sorted(perturbedID)
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

  def generatePARCSInput(self, parameter):
    """
    Generate new input for PARCS to run
    @ In, parameter, DataParser, data in DataParser class
    @ Out, None
    """
    with open(f"{self.workingDir}/{self.inputName}",'w') as file_:
      ## write initial lines
      file_.write(f"!****************************************************************************** \n")
      file_.write(f"CASEID {self.inputName}              OECD NEA MSLB  \n")
      file_.write(f"!****************************************************************************** \n")
      file_.write(f"CNTL \n")
      file_.write(f"      TH_FDBK   {parameter.THFlag}  \n")
      file_.write(f"      CORE_POWER {parameter.power}  \n")
      file_.write(f"      CORE_TYPE  {parameter.coreType}  \n")
      file_.write(f"      PPM        {parameter.initialBoron}  \n")
      file_.write(f"      DEPLETION  T  1.0E-3 T \n")
      file_.write(f"      TREE_XS    T  16 T  T  F  F  T  F  T  F  T  F  F  T  T  F  \n")
      file_.write(f"      BANK_POS   100 100 100 100 100 100  0 46.2 0 \n")
      file_.write(f"      XE_SM      1 1 \n")
      file_.write(f"      SEARCH     ppm \n")
      file_.write(f"      XS_EXTRAP  1.0 0.3 0.8 0.2 \n")
      file_.write(f"      PIN_POWER  T   \n")
      file_.write(f"      PRINT_OPT T F T T F T F F F T  F  F  F  F  F   \n")
      file_.write(f"PARAM \n")
      file_.write(f"      LSOLVER  1 1 20 \n")
      file_.write(f"      NODAL_KERN     NEMMG    ! FMFD    ! FDM \n")
      file_.write(f"      CMFD     2 \n")
      file_.write(f"      DECUSP   2 \n")
      file_.write(f"      INIT_GUESS 0 \n")
      file_.write(f"      conv_ss   1.e-6 5.e-5 1.e-3 0.001 !epseig,epsl2,epslinf,epstf \n")
      file_.write(f"      eps_erf   0.010 \n")
      file_.write(f"      eps_anm   0.000001 \n")
      file_.write(f"      nlupd_ss  5 5 1 \n")
      file_.write(f"GEOM \n")
      file_.write(f"      geo_dim {parameter.NFA} {parameter.NFA} {parameter.NAxial} 1 1 \n") # full core geomerty
      file_.write(f"      Rad_Conf                        !! \n")
      loadingPattern = getcoremap(parameter,[int(child.attrib['FAid']) for child in self.data], parameter.geometry)
      file_.write(loadingPattern)
      file_.write("\n")
      file_.write(f"      grid_x      {parameter.gridX} \n")
      file_.write(f"      neutmesh_x  {parameter.neutmeshX} \n")
      file_.write(f"      grid_y      {parameter.gridY} \n")
      file_.write(f"      neutmesh_y  {parameter.neutmeshY} \n")
      file_.write(f"      grid_z      {parameter.gridZ} \n")
      file_.write(f"      Boun_cond   {parameter.BC} \n")

      for fa in parameter.faDict:
        if fa['name'].lower() != 'none' or float(fa['FAid'])>=0:
          file_.write(f"      assy_type   {fa['type']}   {fa['structure']} \n")
      file_.write("\n")
      ##create pin calculation map
      pinMap = loadingPattern
      for fa in parameter.faDict:
        if fa['name'].lower() == 'none' :
          pinMap = pinMap.replace(fa['type'],"  ")
        elif fa['name'].lower() == 'ref' :
          pinMap = pinMap.replace(fa['type']," 0")
      file_.write("\n")
      file_.write(f"     pincal_loc \n")
      file_.write(f"                        \n")
      file_.write(pinMap)
      file_.write("\n")
      file_.write("\n")
      file_.write(f"TH \n")
      file_.write(f"      unif_th         0.7  600.0  300.0 \n")
      file_.write(f"FDBK \n")
      file_.write(f"      fa_powpit       {parameter.faPower}   {parameter.faPitch} \n")
      file_.write("\n")
      file_.write(f"DEPL \n")
      file_.write(f"      TIME_STP  {parameter.depHistory}  \n")
      file_.write(f"      INP_HST   '../../{parameter.depDir}/boc_exp_fc.dep' -2 1 \n")
      for xs in parameter.xsDict:
        file_.write(f"      PMAXS_F   {xs['id']} '../../{parameter.xsDir}/{xs['name']}'                 {xs['id']}   \n")
      file_.write(f".  \n")


# Outside functions
def findType(faID,faDict):
  """
    Get type of FA ID
    @ In, faID, str, the FA ID
    @ In, faDict, dict,
    @ Out, faType, list, list of FA types
  """
  faType = [id['type'] for id in faDict if id['FAid']==str(faID)][0]
  return faType

def getcoremap(parameter, faID, geometrykey):
  """
    Genrate Loading Pattern
    @ In, parameter, DataParser class, include all parameter information
    @ In, faID, sorted list, geometry key for full or quarter core
    @ Out, loadingPattern, str, Loading Pattern
  """
  faDict = parameter.faDict
  maxType = max([id['type'] for id in faDict])
  numberSpaces = len(str(maxType)) + 2
  emptyMap=[]
  if geometrykey.lower()=='full':
    rows, cols=17,17
    xStart, yStart = 9, 9
  elif geometrykey.lower() =='quarter':
    rows, cols=9,9
    xStart, yStart = 1, 1
  else:
    raise ValueError("No available geometry key. Only full or quarter core is supported")
  for i in range(rows):
      col = []
      for j in range(cols):
          col.append(0)
      emptyMap.append(col)
  idx_ = 0
  val = faID[idx_]
  val = findType(val,faDict)
  for x in range(xStart,xStart+9): # 17x17 core
    for y in range (yStart,x+1):
      val = faID[idx_]
      val = findType(val,faDict)
      if geometrykey.lower()=='full':
        outCoordianate = getIndexFull(x,y,xStart,yStart)
      else:
        outCoordianate = getIndexQuater(x,y)
      for i in outCoordianate:
        emptyMap[i[0]-1][i[1]-1]=val
      idx_ = idx_+1
  loadingPattern = "      "
  for i in emptyMap:
      for j in i:
          value=j
          str_ = f"{value}"
          loadingPattern += f"{str_.rjust(numberSpaces)}"
      loadingPattern += "\n"
      loadingPattern += "      "
  return loadingPattern

def getIndexFull(x, y, x0, y0):
  """
    Get the index of symetric element in a 1/8 th symmetric core map
    to a full core map
    @ In, x, float, x coordinate of the element
    @ In, y, float, y coordinate of the element
    @ In, x0, float, x coordinate of the center element
    @ In, y0, float, y coordinate of the center element
    @ Out, outarray, list, list of indices [(x,y)]
  """
  deltaX = x-x0
  deltaY = y-y0
  temparray = []
  temparray.append([x0+deltaX,y0+deltaY])
  temparray.append([x0+deltaX,y0-deltaY])
  temparray.append([x0-deltaX,y0+deltaY])
  temparray.append([x0-deltaX,y0-deltaY])
  temparray.append([x0+deltaY,y0+deltaX])
  temparray.append([x0+deltaY,y0-deltaX])
  temparray.append([x0-deltaY,y0+deltaX])
  temparray.append([x0-deltaY,y0-deltaX])

  #remove duplicate
  outarray = []
  [outarray.append(i) for i in temparray if i not in outarray]
  return outarray

def getIndexQuater(x, y):
    """
      Get the index of symetric element in a 1/8 th symmetric core map
      to quarter core
      @ In, x, float, x coordinate of the element
      @ In, y, float, y coordinate of the element
      @ Out, outarray, list, list of indices [(x,y)]
    """
    temparray = []
    temparray.append([x,y])
    temparray.append([y,x])
    #remove duplicate
    outarray = []
    [outarray.append(i) for i in temparray if i not in outarray]
    return outarray
