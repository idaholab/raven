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
Created on June 04, 2022
@author: khnguy22 NCSU, luquj NCSU

comments: Interface for SIMULATE3 loading pattern optimzation
"""
import os
import numpy

class SimulateData:
  """
  Class that parses output of SIMULATE3 for a multiple run
  Partially copied from MOF work
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, string or dict, file name to be parsed, read one file at a time?
      @ Out, None
    """
    self.data = {} # dictionary to store the data from model output file
    self.lines = open(os.path.abspath(os.path.expanduser(filen)),"r").readlines() # raw data from model output
    # retrieve data
    self.data['axial_mesh'] = self.axialMeshExtractor()
    self.data['keff'] = self.coreKeffEOC()
    self.data["MaxFDH"] = self.maxFDH()
    self.data["kinf"] = self.kinfEOC()
    self.data["max_boron"] = self.boronEOC()
    self.data["cycle_length"] = self.EOCEFPD()
    self.data["pin_peaking"] = self.pinPeaking()
    self.data["exposure"] = self.burnupEOC()
    self.data["neutron_leakage"] = self.neutron_leakage()
    # self.data["assembly_power"] = self.assemblyPeakingFactors()
    # self.data["fuel_cost"] = self.fuel_cost()

    # this is a dummy variable for demonstration with MOF
    # check if something has been found
    if all(v is None for v in self.data.values()):
      raise IOError("No readable outputs have been found!")
#------------------------------------------------------------------------------------------
  #function to retrivedata
  def getPin(self):
    """
      Retrive total number of pins
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                            {'info_ids':list(of ids of data),
                              'values': list}
    """
    outputDict = None
    for line in self.lines:
      if line.find('Assembly Core Maps . . . .')>=0:
        temp = line.strip().split('(')
        temp = temp[1].split(',')[0]
        break
    outputDict = {'info_ids':['pin_number'], 'values': [int(temp)] }
    return outputDict

  def axialMeshExtractor(self):
    """
      Extracts the axial mesh used in the SIMULATE output file.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                            {'info_ids':list(of ids of data),
                              'values': list}
    """
    outputDict = None
    reverseAxialPositions = [] #SIMULATE reversed lists axial meshes from bottom to top
    searchingHeights = False
    for line in self.lines:
      if "** Studsvik CMS Steady-State 3-D Reactor Simulator **" in line:
        searchingHeights = False
      if "Grid Location Information" in line:
        searchingHeights = False
        break
      if searchingHeights:
        line = line.replace("-","")
        elems = line.strip().split()
        if elems:
          reverseAxialPositions.append(float(elems[-1]))
      if "Axial Nodal Boundaries (cm)" in line:
          searchingHeights = True
    #The bot/top axial node in the reflectors are not considered
    reverseAxialPositions.pop(0)
    reverseAxialPositions.pop(-1)

    forwardAxialPositions = []
    for position in reverseAxialPositions:
      forwardAxialPositions.insert(0,position)

    outputDict = {'info_ids':['no_axial_node'],
                  'values': [len(forwardAxialPositions)] }

    return outputDict

  def getCoreWidth(self):
    """
      Retrive core width
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                            {'info_ids':list(of ids of data),
                              'values': list}
    """
    outputDict = None
    for line in self.lines:
      if line.find("'DIM.PWR'")>=0:
        temp = line.strip().split(' ')
        temp = temp[1].split('/')[0]
        break
    outputDict = {'info_ids':['core_width'], 'values': [int(temp)] }
    return outputDict


  def coreKeffEOC(self):
    """
      Extracts the core K-effective value from the provided simulate file lines.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                            {'info_ids':list(of ids of data),
                              'values': list}
    """
    keffList = []
    outputDict = None
    for line in self.lines:
      if "K-effective . . . . . . . . . . . . ." in line:
        elems = line.strip().split()
        keffList.append(float(elems[-1]))
    if not keffList:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['eoc_keff'], 'values': [keffList[-1]] }
    return outputDict


  # def assemblyPeakingFactors(self):
  #   """
  #     Extracts the assembly radial power peaking factors as a dictionary
  #     with the depletion step in GWD/MTU as the dictionary keys.
  #     @ In, None
  #     @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
  #                           {'info_ids':list(of ids of data),
  #                             'values': list}
  #   """
  #   radialPowerDictionary = {}
  #   searching_ = False
  #   outputDict = None
  #   for line in self.lines:
  #     if "Case" in line and "GWd/MT" in line:
  #       elems = line.strip().split()
  #       depl = elems[-2]
  #       if depl in radialPowerDictionary:
  #         pass
  #       else:
  #         radialPowerDictionary[depl] = {}
  #     if "**   H-     G-     F-     E-     D-     C-     B-     A-     **" in line:
  #       searching_ = False

  #     if searching_:
  #       elems = line.strip().split()
  #       if elems[0] == "**":
  #         posList = elems[1:-1]
  #       else:
  #         radialPowerDictionary[depl][elems[0]] = {}
  #         for i,el in enumerate(elems[1:-1]):
  #           radialPowerDictionary[depl][elems[0]][posList[i]] = float(el)

  #     if "PRI.STA 2RPF  - Assembly 2D Ave RPF - Relative Power Fraction" in line:
  #       searching_ = True

  #   if not radialPowerDictionary:
  #     return ValueError("No values returned. Check Simulate File executed correctly")
  #   else:
  #     maxPeaking = 0.0
  #     for depl in radialPowerDictionary:
  #       for row in radialPowerDictionary[depl]:
  #         for col in radialPowerDictionary[depl][row]:
  #           maxPeaking = max(radialPowerDictionary[depl][row][col],maxPeaking)
  #     outputDict = {'info_ids':['FA_peaking'], 'values': [maxPeaking] }

  #   return outputDict

  def EOCEFPD(self):
    """
      Returns maximum of EFPD values for cycle exposure in the simulate
      file.

      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    list_ = []
    outputDict = None
    for line in self.lines:
      if "Cycle Exp." in line:
        if "EFPD" in line:
          elems = line.strip().split()
          spot = elems.index('EFPD')
          list_.append(float(elems[spot-1]))

    if not list_:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['MaxEFPD'], 'values': [list_[-1]] }

    return outputDict

  def maxFDH(self):
    """
      Returns maximum of F-delta-H values for each cycle exposure in the simulate
      file.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    list_ = []
    outputDict = None
    for line in self.lines:
      if "F-delta-H" in line:
        elems = line.strip().split()
        spot = elems.index('F-delta-H')
        list_.append(float(elems[spot+1]))

    if not list_:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['MaxFDH'], 'values': [max(list_)] }

    return outputDict

  def pinPeaking(self):
    """
      Returns maximum value of pin peaking values, Fq, for each cycle exposure in the simulate
      file.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    outputDict = None
    list_ = []
    for line in self.lines:
      if "Max-3PIN" in line:
        elems = line.strip().split()
        spot = elems.index('Max-3PIN')
        list_.append(float(elems[spot+1]))

    print(f"This is Fq={max(list_)}")

    if not list_:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['pin_peaking'], 'values': [max(list_)] }

    return outputDict

  def boronEOC(self):
    """
      Returns EOC and max boron values in PPM at each depletion step.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    boronList = []
    outputDict = None
    for line in self.lines:
      if "Boron Conc." in line and "ppm" in line:
        elems = line.strip().split()
        spot = elems.index('ppm')
        boronList.append(float(elems[spot-1]))

    if not boronList:
      return ValueError("NO values returned. Check SIMULATE file executed correctly")
    else:
      outputDict = {'info_ids':['eoc_boron', 'max_boron'],
                    'values': [boronList[-1], max(boronList)] }

    return outputDict

  def kinfEOC(self):
    """
      Returns a list of kinf values from Simulate3.
      Only work for PWR
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    kinfList = []
    searchingForKinf = False
    outputDict = None
    for line in self.lines:
      elems = line.strip().split()
      if not elems:
        pass
      else:
        if searchingForKinf:
          if elems[0] == '1':
            kinfList.append(float(elems[1]))
            searchingForKinf = False
        if "PRI.STA 2KIN  - Assembly 2D Ave KINF - K-infinity" in line:
          searchingForKinf = True

    if not kinfList:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['eoc_kinf'], 'values': [ kinfList[-1]] }

    return outputDict

  def relativePower(self):
    """
      Extracts the Relative Core Power from the provided simulate file lines.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    relativePowers = []
    outputDict = None
    for line in self.lines:
      if "Relative Power. . . . . . .PERCTP" in line:
        p1 = line.index("PERCTP")
        p2 = line.index("%")
        searchSpace = line[p1:p2]
        searchSpace = searchSpace.replace("PERCTP","")
        relativePowers.append(float(searchSpace))

    if not relativePowers:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['relative power'], 'values': [relativePowers] }

    return outputDict

  def relativeFlow(self):
    """
      Extracts the Relative Core Flow rate from the provided simulate file lines.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    relativeFlows = []
    outputDict = None
    for line in self.lines:
      if "Relative Flow . . . . . .  PERCWT" in line:
        p1 = line.index("PERCWT")
        p2 = line.index("%")
        searchSpace = line[p1:p2]
        searchSpace = searchSpace.replace("PERCWT","")
        relativeFlows.append(float(searchSpace))

    if not relativeFlows:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['relative flow'], 'values': [relativeFlows] }

    return outputDict

  def thermalPower(self):
    """
      Extracts the operating thermal power in MW from the provided simulate file lines.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    powers = []
    outputDict = None
    for line in self.lines:
      if "Thermal Power . . . . . . . . CTP" in line:
        elems = line.strip().split()
        spot = elems.index('MWt')
        powers.append(float(elems[spot-1]))

    if not powers:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['thermal power'], 'values': [powers] }

    return outputDict

  def coreFlow(self):
    """
      Returns the core coolant flow in Mlb/hr from the provided simulate file lines.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    flows = []
    outputDict = None
    for line in self.lines:
      if "Core Flow . . . . . . . . . . CWT" in line:
        elems = line.strip().split()
        spot = elems.index("Mlb/hr")
        flows.append(float(elems[spot-1]))

    if not flows:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['core flow'], 'values': [flows] }

    return outputDict

  def inletTemperatures(self):
    """
      Returns the core inlet temperatures in degrees Fahrenheit from the
      provided simulate file lines.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    temperatures = []
    outputDict = None
    for line in self.lines:
      if "Inlet . . . .TINLET" in line:
        p1 = line.index("K")
        p2 = line.index("F")
        searchSpace = line[p1:p2]
        searchSpace = searchSpace.replace("K","")
        temperatures.append(float(searchSpace))

    if not temperatures:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['inlet temperatures'], 'values': [temperatures] }

    return outputDict

  def pressure(self):
    """
      Returns the core exit pressure in PSIA.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    pressure = []
    outputDict = None
    for line in self.lines:
      if "Core Exit Pressure  . . . . .  PR" in line:
        p1 = line.index("bar")
        p2 = line.index("PSIA")
        searchSpace = line[p1:p2]
        searchSpace = searchSpace.replace("bar","")
        pressure.append(float(searchSpace))

    if not pressure:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['pressure'], 'values': [pressure] }

    return outputDict

  def burnupEOC(self):
    """
      Extracts the cycle burnups at a each state point within the depletion.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    burnups = []
    for line in self.lines:
      if "Cycle Exp." in line:
        elems = line.strip().split()
        spot = elems.index('GWd/MT')
        burnups.append(float(elems[spot-1]))
    if not burnups:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['exposure'], 'values': [burnups[-1]] }

    return outputDict

  def fuel_cost(self):
    """
      Extracts the fuel types used in the core map and calculates the fuel cost based on a front end approach. 
      This function applies only to quarter core symmetries.
      @ In, Non
      @ Out, outputDict, dict, the dictionary containing the rea data (None if non found)
                            {'info_ids': list(of ids of data),
                            'values': list}
    """
    outputDict = None
    # First, we need to parse the core map from the output file.
    # NOTE: Given that a run is not needed to know the Loading Pattern, this function could be on the input side. 
    FA_list = []
    for line in self.lines:
      if "'FUE.TYP'" in line:
        p1 = line.index(",")
        p2 = line.index("/")
        search_space = line[p1:p2]
        search_space = search_space.replace(",","")
        temp = search_space.split()
        for i in temp:
          FA_list.append(float(i))
    FA_types = list(set(FA_list))
    quartcore_size = len(temp)

    # We separate the core map depending on how many times their elements are counted in the symmetry:  
    # FA_list_A counted once, as it is the center of the core.
    # FA_list_B counted twice, as they are are the centerlines.
    # FA_list_C counted four times, as they are are the rest of fuel assemblies.
    FA_list_A = FA_list[0]
    FA_list_B = FA_list[1:quartcore_size] + FA_list[quartcore_size:quartcore_size*(quartcore_size-1)+1:quartcore_size]
    FA_list_C = []
    for i in range(quartcore_size-1):
      FA_list_C.append(FA_list[(i+1)*quartcore_size + 1: (i+2)*quartcore_size])
    FA_list_C = [item for sublist in FA_list_C for item in sublist] # To flatten FA_list_C
    # Now we proceed to count how many fuel types of each type are there in our core.
    FA_count_A = [float(fa == FA_list_A) for fa in FA_types]
    FA_count_B = [float(FA_list_B.count(fa)*2) for fa in FA_types]
    FA_count_C = [float(FA_list_C.count(fa)*4) for fa in FA_types]
    FA_count = [FA_count_A[j] + FA_count_B[j] + FA_count_C[j] for j in range(len(FA_types))]
    # And create a dictionary with all the fuel types count.
    FA_types_dict = {int(FA_types[i]):FA_count[i] for i in range(len(FA_types))}
    
    # Dictionary with the unit cost for each FA type.

    # FA type 0 = empty         -> M$ 0.0
    # FA type 1 = reflector     -> M$ 0.0
    # FA type 2 = 2.00 wt%      -> M$ 2.69520839
    # FA type 3 = 2.50 wt%      -> M$ 3.24678409
    # FA type 4 = 2.50 wt% + Gd -> M$ 3.24678409
    # FA type 5 = 3.20 wt%      -> M$ 4.03739539
    # FA type 6 = 3.20 wt% + Gd -> M$ 4.03739539
    # The cost of burnable poison is not being considered.
    
    cost_dict = {
      0: 0,
      1: 0,
      2: 2.69520839,
      3: 3.24678409,
      4: 3.24678409,
      5: 4.03739539,
      6: 4.03739539
    }

    fuel_cost = 0
    for fuel_type, fuel_count in FA_types_dict.items():
      fuel_cost += fuel_count * cost_dict[fuel_type]

    if not fuel_cost:
      return ValueError("No values returned. Check Simulate file executed correctly.")
    else:
      outputDict = {'info_ids':['fuel_cost'], 'values': [fuel_cost]}
    return outputDict

  def neutron_leakage(self):
    """
      Returns Maximum neutron leakage found in the current cycle.
      @ In, None
      @ Out, outputDict, dict, the dictionary containing the read data (None if none found)
                        {'info_ids':list(of ids of data),
                          'values': list}
    """
    outputDict = None
    leakage_list = []
    for line in self.lines:
      if "Total Neutron Leakage" in line:
        elems = line.strip().split()
        leakage_list.append(float(elems[-1]))
    if not leakage_list:
      return ValueError("No values returned. Check Simulate File executed correctly")
    else:
      outputDict = {'info_ids':['neutron_leakage'], 'values':[10000*max(leakage_list)]}
    return outputDict

  def writeCSV(self, fileout):
    """
      Print Data into CSV format
      @ In, fileout, str, the output file name
      @ Out, None
    """
    fileObject = open(fileout.strip()+".csv", mode='wb+') if not fileout.endswith('csv') else open(fileout.strip(), mode='wb+')
    headers=[]
    nParams = numpy.sum([len(data['info_ids']) for data in self.data.values() if data is not None and type(data) is dict])
    outputMatrix = numpy.zeros((nParams,1))
    index=0
    for data in self.data.values():
      if data is not None and type(data) is dict:
        headers.extend(data['info_ids'])
        for i in range(len(data['info_ids'])):
          outputMatrix[index]= data['values'][i]
          index=index+1
    numpy.savetxt(fileObject, outputMatrix.T, delimiter=',', header=','.join(headers), comments='')
    fileObject.close()

