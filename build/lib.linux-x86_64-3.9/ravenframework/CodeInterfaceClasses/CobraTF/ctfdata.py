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
Modified by Alp Tezbasaran @ INL
July 2018
"""

import numpy as np

class ctfdata:
  """
    Class that parses CTF output file and reads in (output files type: .ctf.out) and write a csv file
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, string, file name to be parsed
      @ Out, None
    """

    # convert output file name (deck.ctf.out to deck.out)
    if ('deck.ctf.out' in filen):
      filen = filen.replace('deck.ctf.out', 'deck.out')

    # check file existence (1)
    if ("ctf.out" not in filen) and  ("deck.out" not in filen):
      raise IOError(
            "Check if the supported output file (*.ctf.out or deck.out) is included.")

    # check file existence (2)
    try:
      self.lines = open(filen, "r").readlines()
    except:
      raise IOError('input file missing')

    self.deckEndTimeInfo = self.getTimeDeck(self.lines)
    self.majorData, self.headerName = self.getMajor(self.lines)

  def returnData(self):
    """
      Method to return the data in a dictionary
      @ In, None
      @ Out, data, dict, the dictionary containing the data {var1:array,var2:array,etc}
    """
    data = {var:self.majorData[:,i] for i, var in enumerate(self.headerName)}
    return data

  def getTimeDeck(self, lines):
    """
      Method to check ended time of the simulation (multi-deck compatible)
      @ In, lines, list of lines of the output file
      @ Out, times, dict containing the information {'time':float,'sliceCoordinates':tuple(startLine,EndLine)}.
    """
    times = {}
    startLineNumber, endLineNumber = 0, 1
    for cnt, line in enumerate(lines):
      if ('simulation time =') in line:
        # keep updating until the end of simulation time
        endTime = line.split()[3]
      if ('Sum of timed sections') in line:
        startLineNumber = endLineNumber
        endLineNumber = cnt + 1
        times = {'time': endTime, 'sliceCoordinates': (
            startLineNumber, endLineNumber)}
    return times

  def getMajor(self, lines):
    """
      Method that looks for key word MAJOR EDIT for reading major edit block
      @ In, lines, list of lines of the output file (.ctf.out)
      @ Out, (dictArray, header), tuple, tuple containing:
                                       -[0] -> dictionary containing the edit info
                                       -[1] -> header
    """
    # Booleans
    # start output reading (skip 'input file listing' in the output)
    outTime1 = False                    # Time information
    outTime2 = False                    # Time information
    outType1 = False                    # Channel data first group columns
    outType2 = False                    # Channel data second group columns
    outType3 = False                    # Channel data third group columns
    outType4 = False                    # Fuel Rods
    outType5 = False                    # Cylindrical Rods
    outType6 = False                    # Heat Slabsb
    # descriptions in the output file to discriminate different types of information
    # to check total number of channels & node
    readChannelCount = False
    checkNodeNum = False
    # count time step
    timeStepCount = 0
    # number of averaged variables (total variables that are supposed to be printed in the major edit output file)
    averageVariablesCount = 15
    # number of variables (total variables that are supposed to be printed in the major edit output file)
    variablesCount = 43                             # Channel data (43 column)
    # number of variables for fuel rod data
    variableFuelCount = 10
    # number of variables for cylindrical rod data
    variableCylCount = 8
    # number of variables for heat slabs
    variableHeatSlab = 8
    # number of channels
    channelCount = 0
    # number of axial nodes (channel)
    nodeCount = 0
    # number of fuel nodes
    fuelCount = 0
    # number of surfaces (total)
    surfaceCount = 0
    # number of heat slabs
    heatSlabCount = 0

    # 1. count the total number of time steps, channels, axial nodes, fuel rods and cylindrical rods.
    for line in lines:
      if all(x in line for x in ['aver.', 'properties' , 'channels']):
        outTime1 = True # for output reading start (stay 'True' once it is activated)
        outTime2 = True # to check simTime evolution
        timeStepCount += 1
      if all(x in line for x in ['fluid', 'properties' , 'channel']):
        outType1 = True
      # to read total number of channels
      if all(x in line for x in ['of', 'channels', 'nodes', '(nominal)']):
        readChannelCount = True
      if (readChannelCount) and (line != '\n') and (line.split()[0].isdigit()):
      # read total channel number
        channelCount = int(line.split()[0])
        nodeCount = int(line.split()[1])
      # to stop reading channel count
      if 'channel           channels above                   channels below' in line:
        readChannelCount = False
      # read total fuel rod number
      if all(x in line for x in ['no.', 'of' , 'fuel', 'rods']):
        fuelCount = int(line.split()[5])                            # this number includes fuel rods and cylindrical tubes (output inherently reports it that way)
      # read total surface number
      if all(x in line for x in ['no.', 'of' , 'fuel', 'surfaces']):
        surfaceCount = int(line.split()[-1])                        # total surface count (not surface number per node)
      # read total heat slab number
      if all(x in line for x in ['no.', 'of' , 'heat', 'slabs']):
        heatSlabCount = int(line.split()[-1])                        # total heat slab count

    dictArray = np.zeros((int(timeStepCount), int(
            variablesCount * channelCount * (nodeCount + 1)  + variableFuelCount * fuelCount *
              surfaceCount * (nodeCount + 2) + variableHeatSlab * heatSlabCount * (nodeCount)
            + 1 * averageVariablesCount * (nodeCount + 1) + 1)))
    # 3. read input again (assign values to dictArray)
    # initialize booleans
    outTime1 = False                # Time information
    outTime2 = False                # Time information
    outTypeAvg = False              # Average chanel
    outType1 = False                # Channel data first group
    outType2 = False                # Channel data second group
    outType3 = False                # Channel data third group
    outType4 = False                # Fuel Rod
    outType5 = False                # Cylinder Rod
    outType6 = False                # Heat Slab
    headerWrite = True

    internalFlow = False            # Internal Flow flag
    externalFlow = False            # External flow flag

    timeStepCount = 0               # Time pointer
    variableNumber = 0              # Variable pointer

    # create new array (for header names)
    header = []
    for line in lines:
      if all(x in line for x in ['aver.', 'properties' , 'channels']):
        # output type (average values)
        outTypeAvg = True
        # outTime1: output reading start (stay 'True' once it is activated)
        # outTime2: check simTime evolution
        outTime1, outTime2 = True, True
        # save simTime to major_Dict
        simTime = line.split()[3]
        # save time info. into dictArray[]
        dictArray[timeStepCount, 0] = simTime
        # array of header names (write only once initially)
        if (timeStepCount >= 1):
          headerWrite = False
        else:
          header.append('time')
        # time step increase
        timeStepCount += 1
        # rewind varablesNumber every time step
        variableNumber = 0
        # to check time evolution
        outTime2 = False
      # output type 1
      if all(x in line for x in ['fluid', 'properties' , 'channel']):
        outType1 = True
        channelNumber = line.split()[9]
      # output type 2
      if all(x in line for x in ['enthalpy', 'density', 'net']):
        outType2 = True
      # output type 3
      if all(x in line for x in ['----', 'gas' , 'volumetric', 'analysis']):
        outType3 = True
      # output type 4 (fuel rod)
      if all(x in line for x in ['nuclear', 'fuel' , 'rod', 'no.']):
        outType4 = True
        fuelNumber = int(line.split()[-6])
      # surface of the fuel rods
      if (outType4 == True) and len(line.split()) == 5 and all(x in line for x in ['surface', 'no.']):
        surfaceNumber = int(line.split()[-3])
      # output type 5 (cylindrical tube)
      if all(x in line for x in ['cylindrical', 'tube' , 'rod', 'no.']):
        outType5 = True
        cylRodNumber = int(line.split()[-6])
      # surface number of the cylindrical tubes
      if (outType5 == True) and len(line.split()) == 5 and all(x in line for x in ['surface', 'no.']):
        surfaceNumber = int(line.split()[-3])
      # decide if the flow is external or internal
      if (outType5) and (line != '\n') and (line.split()[0].isdigit()):
        if line.split()[3].replace('.', '', 1).isdigit():
          internalFlow = True
        elif line.split()[-2].replace('.', '', 1).isdigit():
          externalFlow = True
      # heat slab number
      if all(x in line for x in ['heat', 'slab' , '(tube)', 'no.','simulation', 'time']):
        outType6 = True
        heatSlabNumber = int(line.split()[3])
      # decide if the flow is internal or external
      if outType6 and ('fluid channel on inside surface =  0' in line):
        externalFlow = True
      elif outType6 and ('fluid channel on outside surface =  0' in line):
        internalFlow = True
      # skip any blank line and lines that don't start with number
      if (outTime1 == True) and (line != '\n') and (line.split()[0].isdigit()):
        # filter few more lines that include characters
        if (line.split()[1].replace('.', '', 1).isdigit()):
          if outTypeAvg:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'AVG_ch' + '_ax' + str(ax)
            for j in range(1, len(line.split())):
              if line.split()[2] != '*****':
                variableNumber += 1
                dictArray[timeStepCount - 1,
                        variableNumber] = line.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_quality'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_voidFractionLiquid'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_voidFractionVapor'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_volumeEntrainFraction'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_massFlowRateLiquid'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_massFlowRateVapor'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_massFlowRateEntrain'
                  header.append(headerName)
                elif j == 9:
                  headerName = keyName + '_massFlowRateIntegrated'
                  header.append(headerName)
                elif j == 10:
                  headerName = keyName + '_enthalpyIncreaseLiquid'
                  header.append(headerName)
                elif j == 11:
                  headerName = keyName + '_enthalpyIncreaseVapor'
                  header.append(headerName)
                elif j == 12:
                  headerName = keyName + '_enthalpyIncreaseIntegrated'
                  header.append(headerName)
                elif j == 13:
                  headerName = keyName + '_enthalpyMixture'
                  header.append(headerName)
                elif j == 14:
                  headerName = keyName + '_heatAddedToLiquid'
                  header.append(headerName)
                elif j == 15:
                  headerName = keyName + '_heatAddedToVapor'
                  header.append(headerName)
                elif j == 16:
                  headerName = keyName + '_heatAddedIntegrated'
                  header.append(headerName)
                else:
                  raise IOError(
                      "Error: Unexpected output file format. Check the oufput file (output type: average channel).")
                # at end of rod info, reset outTypeAvg booleans
            if ax == 1:
              outTypeAvg = False
          # read output type 1
          if outType1:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'ch' + channelNumber + '_ax' + str(ax)
            for j in range(1, len(line.split())):
              variableNumber += 1
              dictArray[timeStepCount - 1,
                          variableNumber] = line.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_pressure'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_velocityLiquid'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_velocityVapor'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_velocityEntrain'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_voidFractionLiquid'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_voidFractionVapor'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_volumeEntrainFraction'
                  header.append(headerName)
                elif j == 9:
                  headerName = keyName + '_massFlowRateLiquid'
                  header.append(headerName)
                elif j == 10:
                  headerName = keyName + '_massFlowRateVapor'
                  header.append(headerName)
                elif j == 11:
                  headerName = keyName + '_massFlowRateEntrain'
                  header.append(headerName)
                elif j == 12:
                  headerName = keyName + '_flowRegimeID'
                  header.append(headerName)
                elif j == 13:
                  headerName = keyName + '_heatAddedToLiquid'
                  header.append(headerName)
                elif j == 14:
                  headerName = keyName + '_heatAddedToVapor'
                  header.append(headerName)
                elif j == 15:
                  headerName = keyName + '_evaporationRate'
                  header.append(headerName)
                else:
                  raise IOError(
                     "Error: Unexpected output file format. Check the oufput file (output type 3).")
            # at end of rod info, reset outType1 booleans
            if ax == 1:
              outType1 = False

          # read output type 2
          if outType2:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'ch' + channelNumber + '_ax' + str(ax)
            for j in range(1, len(line.split())):
              variableNumber += 1
              dictArray[timeStepCount - 1,
                       variableNumber] = line.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_enthalpyVapor'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_enthalpySaturatedVapor'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_enthalpyVapor-SaturatedVapor'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_enthalpyLiquid'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_enthalpySaturatedLiquid'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_enthalpyLiquid-SaturatedLiquid'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_enthalpyMixture'
                  header.append(headerName)
                elif j == 9:
                  headerName = keyName + '_densityLiquid'
                  header.append(headerName)
                elif j == 10:
                  headerName = keyName + '_densityVapor'
                  header.append(headerName)
                elif j == 11:
                  headerName = keyName + '_densityMixture'
                  header.append(headerName)
                elif j == 12:
                  headerName = keyName + '_netEntrainRate'
                  header.append(headerName)
                else:
                  raise IOError(
                      "Error: Unexpected output file format. Check the oufput file (output type 2).")
                # at end of rod info, reset outType2 booleans
            if ax == 1:
              outType2 = False

          # read output type 3
          if outType3:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'ch' + channelNumber + '_ax' + str(ax)
            for j in range(1, len(line.split())):
              variableNumber += 1
              dictArray[timeStepCount - 1,
                variableNumber] = line.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_enthalpyNonCondensableMixture'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_densityNonCondensableMixture'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_volumeFractionSteam'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_volumeFractionAir'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_dummy1'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_dummy2'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_dummy3'
                  header.append(headerName)
                elif j == 9:
                  headerName = keyName + '_dummy4'
                  header.append(headerName)
                elif j == 10:
                  headerName = keyName + '_equiDiameterLiquidDroplet'
                  header.append(headerName)
                elif j == 11:
                  headerName = keyName + '_avgDiameterLiquidDroplet'
                  header.append(headerName)
                elif j == 12:
                  headerName = keyName + '_avgFlowRateLiquidDroplet'
                  header.append(headerName)
                elif j == 13:
                  headerName = keyName + '_avgVelocityLiquidDroplet'
                  header.append(headerName)
                elif j == 14:
                  headerName = keyName + '_evaporationRateLiquidDroplet'
                  header.append(headerName)
                else:
                  raise IOError(
                    "Error: Unexpected output file format. Check the oufput file (output type 3).")
                # at end of rod info, reset outType3 booleans
            if ax == 1:
              outType3 = False
        if (line.split()[1].replace('.', '', 1).isdigit()) or ('*' in line.split()[1]):

          # read output type 4
          if outType4:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'fuelRod' + str(fuelNumber) + '_surface' + str(surfaceNumber) + '_ax' + str(ax)
            # line filtering due to the existence of '*' in the output
            if '*' in line:
              lineFiltered = line.replace('*', "")
            else:
              lineFiltered = line
            for j in range(1, len(lineFiltered.split())):
              variableNumber += 1
              # skip the non-numeric elements
              if j != 5:
                dictArray[timeStepCount - 1,
                          variableNumber] = lineFiltered.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_fluidTemperatureLiquid'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_fluidTemperatureVapor'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_surfaceHeatflux'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_heatTransferMode'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_cladOutTemperature'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_cladInTemperature'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_gapConductance'
                  header.append(headerName)
                elif j == 9:
                  headerName = keyName + '_fuelTemperatureSurface'
                  header.append(headerName)
                elif j == 10:
                  headerName = keyName + '_fuelTemperatureCenter'
                  header.append(headerName)
                else:
                  raise IOError(
                      "Error: Unexpected output file format. Check the oufput file (output type 4, fuel rod).")
            # at end of rod info, reset outType4 booleans
            if ax == 1:
              outType4 = False

          # read output type 5 for external flow
          if outType5 and externalFlow:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'cylRod' + str(cylRodNumber) + '_surface' + str(surfaceNumber) + '_ax' + str(ax)
            # line filtering due to the existence of '*' in the output
            if '*' in line:
              lineFiltered = line.replace('*', "")
            else:
              lineFiltered = line
            for j in range(1, len(lineFiltered.split())):
              variableNumber += 1
              # skip the non-numeric elements
              if j != 3:
                dictArray[timeStepCount - 1,
                          variableNumber] = lineFiltered.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_outsideSurfaceHeatFlux'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_outsideSurfaceTransferMode'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_outsideSurfaceWallTemperature'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_outsideSurfaceVaporTemperature'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_outsideSurfaceLiquidTemperature'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_insideSurfaceWallTemperature'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_insideSurfaceHeatFlux'
                  header.append(headerName)
                else:
                  raise IOError(
                      "Error: Unexpected output file format. Check the oufput file (output type 5, fuel rod, external flow).")
            # at end of rod info, reset outType5 booleans
            if ax == 1:
              outType5 = False
              externalFlow = False

          # read output type 5 for internal flow
          if outType5 and internalFlow:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'cylRod' + str(cylRodNumber) + '_surface' + str(surfaceNumber) + '_ax' + str(ax)
            # line filtering due to the existence of '*' in the output
            if '*' in line:
              lineFiltered = line.replace('*', "")
            else:
              lineFiltered = line
            for j in range(1, len(lineFiltered.split())):
              variableNumber += 1
              # skip the non-numeric elements
              if j != 7:
                dictArray[timeStepCount - 1,
                          variableNumber] = lineFiltered.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_outsideSurfaceHeatFlux'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_outsideSurfaceWallTemperature'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_insideSurfaceLiquidTemperature'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_insideSurfaceVaporTemperature'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_insideSurfaceWallTemperature'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_insideSurfaceTransferMode'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_insideSurfaceHeatFlux'
                  header.append(headerName)
                else:
                  raise IOError(
                      "Error: Unexpected output file format. Check the oufput file (output type 5, fuel rod, internal flow).")
            # at end of rod info, reset outType5 booleans
            if ax == 1:
              outType5 = False
              internalFlow = False

          # read output type 6 for internal flow
          if outType6 and internalFlow:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'heatSlab' + str(heatSlabNumber) + '_ax' + str(ax)
            # line filtering due to the existence of '*' in the output
            if '*' in line:
              lineFiltered = line.replace('*', "")
            else:
              lineFiltered = line
            for j in range(1, len(lineFiltered.split())):
              variableNumber += 1
              # skip the non-numeric elements
              if j != 7:
                dictArray[timeStepCount - 1,
                          variableNumber] = lineFiltered.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_outsideSurfaceHeatFlux'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_outsideSurfaceWallTemperature'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_insideSurfaceLiquidTemperature'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_insideSurfaceVaporTemperature'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_insideSurfaceWallTemperature'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_insideSurfaceTransferMode'
                  header.append(headerName)
                elif j == 8:
                  headerName = keyName + '_insideSurfaceHeatFlux'
                  header.append(headerName)
                else:
                  raise IOError(
                      "Error: Unexpected output file format. Check the oufput file (output type 6, heat slab, internal flow).")
            # at end of rod info, reset outType6 booleans
            if ax == 2:
              outType6 = False
              internalFlow = False

          # read output type 6 for external flow
          if outType6 and externalFlow:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'heatSlab' + str(heatSlabNumber) + '_ax' + str(ax)
            # line filtering due to the existence of '*' in the output
            if '*' in line:
              lineFiltered = line.replace('*', "")
            else:
              lineFiltered = line
            for j in range(1, len(lineFiltered.split())):
              variableNumber += 1
              # skip the non-numeric elements
              if j != 3:
                dictArray[timeStepCount - 1,
                          variableNumber] = lineFiltered.split()[j]
              if headerWrite:
                if j == 1:
                  headerName = keyName + '_height'
                  header.append(headerName)
                elif j == 2:
                  headerName = keyName + '_outsideSurfaceHeatFlux'
                  header.append(headerName)
                elif j == 3:
                  headerName = keyName + '_outsideSurfaceTransferMode'
                  header.append(headerName)
                elif j == 4:
                  headerName = keyName + '_outsideSurfaceWallTemperature'
                  header.append(headerName)
                elif j == 5:
                  headerName = keyName + '_outsideSurfaceVaporTemperature'
                  header.append(headerName)
                elif j == 6:
                  headerName = keyName + '_outsideSurfaceLiquidTemperature'
                  header.append(headerName)
                elif j == 7:
                  headerName = keyName + '_insideSurfaceWallTemperature'
                  header.append(headerName)
                else:
                  raise IOError(
                      "Error: Unexpected output file format. Check the oufput file (output type 6, heat slab, external flow).")
            # at end of rod info, reset outType6 booleans
            if ax == 2:
              outType6 = False
              externalFlow = False

    return dictArray, header
