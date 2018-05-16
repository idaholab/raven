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

import numpy as np

class ctfdata:
  """
    Class that parses CTF output file and reads in (output files type: .ctf.out) and write a csv file
  """

  def __init__(self, filen):
    """
      Constructor
      @ In, filen, string, file name to be parsed
      @ Out, None
    """
    # check file existence (1)
    if "ctf.out" not in filen:
      raise IOError(
            "Check if the supported output file (*.ctf.out) is included.")
    # check file existence (2)
    try:
      self.lines = open(filen, "r").readlines()
    except:
      raise IOError('input file missing')
    self.deckEndTimeInfo = self.getTimeDeck(self.lines)
    self.majorData, self.headerName = self.getMajor(self.lines)

  def writeCSV(self, filen):
    """
      Method that writes the csv file from major edit data
      @ In, filen, string (input file name)
      @ Out, None
    """
    # create string for header names
    headerString = ",".join(self.headerName)
    # write & save array as csv file
    np.savetxt(filen, self.majorData, delimiter=',', header=headerString, comments='')

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
      @ Out, majorDict, dict, dictionary containing the major edit info
    """
    # Booleans
    # start output reading (skip 'input file listing' in the output)
    outTime1 = False
    outTime2 = False
    outType1 = False
    outType2 = False
    outType3 = False
    outType4 = False
    # descriptions in the output file to discriminate different types of information
    # to check total number of channels & node
    checkChannelNum = False
    checkNodeNum = False
    # count time step
    timestepCount = 0
    # number of averaged variables (total variables that are supposed to be printed in the major edit output file)
    averageVariablesCount = 15
    # number of variables (total variables that are supposed to be printed in the major edit output file)
    variablesCount = 38
    # number of channels
    channelCount = 0
    # number of axial nodes (channel)
    nodeCount = 0
    # number of fuel nodes
    fuelCount = 0
    count1 = 0
    count2 = 0
    # 1. count the total number of time steps, channels, axial nodes, and fuel rods.
    for line in lines:
      # to check total number of channel
      if 'subchannel data' in line:
        checkChannelNum = True
      if 'axial loss coefficients' in line:
        checkChannelNum = False
        checkNodeNum = True
      if '------- *******************' in line:
        checkNodeNum = False
      if all(x in line for x in ['aver.', 'properties' , 'channels']):
        outTime1 = True # for output reading start (stay 'True' once it is activated)
        outTime2 = True # to check simTime evolution
        timestepCount += 1
      if all(x in line for x in ['fluid', 'properties' , 'channel']):
        outType1 = True
        channelNumber = line.split()[9]
      if all(x in line for x in ['nuclear', 'fuel' , 'rod', 'no.']):
        outType4 = True
      # read total channel number
      if (checkChannelNum == True) and (line != '\n') and (line.split()[0].isdigit()):
        channelCount = int(line.split()[0])
      # read total node number per channel
      if (checkNodeNum == True) and (line != '\n') and (line.split()[0].isdigit()):
        nodeCount = int(line.split()[len(line.split()) - 1])
      # skip any blank line and lines that don't start with number
      if (outTime1 == True) and (line != '\n') and (line.split()[0].isdigit()):
        # filter few more lines that include characters
        if (line.split()[1].replace('.', '', 1).isdigit()):
          # read output type 1
          if outType1:
            count1 = count1 + 1
            if (count1 == 1):  # read axial node number
              nodeCount = int(line.split()[0])
              outType1 = False
          if outType4:
            count2 = count2 + 1
            if (count2 == 1):  # read fuel rod number
              fuelCount = int(line.split()[0])
              outType4 = False
    # check if output file (*ctf.out) has information for channels and nodes
    if (count1 == 0):
      channelCount = 1
      variablesCount = 0
    # 2. create numpy array (based on the output reading)
    if count2 == 0:  # w/o fuel rod
      dictArray = np.zeros((int(timestepCount), int(
          (variablesCount + averageVariablesCount) * channelCount * nodeCount + 1)))
    else:  # with fuel rod (10)
      dictArray = np.zeros((int(timestepCount), int(
            (variablesCount + averageVariablesCount + 10) * channelCount * nodeCount + 1)))
    # 3. read input again (assign values to dictArray)
    # initialize booleans
    outTime1 = False
    outTime2 = False
    outTypeAvg = False
    outType1 = False
    outType2 = False
    outType3 = False
    outType4 = False
    headerWrite = True

    timestepCount = 0
    variableNumber = 0
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
        dictArray[timestepCount, 0] = simTime
        # array of header names (write only once initially)
        if (timestepCount >= 1):
          headerWrite = False
        else:
          header.append('time')
        # time step increase
        timestepCount += 1
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
      # output type 4 (fuel ord)
      if all(x in line for x in ['nuclear', 'fuel' , 'rod', 'no.']):
        outType4 = True
      # skip any blank line and lines that don't start with number
      if (outTime1 == True) and (line != '\n') and (line.split()[0].isdigit()):
        # filter few more lines that include characters
        if (line.split()[1].replace('.', '', 1).isdigit()):
          if outTypeAvg:
            # define axial location
            ax = int(line.split()[0])
            # save header name of each variable
            keyName = 'AVG_ch' + '_ax' + str(ax)
            for j in range(2, len(line.split())):
              variableNumber += 1
              dictArray[timestepCount - 1,
                        variableNumber] = line.split()[j]
              if headerWrite:
                if j == 2:
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
            for j in range(2, len(line.split())):
              variableNumber += 1
              dictArray[timestepCount - 1,
                          variableNumber] = line.split()[j]
              if headerWrite:
                if j == 2:
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
            for j in range(2, len(line.split())):
              variableNumber += 1
              dictArray[timestepCount - 1,
                       variableNumber] = line.split()[j]
              if headerWrite:
                if j == 2:
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
            for j in range(2, len(line.split())):
              variableNumber += 1
              dictArray[timestepCount - 1,
                variableNumber] = line.split()[j]
              if headerWrite:
                if j == 2:
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
            fuelNumber = int(line.split()[0])
            # save header name of each variable
            keyName = 'fuel_rod' + str(fuelNumber)
            # line filtering due to the existence of '*' in the output
            if '*' in line:
              lineFiltered = line.replace('*', "")
            else:
              lineFiltered = line
            for j in range(2, len(lineFiltered.split())):
              variableNumber += 1
              # skip the non-numeric elements
              if j != 5:
                dictArray[timestepCount - 1,
                          variableNumber] = lineFiltered.split()[j]
              if headerWrite:
                if j == 2:
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
                  headerName = keyName + '_caldOutTemperature'
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
            # at end of rod info, reset outType3 booleans
            if fuelNumber == 1:
              outType4 = False
    return dictArray, header
