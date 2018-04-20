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
    """Class that parses CTF output file and reads in (types of output files??) and write a csv file"""

    def __init__(self, filen):
        
        # Constructor
        # @ In, filen, string, file name to be parsed
        # @ Out, None

        # check file existence (1)
        if "ctf.out" not in filen:
            raise ValueError(
                "Check if the supported output file (*.ctf.out) is included.")
        # check file existence (2)
        try:
            self.lines = open(filen, "r").readlines()
        except IOError:
            raise ValueError('Case file missing: ' + file_output)

        self.deckEndTimeInfo = self.getTimeDeck(self.lines)
        startLine, endLine = self.deckEndTimeInfo['sliceCoordinates'][0:2]
        self.majordata, self.HeaderName = self.getMajor(self.lines)

    def writeCSV(self, filen):
        
        # Method that writes the csv file from major edit data
        # @ In, filen, string (input file name)
        # @ Out, None
        
        # create string for header names
        header_str = ",".join(self.HeaderName)
        # write & save array as csv file
        np.savetxt(filen, self.majordata, delimiter=',', header=header_str, comments='')

    def getTimeDeck(self, lines):
        
        # Method to check ended time of the simulation (multi-deck compatible)
        # @ In, lines, list of lines of the output file
        # @ Out, times, dict containing the information {'time':float,'sliceCoordinates':tuple(startLine,EndLine)}.
        
        times = {}
        startLineNumber, endLineNumber = 0, 1

        for cnt, line in enumerate(lines):
            if ('simulation time =') in line:
                # keep updating until the end of simulation time
                end_time = line.split()[3]

            if ('Sum of timed sections') in line:
                startLineNumber = endLineNumber
                endLineNumber = cnt + 1
                times = {'time': end_time, 'sliceCoordinates': (
                    startLineNumber, endLineNumber)}
        return times
     
    def getMajor(self, lines):
        # Method that looks for key word MAJOR EDIT for reading major edit block
        # @ In, lines, list of lines of the output file
        # @ Out, majorDict, dict, dictionary containing the major edit info

        # Booleans
        # start output reading (skip 'input file listing' in the output)
        out_time = False
        out_time2 = False
        o_type1 = False
        o_type2 = False
        o_type3 = False
        o_type4 = False

        # to check total number of channels & node
        checkChannelNum = False
        checkNodeNum = False

        # count time step
        timestep_cnt = 0
        # number of averaged variables (total variables in major edit)
        avg_variables_cnt = 15
        # number of variables (total variables in major edit)
        variables_cnt = 38
        # number of channels
        channel_cnt = 0
        # number of axial nodes (channel)
        node_cnt = 0
        # number of fuel nodes
        fuel_cnt = 0
        count1 = 0
        count2 = 0
        # 1. count the total number of time steps, channels, axial nodes, and fuel rods.
        for line in lines:
            out_tag_avg = 'aver. properties for channels'
            out_tag1 = 'fluid properties for channel'
            out_tag4 = 'nuclear fuel rod no.'
            # to check total number of channel
            if 'subchannel data' in line:
                checkChannelNum = True
            if 'axial loss coefficients' in line:
                checkChannelNum = False
                checkNodeNum = True
            if '------- *******************' in line:
                checkNodeNum = False

            if out_tag_avg in line:
                # out_time : output reading start (stay 'True' once it is activated)
                # out_time2: check sim_time evolution
                out_time, out_time2 = True, True
                timestep_cnt += 1

            if out_tag1 in line:
                o_type1 = True
                chan_no = line.split()[9]
                ##channel_cnt += 1

            if out_tag4 in line:
                o_type4 = True

            # read total channel number
            if (checkChannelNum == True) and (line != '\n') and (line.split()[0].isdigit()):
                channel_cnt = int(line.split()[0])
            # read total node number per channel
            if (checkNodeNum == True) and (line != '\n') and (line.split()[0].isdigit()):
                node_cnt = int(line.split()[len(line.split()) - 1])

            # skip any blank line and lines that don't start with number
            if (out_time == True) and (line != '\n') and (line.split()[0].isdigit()):
                # filter few more lines that include characters
                if (line.split()[1].replace('.', '', 1).isdigit()):
                    # read output type 1
                    if o_type1:
                        count1 = count1 + 1
                        if (count1 == 1):  # read axial node number
                            node_cnt = int(line.split()[0])
                            o_type1 = False

                    if o_type4:
                        count2 = count2 + 1
                        # print(line)
                        if (count2 == 1):  # read fuel rod number
                            fuel_cnt = int(line.split()[0])
                            o_type4 = False

        # check if output file (*ctf.out) has information for channels and nodes
        if (count1 == 0):
            channel_cnt = 1
            variables_cnt = 0

        # 2. create numpy array (based on the output reading)
        if count2 == 0:  # w/o fuel rod
            DictArray = np.zeros((int(timestep_cnt), int(
                (variables_cnt + avg_variables_cnt) * channel_cnt * node_cnt + 1)))
        else:  # with fuel rod (10)
            DictArray = np.zeros((int(timestep_cnt), int(
                (variables_cnt + avg_variables_cnt + 10) * channel_cnt * node_cnt + 1)))

        print(DictArray.shape)
        print(variables_cnt, avg_variables_cnt, channel_cnt, count1)

        # 3. read input again (assign values to DictArray)
        # initialize booleans
        out_time = False
        out_time2 = False
        o_type_avg = False
        #
        o_type1 = False
        o_type2 = False
        o_type3 = False
        o_type4 = False
        header_w = True

        timestep_cnt = 0
        var_no = 0
        # create new array (for header names)
        header = []
        for line in lines:
            out_tag_avg = 'aver. properties for channels'
            out_tag1 = 'fluid properties for channel'
            out_tag2 = 'enthalpy                                                  density             net'
            out_tag3 = '-------------------------------- gas volumetric analysis --------------------------------'
            out_tag4 = 'nuclear fuel rod no.'

            if out_tag_avg in line:
                # output type (average values)
                o_type_avg = True
                # out_time : output reading start (stay 'True' once it is activated)
                # out_time2: check sim_time evolution
                out_time, out_time2 = True, True
                # save sim_time to major_Dict
                sim_time = line.split()[3]

                # save time info. into DictArray[]
                DictArray[timestep_cnt, 0] = sim_time
                # array of header names (write only once initially)
                if (timestep_cnt >= 1):
                    header_w = False
                # array of header names (only add once when header_w=True)
                if header_w:
                    header.append('time')
                # time step increase
                timestep_cnt += 1
                # rewind var_no every time step
                var_no = 0
                # to check time evolution
                out_time2 = False

            # output type 1
            if out_tag1 in line:
                o_type1 = True
                chan_no = line.split()[9]
            # output type 2
            if out_tag2 in line:
                o_type2 = True
            # output type 3
            if out_tag3 in line:
                o_type3 = True
            # output type 4 (fuel ord)
            if out_tag4 in line:
                o_type4 = True

            # skip any blank line and lines that don't start with number
            if (out_time == True) and (line != '\n') and (line.split()[0].isdigit()):
                # filter few more lines that include characters
                if (line.split()[1].replace('.', '', 1).isdigit()):
                    if o_type_avg:
                        # define axial location
                        ax = int(line.split()[0])

                        # save header name of each variable
                        key_name = 'AVG_ch' + '_ax' + str(ax)
                        for j in range(2, len(line.split())):

                            # print(DictArray.shape)

                            var_no += 1
                            DictArray[timestep_cnt - 1,
                                      var_no] = line.split()[j]

                            if header_w:
                                #key_name = 'ch' + chan_no + '_ax' + str(ax)
                                if j == 2:
                                    header_name = key_name + '_quality'
                                    header.append(header_name)
                                elif j == 3:
                                    header_name = key_name + '_voidFractionLiquid'
                                    header.append(header_name)
                                elif j == 4:
                                    header_name = key_name + '_voidFractionVapor'
                                    header.append(header_name)
                                elif j == 5:
                                    header_name = key_name + '_volumeEntrainFraction'
                                    header.append(header_name)
                                elif j == 6:
                                    header_name = key_name + '_massFlowRateLiquid'
                                    header.append(header_name)
                                elif j == 7:
                                    header_name = key_name + '_massFlowRateVapor'
                                    header.append(header_name)
                                elif j == 8:
                                    header_name = key_name + '_massFlowRateEntrain'
                                    header.append(header_name)
                                elif j == 9:
                                    header_name = key_name + '_massFlowRateIntegrated'
                                    header.append(header_name)
                                elif j == 10:
                                    header_name = key_name + '_enthalpyIncreaseLiquid'
                                    header.append(header_name)
                                elif j == 11:
                                    header_name = key_name + '_enthalpyIncreaseVapor'
                                    header.append(header_name)
                                elif j == 12:
                                    header_name = key_name + '_enthalpyIncreaseIntegrated'
                                    header.append(header_name)
                                elif j == 13:
                                    header_name = key_name + '_enthalpyMixture'
                                    header.append(header_name)
                                elif j == 14:
                                    header_name = key_name + '_heatAddedToLiquid'
                                    header.append(header_name)
                                elif j == 15:
                                    header_name = key_name + '_heatAddedToVapor'
                                    header.append(header_name)
                                elif j == 16:
                                    header_name = key_name + '_heatAddedIntegrated'
                                    header.append(header_name)
                                else:
                                    raise IOError(
                                        "Error: Unexpected output file format. Check the oufput file (output type: average channel).")

                        # at end of rod info, reset o_type_avg booleans
                        if ax == 1:
                            o_type_avg = False

                    # read output type 1
                    if o_type1:
                        # define axial location
                        ax = int(line.split()[0])

                        # save header name of each variable
                        key_name = 'ch' + chan_no + '_ax' + str(ax)

                        for j in range(2, len(line.split())):
                            var_no += 1
                            DictArray[timestep_cnt - 1,
                                      var_no] = line.split()[j]
                            if header_w:
                                #key_name = 'ch' + chan_no + '_ax' + str(ax)
                                if j == 2:
                                    header_name = key_name + '_pressure'
                                    header.append(header_name)
                                elif j == 3:
                                    header_name = key_name + '_velocityLiquid'
                                    header.append(header_name)
                                elif j == 4:
                                    header_name = key_name + '_velocityVapor'
                                    header.append(header_name)
                                elif j == 5:
                                    header_name = key_name + '_velocityEntrain'
                                    header.append(header_name)
                                elif j == 6:
                                    header_name = key_name + '_voidFractionLiquid'
                                    header.append(header_name)
                                elif j == 7:
                                    header_name = key_name + '_voidFractionVapor'
                                    header.append(header_name)
                                elif j == 8:
                                    header_name = key_name + '_volumeEntrainFraction'
                                    header.append(header_name)
                                elif j == 9:
                                    header_name = key_name + '_massFlowRateLiquid'
                                    header.append(header_name)
                                elif j == 10:
                                    header_name = key_name + '_massFlowRateVapor'
                                    header.append(header_name)
                                elif j == 11:
                                    header_name = key_name + '_massFlowRateEntrain'
                                    header.append(header_name)
                                elif j == 12:
                                    header_name = key_name + '_flowRegimeID'
                                    header.append(header_name)
                                elif j == 13:
                                    header_name = key_name + '_heatAddedToLiquid'
                                    header.append(header_name)
                                elif j == 14:
                                    header_name = key_name + '_heatAddedToVapor'
                                    header.append(header_name)
                                elif j == 15:
                                    header_name = key_name + '_evaporationRate'
                                    header.append(header_name)
                                else:
                                    raise IOError(
                                        "Error: Unexpected output file format. Check the oufput file (output type 3).")

                        # Conversion to SI unit (??????????????)

                        # at end of rod info, reset o_type1 booleans
                        if ax == 1:
                            o_type1 = False

                    # read output type 2
                    if o_type2:
                        # define axial location
                        ax = int(line.split()[0])

                        # save header name of each variable
                        key_name = 'ch' + chan_no + '_ax' + str(ax)

                        for j in range(2, len(line.split())):
                            var_no += 1
                            DictArray[timestep_cnt - 1,
                                      var_no] = line.split()[j]
                            if header_w:
                                #key_name = 'ch' + chan_no + '_ax' + str(ax)
                                if j == 2:
                                    header_name = key_name + '_enthalpyVapor'
                                    header.append(header_name)
                                elif j == 3:
                                    header_name = key_name + '_enthalpySaturatedVapor'
                                    header.append(header_name)
                                elif j == 4:
                                    header_name = key_name + '_enthalpyVapor-SaturatedVapor'
                                    header.append(header_name)
                                elif j == 5:
                                    header_name = key_name + '_enthalpyLiquid'
                                    header.append(header_name)
                                elif j == 6:
                                    header_name = key_name + '_enthalpySaturatedLiquid'
                                    header.append(header_name)
                                elif j == 7:
                                    header_name = key_name + '_enthalpyLiquid-SaturatedLiquid'
                                    header.append(header_name)
                                elif j == 8:
                                    header_name = key_name + '_enthalpyMixture'
                                    header.append(header_name)
                                elif j == 9:
                                    header_name = key_name + '_densityLiquid'
                                    header.append(header_name)
                                elif j == 10:
                                    header_name = key_name + '_densityVapor'
                                    header.append(header_name)
                                elif j == 11:
                                    header_name = key_name + '_densityMixture'
                                    header.append(header_name)
                                elif j == 12:
                                    header_name = key_name + '_netEntrainRate'
                                    header.append(header_name)
                                else:
                                    raise IOError(
                                        "Error: Unexpected output file format. Check the oufput file (output type 2).")

                        # Conversion to SI unit (??????????????????)
                        # vapor ethalpy [btu/lbm]

                        # at end of rod info, reset o_type2 booleans
                        if ax == 1:
                            o_type2 = False

                    # read output type 3
                    if o_type3:
                        # define axial location
                        ax = int(line.split()[0])

                        # print(len(line.split()))

                        # save header name of each variable
                        key_name = 'ch' + chan_no + '_ax' + str(ax)
                        for j in range(2, len(line.split())):
                            var_no += 1
                            DictArray[timestep_cnt - 1,
                                      var_no] = line.split()[j]
                            if header_w:
                                #key_name = 'ch' + chan_no + '_ax' + str(ax)
                                if j == 2:
                                    header_name = key_name + '_enthalpyNonCondensableMixture'
                                    header.append(header_name)
                                elif j == 3:
                                    header_name = key_name + '_densityNonCondensableMixture'
                                    header.append(header_name)
                                elif j == 4:
                                    header_name = key_name + '_volumeFractionSteam'
                                    header.append(header_name)
                                elif j == 5:
                                    header_name = key_name + '_volumeFractionAir'
                                    header.append(header_name)
                                elif j == 6:
                                    header_name = key_name + '_dummy1'
                                    header.append(header_name)
                                elif j == 7:
                                    header_name = key_name + '_dummy2'
                                    header.append(header_name)
                                elif j == 8:
                                    header_name = key_name + '_dummy3'
                                    header.append(header_name)
                                elif j == 9:
                                    header_name = key_name + '_dummy4'
                                    header.append(header_name)
                                elif j == 10:
                                    header_name = key_name + '_equiDiameterLiquidDroplet'
                                    header.append(header_name)
                                elif j == 11:
                                    header_name = key_name + '_avgDiameterLiquidDroplet'
                                    header.append(header_name)
                                elif j == 12:
                                    header_name = key_name + '_avgFlowRateLiquidDroplet'
                                    header.append(header_name)
                                elif j == 13:
                                    header_name = key_name + '_avgVelocityLiquidDroplet'
                                    header.append(header_name)
                                elif j == 14:
                                    header_name = key_name + '_evaporationRateLiquidDroplet'
                                    header.append(header_name)
                                else:
                                    raise IOError(
                                        "Error: Unexpected output file format. Check the oufput file (output type 3).")

                        # at end of rod info, reset o_type3 booleans
                        if ax == 1:
                            o_type3 = False

                if (line.split()[1].replace('.', '', 1).isdigit()) or ('*' in line.split()[1]):
                    # read output type 4
                    if o_type4:
                        # define axial location
                        fuel_no = int(line.split()[0])
                        # save header name of each variable
                        key_name = 'fuel_rod' + str(fuel_no)

                        # line filtering due to the existence of '*' in the output
                        if '*' in line:
                            line_filtered = line.replace('*', "")
                        else:
                            line_filtered = line

                        # print(line_filtered)
                        for j in range(2, len(line_filtered.split())):
                            var_no += 1
                            # skip the non-numeric elements
                            if j != 5:
                                DictArray[timestep_cnt - 1,
                                          var_no] = line_filtered.split()[j]

                            if header_w:
                                # print(header_w)
                                if j == 2:
                                    header_name = key_name + '_fluidTemperatureLiquid'
                                    header.append(header_name)
                                elif j == 3:
                                    header_name = key_name + '_fluidTemperatureVapor'
                                    header.append(header_name)
                                elif j == 4:
                                    header_name = key_name + '_surfaceHeatflux'
                                    header.append(header_name)
                                elif j == 5:
                                    header_name = key_name + '_heatTransferMode'
                                    header.append(header_name)
                                elif j == 6:
                                    header_name = key_name + '_caldOutTemperature'
                                    header.append(header_name)
                                elif j == 7:
                                    header_name = key_name + '_cladInTemperature'
                                    header.append(header_name)
                                elif j == 8:
                                    header_name = key_name + '_gapConductance'
                                    header.append(header_name)
                                elif j == 9:
                                    header_name = key_name + '_fuelTemperatureSurface'
                                    header.append(header_name)
                                elif j == 10:
                                    header_name = key_name + '_fuelTemperatureCenter'
                                    header.append(header_name)
                                else:
                                    raise IOError(
                                        "Error: Unexpected output file format. Check the oufput file (output type 4, fuel rod).")

                        # at end of rod info, reset o_type3 booleans
                        if fuel_no == 1:
                            o_type4 = False

        return DictArray, header
