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
""" This example demonstrates how to use the FMU.get*() and FMU.set*() functions
 to set custom input and control the simulation """

from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_test_file
import numpy as np
import shutil
import sys


def simulateCustomInputFMU(fmuFilename,pathToRaven,show_plot=True):

  # define the model name and simulation parameters

  start_time = 0.0
  stop_time = 4.0
  step_size = 1.
  # set inputs
  y1 = [0.0, 0.0, 1.0, 1.0]
  y2 = [0.0, 1.0, 0.0, 1.0]

  # read the model description
  model_description = read_model_description(fmuFilename)
  outputFileName = model_description.coSimulation.modelIdentifier + ".out"
  print(dump(fmuFilename))

  # collect the value references
  vrs = {}
  for variable in model_description.modelVariables:
    vrs[variable.name] = variable.valueReference

  # get the value references for the variables we want to get/set
  vr_y1, vr_y2 = vrs['y1'], vrs['y2']
  vr_outputs = vrs['ans']
  #vr_models  = vrs['model_path']
  vr_paths = vrs['raven_path']
  # extract the FMU
  unzipdir = extract(fmuFilename)
  print("extracted")

  fmu = FMU2Slave(guid=model_description.guid,
              unzipDirectory=unzipdir,
              modelIdentifier=model_description.coSimulation.modelIdentifier,
              instanceName='instance1')

  # initialize
  print("initializing", flush=True)
  sys.path.append(pathToRaven)
  fmu.instantiate(loggingOn=True)
  fmu.setupExperiment(startTime=start_time)
  fmu.enterInitializationMode()
  fmu.exitInitializationMode()
  print("initialized")

  fmu.setString( [vr_paths], [pathToRaven])
  #fmu.setString( [vr_models], [model_path])

  time = start_time

  rows = []  # list to record the results
  cnt = 0
  # simulation loop
  print("Starting loop")
  while time < stop_time:
    # set the input
    fmu.setReal([vr_y1], [y1[cnt]])
    fmu.setReal([vr_y2], [y2[cnt]])
    cnt+=1

    # perform one step
    print("starting step")
    fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    # get the values for 'inputs' and 'outputs[4]'
    in_y1, in_y2, outputs = fmu.getReal([vr_y1, vr_y2, vr_outputs])
    # append the results
    rows.append((time, in_y1, in_y2, outputs))

    # advance the time
    time += step_size

  fmu.terminate()
  fmu.freeInstance()

  # clean up
  shutil.rmtree(unzipdir, ignore_errors=True)

  # convert the results to a structured NumPy array
  result = np.array(rows, dtype=np.dtype([('time', np.float64), ('y1', np.float64), ('y2', np.float64), ('ans', np.float64)]))

  # plot the results
  if show_plot:
    plot_result(result)
  result = np.atleast_2d(result)
  print(result[0],result.shape)
  print("result", result)
  np.savetxt(outputFileName, result[0], delimiter=",")
  #open(outputFileName, "w").write(str(result))
  return time

if __name__ == '__main__':
  import os
  fmuFilename = './FMURom/rom_out.fmu'
  pathToRaven = os.path.abspath(os.sep.join(['..','..','..','..']))
  print("Loading", fmuFilename)
  print("pathToRaven", pathToRaven)
  simulateCustomInputFMU(fmuFilename, pathToRaven, False)
