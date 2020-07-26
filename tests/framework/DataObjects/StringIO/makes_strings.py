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
  A Code model for testing string outputs
"""

import sys
import numpy as np

def read(file):
  """
    Reads in input file.
    @ In, file, string, file to read
    @ Out, data, dict, input params
  """
  data = {}
  for line in open(inFile,'r'):
    line = line.split('#')[0].strip()
    if line == '':
      continue
    arg, val = (a.strip() for a in line.split('='))
    data[arg] = float(val)
  return data

def main(input):
  """
    Runs the case.
    @ In, input, dict, from input file
    @ Out, data, dict, results
  """
  x0 = input['x0']
  # single string output
  ss = 'lower' if x0 < 0.5 else 'upper'
  # vector string output
  time = np.arange(5)
  st = np.array(list(ss))
  # single float output
  fs = 0 if x0 < 0.5 else 1
  ft = 100 * fs + np.arange(5)
  result = {'x0': x0, 'time': time, 'ss': ss, 'st': st, 'fs': fs, 'ft': ft}
  return result

def write(data, file):
  """
    Runs the case.
    @ In, file, string, file to write to
    @ In, data, dict, results from run
    @ Out, None
  """
  x0 = data['x0']
  time = data['time']
  ss = data['ss']
  st = data['st']
  fs = data['fs']
  ft = data['ft']
  with open(file + '.csv', 'w') as f:
    f.writelines('time,x0,ss,st,fs,ft\n')
    for t in data['time']:
      f.writelines(f'{time[t]},{x0},{ss},{st[t]},{fs},{ft[t]}\n')

#can be used as a code as well
if __name__=="__main__":
  # read arguments
  inFile = sys.argv[sys.argv.index('-i')+1]
  outFile = sys.argv[sys.argv.index('-o')+1]

  input = read(inFile) # construct the input
  res = main(input)    # run the code
  write(res, outFile)  # write the results
