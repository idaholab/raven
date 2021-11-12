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
from __future__ import division, print_function, absolute_import
import csv
import sys
import argparse

# Build Parser
parser = argparse.ArgumentParser(description="Reads several variables from input and performs simple calculation to produce output.")
parser.add_argument('-i', dest='input_file', required=True,
  help='Input file with variables a, b, c, d, and e', metavar='INFILE')
parser.add_argument('-o', dest='output_file', required=True,
  help='Output file name without .csv extension', metavar = 'OUTFILE')
args = parser.parse_args()

# Read values from INFILE
with open(args.input_file) as rf:
  indata = rf.readlines()
  for line in indata:
    # parse input, assign values to variables
    if line.strip().startswith("#"):
      continue
    variable, value = line.split("=")
    exec('%s = %f' % (variable.strip(),float(value.strip())))
rf.close()

# Calculation
f = a*b
g = c/5 + d/3
h = g + a * e

# Print to csv file
out_file_name = args.output_file + '.csv'
print("Output will be printed to", out_file_name)
if sys.version_info[0] > 2:
  wf = open(out_file_name, 'w', newline='')
else:
  wf = open(out_file_name, 'wb')
writer = csv.writer(wf, delimiter=',')
var_name = ['a','b','c','d','e','f','g','h']
data = [a,b,c,d,e,f,g,h]
writer.writerow(var_name)
writer.writerow(data)
wf.close()
