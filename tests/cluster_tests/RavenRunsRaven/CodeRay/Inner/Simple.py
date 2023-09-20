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
import sys
import argparse
import configparser
import time

def checkAux():
  """
    Checks for aux file
    @ In, None
    @ Out, None
  """
  try:
    open('simple.aux', 'r')
  except FileNotFoundError:
    raise RuntimeError('Aux file not found prior to running!')

def getInput():
  """
    Collects input from parser
    @ In, None
    @ Out, args, argparse.parser, dict-like argument interface
  """
  parser = argparse.ArgumentParser(description='RrR Test Code')
  parser.add_argument('-i', type=str, help='main input file')
  args = parser.parse_args()
  return args

def readInput(infile):
  """
    reads input from file
    @ In, infile, string, filename
    @ Out, (a, b, x, y, out), tuple, required inputs
  """
  config = configparser.ConfigParser()
  config.read(infile)
  a = float(config['FromOuter']['a'])
  b = float(config['FromOuter']['b'])
  x = float(config['FromInner']['x'])
  y = float(config['FromInner']['y'])
  out = config['Output']['output']
  return a, b, x, y, out

def run(*args):
  """
    main run method.
    @ In, args, list, list of things to sum up
    @ Out, run, float, sum of entries in list
  """
  return sum(args)

def write(a, b, c, x, y, out):
  """
    Write inputs and outputs to file
    @ In, a, float, float
    @ In, b, float, float
    @ In, c, float, float
    @ In, x, float, float
    @ In, y, float, float
    @ In, out, string, filename to write results to
  """
  print('Writing to', out, time.ctime())
  with open(out, 'w') as f:
    f.writelines(','.join('abcxy') + '\n')
    f.writelines(','.join(str(i) for i in [a, b, c, x, y]) + '\n')

if __name__ == '__main__':
  checkAux()
  args = getInput()
  infileName = args.i
  a, b, x, y, out = readInput(infileName)
  c = run(a, b, x, y)
  write(a, b, c, x, y, out)
  print("Goodbye", time.ctime())
