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
import numpy as np

def readInput(inFile):
  """
    Reads in the key values.
    @ In, inFile, Python file object, file containing inputs
    @ Out, (x,y,z), tuple(float,float,float), input values
  """
  x, y, z = 0,0,0
  for line in inFile:
    var,val = line.strip().split('=')
    if   var.strip() == 'x':
      x = float(val)
    elif var.strip() == 'y':
      y = float(val)
    elif var.strip() == 'z':
      z = float(val)
  if x is None or y is None or z is None:
    raise IOError('x,y,z were not all found in input file',inFile)
  return x,y,z

def run(x,y,z):
  """
    Performs elementary calculations.
    For testing purposes, number of time steps is dependent on x value.
    @ In, x, float, value
    @ In, y, float, value
    @ In, z, float, value
    @ Out, a, float, value
    @ Out, b, float, value
    @ Out, c, float, value
  """
  xs=[]
  ys=[]
  zs=[]
  ts=[]
  a=[]
  b=[]
  c=[]
  if -0.33 > x:
    numSteps = 5
    dt = 0.1
  elif 0.33 > x:
    numSteps = 4
    dt = 0.11
  else:
    numSteps = 3
    dt = 0.3
  for t in range(numSteps):
    ts.append(dt*t)
    xs.append(x + t*dt)
    ys.append(y + t*dt)
    zs.append(z + t*dt)
    a.append(xs[-1]*(ys[-1]+zs[-1]))
    b.append(ys[-1]*(xs[-1]+zs[-1]))
    c.append(zs[-1]*(xs[-1]+ys[-1]))
  return xs,ys,zs,a,b,c,ts

def write(xs,ys,zs,ts,a,b,c,outname):
  """
    Writes to a CSV file, but intentionally truncates it
    @ In, x, float, value
    @ In, y, float, value
    @ In, z, float, value
    @ In, a, float, value
    @ In, b, float, value
    @ In, c, float, value
    @ In, Outname, string, name of output file
    @ Out, None
  """
  out = open(outname+'.csv','w')
  out.writelines('t,x,y,z,xt,yt,zt,a,b,c\n')
  for i in range(len(ts)):
    out.writelines(','.join('%1.15e' %s for s in [ts[i],xs[0],ys[0],zs[0],xs[i],ys[i],zs[i],a[i],b[i],c[i]])+'\n')
  out.close()

if __name__ == '__main__':
  if '-i' not in sys.argv:
    raise IOError('No input file was specified with "-i"!')
  if '-o' not in sys.argv:
    raise IOError('No output file was specified with "-o"!')
  if len(sys.argv)<5:
    raise IOError('Insufficient arguments! Need -i Input -o Output')
  inFile = open(sys.argv[2],'r')
  x,y,z = readInput(inFile)
  xs,ys,zs,a,b,c,ts = run(x,y,z)
  write(xs,ys,zs,ts,a,b,c,sys.argv[4])
