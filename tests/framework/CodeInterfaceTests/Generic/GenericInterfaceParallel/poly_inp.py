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
import numpy as np
import os
import xml.etree.ElementTree as ET

def eval(x,y,z):
  """
    Performs evaluations.
    @ In, x, float, scalar
    @ In, y, float, scalar
    @ In, z, float, scalar
    @ Out, list(float), input values and output value
  """
  dat=[]
  c = 0
  for i in [0.3,0.5,0.7,1.0]:
    for j in [1.3,1.5,1.7,2.0]:
      c+=1
      dat.append([c,i,j,x,y,z,(i-x)*(j-y)+z])
  return dat

def run(xin,yin,out):
  """
    Running interface for RAVEN.
    @ In, xin, str, filename for input containing x and z
    @ In, yin, str, filename for input containing y
    @ In, out, str, output file base name
    @ Out, None
  """
  inx = ET.parse(xin)
  root = inx.getroot()
  iny = file(yin,'r')
  if not os.path.isfile('dummy.e'):
    raise IOError('Missing dummy exodus file "dummy.e"!')
  x = float(root.find('x').text)
  z = float(root.find('z').text)
  for line in iny:
    if line.startswith('y ='):
      y=float(line.split('=')[1])

  dat = eval(x,y,z)

  outf = file(out+'.csv','w')
  outf.writelines('step,i,j,x,y,z,poly\n')
  for e in dat:
    outf.writelines(','.join(str(i) for i in e)+'\n')
  outf.close()

if __name__=='__main__':
  import sys
  args = sys.argv
  inp1 = args[args.index('-i')+1] if '-i' in args else None
  inp2 = args[args.index('-a')+1] if '-a' in args else None
  out  = args[args.index('-o')+1] if '-o' in args else None
  run(inp1,inp2,out)
