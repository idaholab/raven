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

def eval(x,y):
  """
    Function to evaluate for given inputs
    @ In, x, float, input value for input variable x
    @ In, y, float, input value for input variable y
    @ Out, data, list, the evaluated values
  """
  data=[]
  c = 0
  for i in [0.3,0.5,0.7,1.0]:
    for j in [1.3,1.5,1.7,2.0]:
      c+=1
      data.append([c,i,j,x,y,np.exp((i-x)*(j-y))])
  return data

def run(inp):
  """
    Method to execute the code
    @ In, inp, str, input file name
    @ Out, None
  """

  with open(inp,'r') as inx:
    for line in inx:
      if line.startswith('x ='):
        x=float(line.split('=')[1])
      elif line.startswith('case ='):
        case=line.split('=')[1].strip()
      elif line.startswith('auxfile ='):
        aux=line.split('=')[1].strip()

  with open(aux,'r') as iny:
    for line in iny:
      if line.startswith('y ='):
        y=float(line.split('=')[1])

  dat = eval(x,y)

  with open(case+'.csv','w') as out:
    out.writelines('step,i,j,x,y,poly\n')
    for e in dat:
      out.writelines(','.join(str(i) for i in e)+'\n')


if __name__=='__main__':
  import sys
  args = sys.argv
  inp1 = args[args.index('-i')+1] if '-i' in args else None
  run(inp1)
