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
import time

def evalAndWrite(x, y, case):
  dat=[]
  c = 0
  outf = open(case+'.csv','w')
  outf.writelines('step,i,j,x,y,poly\n')
  outf.close()
  for i in [0.3,0.5,0.7,1.0]:
    for j in [1.3,1.5,1.7,2.0]:
      # open in append mode
      outf = open(case+'.csv','a')
      # we sleep to make sure that we dont finish
      # too quick (so we stoppingCriteriaFunction has time
      # to check the output to stop the simulation)
      c+=1
      row = [c,i,j,x,y,(i-x)*(j-y)]
      outf.writelines(','.join(str(i) for i in row)+'\n')
      outf.close()
      time.sleep(0.5)
        

def run(xin):
  inx = open(xin,'r')
  for line in inx:
    if   line.startswith('x ='      ):
      x=float(line.split('=')[1])
    elif line.startswith('case ='   ):
      case=line.split('=')[1].strip()
    elif line.startswith('auxfile ='):
      aux=line.split('=')[1].strip()
  iny = open(aux,'r')
  for line in iny:
    if line.startswith('y ='):
      y=float(line.split('=')[1])

  evalAndWrite(x, y, case)

if __name__=='__main__':
  import sys
  args = sys.argv
  inp1 = args[args.index('-i')+1] if '-i' in args else None
  run(inp1)
