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
#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Determines diamater of impact crater in dry sand given a small object near Earth's surface
#     Inputs:
#       m - object mass
#       E - kinetic energy of object
#       r - radius of object
#     Outputs:
#       D - diameter of crater
#
import numpy as np

def run(self,Input):
  """
    Method require by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  m = Input.get('m',0.5)
  E = Input.get('E',25)
  r = Input.get('r',0.2)
  self.m = m
  self.E = E
  self.r = r
  self.D = 4.4e-3 * (E*4.*m/(3.*np.pi*r*r*r))**(1./3.4)

#can be used as a code as well
if __name__=="__main__":
  import sys
  inFile = sys.argv[sys.argv.index('-i')+1]
  outFile = sys.argv[sys.argv.index('-o')+1]
  #construct the input
  Input = {}
  for line in open(inFile,'r'):
    arg,val = (a.strip() for a in line.split('='))
    Input[arg] = float(val)
  #make a dummy class to hold values
  class IO:
    """
      Dummy class to hold values like RAVEN does
    """
    pass
  io = IO()
  #run the code
  run(io,Input)
  #write output
  outFile = open(outFile+'.csv','w')
  outFile.writelines('m,r,E,D\n')
  inpstr = ','.join('{}' for _ in range(4)).format(io.m,io.r,io.E,io.D)
  outFile.writelines(inpstr+'\n')
  outFile.close()
