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
# Simulates time-dependent track of a projectile through the air from start to 0,
#     assuming no air resistance.
#     Inputs:
#       (x0,y0) - initial position
#       v0 - initial total velocity
#       ang - angle of initial motion, in degrees, with respect to flat ground
#     Outputs:
#       (x,y) - vector positions of projectile in time
#       t - corresponding time steps
#
import numpy as np

def prange(v,th,y0=0,g=9.8):
  """
    Calculates the analytic range.
    @ In, v, float, velocity of the projectile
    @ In, th, float, angle to the ground for initial projectile motion
    @ In, y0, float, optional, initial height of projectile
    @ In, g, float, optional, gravitational constant (m/s/s)
    @ Out, prange, float, range
  """
  return v*np.cos(th)/g * (v*np.sin(th) + np.sqrt(v*v*np.sin(th)**2+2.*g*y0))

def timeToGround(v,th,y0=0,g=9.8):
  """
    Calculates the analytic time of flight
    @ In, v, float, velocity of the projectile
    @ In, th, float, angle to the ground for initial projectile motion
    @ In, y0, float, optional, initial height of projectile
    @ In, g, float, optional, gravitational constant (m/s/s)
    @ Out, timeToGround, float, time projectile is above the ground
  """
  return v*np.sin(th)/g + np.sqrt(v*v*np.sin(th)**2+2.*g*y0)/g

def xPosition(x0,v,t):
  """
    Calculates the x position in time
    @ In, x0, float, initial horizontal position
    @ In, v, float, velocity of the projectile
    @ In, t, float, time of flight
    @ Out, xPosition, float, horizontal position
  """
  return x0 + v*t

def yPosition(y0,v,t):
  """
    Calculates the analytic vertical position in time
    @ In, y0, float, initial vertical position
    @ In, v, float, velocity of the projectile
    @ In, t, float, time of flight
    @ Out, yPosition, float, vertical position
  """
  return y0 + v*t - 4.9*t*t

def run(self,Input):
  """
    Method require by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  x0 = Input.get('x0',0.0)
  y0 = Input.get('y0',0.0)
  v0 = Input.get('v0',1.0)
  ang = Input.get('angle',45.)*np.pi/180.
  timeOption = Input.get('timeOption', 1)
  self.x0 = x0
  self.y0 = y0
  self.v0 = v0
  self.ang = ang
  if timeOption == 0:
    ts = np.linspace(0,1,10)
  else:
    ts = np.linspace(0,timeToGround(v0,ang,y0),20)

  vx0 = np.cos(ang)*v0
  vy0 = np.sin(ang)*v0
  r = prange(v0,ang,y0)

  self.x = np.zeros(len(ts))
  self.y = np.zeros(len(ts))
  self.r = np.zeros(len(ts))
  for i,t in enumerate(ts):
    self.x[i] = xPosition(x0,vx0,t)
    self.y[i] = yPosition(y0,vy0,t)
    self.r[i] = r
  self.t = ts

#can be used as a code as well
if __name__=="__main__":
  import sys
  print 'Welcome to RAVEN\'s Simple Projectile Motion Simulator!'
  inFile = sys.argv[sys.argv.index('-i')+1]
  print 'Reading input from',inFile,'...'
  outFile = sys.argv[sys.argv.index('-o')+1]
  #construct the input
  Input = {}
  for line in open(inFile,'r'):
    if line.startswith('#'):
      continue
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
  print 'Simulating ...'
  run(io,Input)
  #write output
  print 'Writing output to',outFile+'.csv','...'
  outFile = open(outFile+'.csv','w')
  outFile.writelines('x0,y0,v0,ang,r,x,y,t\n')
  inpstr = ','.join(str(i) for i in (io.x0,io.y0,io.v0,io.ang))
  for i in range(len(io.t)):
    outFile.writelines(inpstr+',%f,%f,%f,%f\n' %(io.r[i],io.x[i],io.y[i],io.t[i]))
  outFile.close()
  print 'Done!'
  print ''
