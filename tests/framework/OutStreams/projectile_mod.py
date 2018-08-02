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

def time_to_ground(v,th,y0=0,g=9.8):
  """
    Calculates the analytic time of flight
    @ In, v, float, velocity of the projectile
    @ In, th, float, angle to the ground for initial projectile motion
    @ In, y0, float, optional, initial height of projectile
    @ In, g, float, optional, gravitational constant (m/s/s)
    @ Out, time_to_ground, float, time projectile is above the ground
  """
  return v*np.sin(th)/g + np.sqrt(v*v*np.sin(th)**2+2.*g*y0)/g

def x_pos(x0,v,t,discvar):
  """
    Calculates the x position in time
    @ In, x0, float, initial horizontal position
    @ In, v, float, velocity of the projectile
    @ In, t, float, time of flight
    @ In, discvar, float, mistery var
    @ Out, x_pos, float, horizontal position
  """
  if discvar < 0.0:
    return x0 + v*t
  else:
    return x0 + v*t

def y_pos(y0,v,t,discvar):
  """
    Calculates the analytic vertical position in time
    @ In, y0, float, initial vertical position
    @ In, v, float, velocity of the projectile
    @ In, t, float, time of flight
    @ In, discvar, float, mistery var
    @ Out, y_pos, float, vertical position
  """
  if discvar < 0.0:
    return y0 + v*t - 4.9*t*t
  else:
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
  self.x0 = x0
  self.y0 = y0
  self.v0 = v0
  self.ang = ang

  #ts = np.linspace(0,1,10)
  ts = np.linspace(0,time_to_ground(v0,ang,y0),50)

  vx0 = np.cos(ang)*v0
  vy0 = np.sin(ang)*v0
  r = prange(v0,ang,y0)

  discVar = Input.get('var1',0.0)
  if discVar < 0.0:
    discVar = -1.0
  else:
    discVar = 1.0

  self.x = np.zeros(len(ts))
  self.y = np.zeros(len(ts))
  self.r = np.zeros(len(ts))
  for i,t in enumerate(ts):
    self.x[i] = x_pos(x0,vx0,t,discVar)
    self.y[i] = y_pos(y0,vy0,t,discVar)
    self.r[i] = r
  self.time = ts