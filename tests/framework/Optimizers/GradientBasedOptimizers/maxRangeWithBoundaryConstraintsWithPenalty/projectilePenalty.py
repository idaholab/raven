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
"""
import numpy as np

def range(v,th,y0=0,g=9.8):
  """
    This method is aimed to compute the range of a projectile through the air given
    the velocity, angle, elevation and gravity acceleration
    @ In, v,  float, velocity
    @ In, th, float, angle (rad)
    @ In, y0, float, elevation
    @ In, g,  float, gravity
  """
  return v*np.cos(th)/g * (v*np.sin(th)*np.sqrt(v*v*np.sin(th)**2+2.*g*y0))

def run(self,Input):
  x0 = Input.get('x0',0.0)
  y0 = Input.get('y0',0.0)
  v0 = Input.get('v0',1.0)
  ang = Input.get('angle',45.)*np.pi/180.

  ts = np.linspace(0,1,10)

  vx0 = np.cos(ang)*v0
  vy0 = np.sin(ang)*v0
  r = range(v0,ang,y0)
  # ***************************************************************************
  # * this is an example of a penalty function applied on the loss function.  *
  # * the v0 should converge around 40                                        *
  # ***************************************************************************
  if v0 > 40.0:
    r -= 100.0*(v0-40.0)**2
  # ***************************************************************************
  # * end of penalty function                                                 *
  # ***************************************************************************

  self.x = np.zeros(len(ts))
  self.y = np.zeros(len(ts))
  self.r = np.zeros(len(ts))
  for i,t in enumerate(ts):
    self.x[i] = x0 + vx0*t
    self.y[i] = y0 + vy0*t - 4.9*t*t
    self.r[i] = r
  self.time = ts

