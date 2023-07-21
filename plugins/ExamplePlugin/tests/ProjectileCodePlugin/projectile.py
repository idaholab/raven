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
Note: Exact duplicate of raven/doc/workshop/ExternalModels/projectile.py
Used to demonstrate the Code Plugin system
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
import numpy as np

in_vars = ['x0', 'y0', 'v0', 'ang', 'timeOption']
out_vars = ['x', 'y', 'r', 't', 'v', 'a']

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

def x_pos(x0,v,t):
  """
    Calculates the x position in time
    @ In, x0, float, initial horizontal position
    @ In, v, float, velocity of the projectile
    @ In, t, float, time of flight
    @ Out, x_pos, float, horizontal position
  """
  return x0 + v*t

def y_pos(y0,v,t):
  """
    Calculates the analytic vertical position in time
    @ In, y0, float, initial vertical position
    @ In, v, float, velocity of the projectile
    @ In, t, float, time of flight
    @ Out, y_pos, float, vertical position
  """
  return y0 + v*t - 4.9*t*t

def calc_vel(y0, y, v0, ang, g=9.8):
  E_m = 0.5 * v0*v0 + g*y0
  vel = np.sqrt(v0*v0 - 2*g*(y-y0))
  x_vel = v0 * np.cos(ang)
  y_vel = np.sqrt(vel*vel - x_vel*x_vel)
  return x_vel, y_vel, vel

def current_angle(v0, ang, vel):
  return np.arccos(v0 * np.cos(ang) / vel)

def run(raven, inputs):
  vars = {'x0': get_from_raven('x0', raven, 0),
          'y0': get_from_raven('y0', raven, 0),
          'v0': get_from_raven('v0', raven, 1),
          'angle': get_from_raven('angle', raven, 45),
          'timeOption': get_from_raven('timeOption', raven, 0)}
  res = main(vars)
  raven.x = res['x']
  raven.y = res['y']
  raven.t = res['t']
  raven.r = res['r'] * np.ones(len(raven.x))
  raven.v = res['v']
  raven.a = res['a']

def get_from_raven(attr, raven, default=None):
  return np.squeeze(getattr(raven, attr, default))

def main(Input):
  x0 = Input.get('x0', 0)
  y0 = Input.get('y0', 0)
  v0 = Input.get('v0', 1)
  ang = Input.get('angle', 45)
  g = Input.get('g', 9.8)
  timeOption = Input.get('timeOption', 0)
  ang = ang * np.pi / 180
  if timeOption == 0:
    ts = np.linspace(0,1,10)
  else:
    # due to numpy library update, the return shape of np.linspace
    # is changed when an array-like input is provided, i.e. return from time_to_ground
    ts = np.linspace(0,time_to_ground(v0,ang,y0),10)

  vx0 = np.cos(ang)*v0
  vy0 = np.sin(ang)*v0
  r = prange(v0,ang,y0)

  x = np.zeros(len(ts))
  y = np.zeros(len(ts))
  v = np.zeros(len(ts))
  a = np.zeros(len(ts))
  for i,t in enumerate(ts):
    x[i] = x_pos(x0,vx0,t)
    y[i] = y_pos(y0,vy0,t)
    vx, vy, vm = calc_vel(y0, y[i], v0, ang, g)
    v[i] = vm
    a[i] = current_angle(v0, ang, vm)
  t = ts
  res = {'x': x, 'y': y, 'r': r, 't': ts, 'v': v, 'a': a,
         'x0': x0, 'y0': y0, 'v0': v0, 'ang': ang, 'timeOption': timeOption}
  return res

#can be used as a code as well
if __name__=="__main__":
  import sys
  textOutput = False
  if '-i' not in sys.argv:
    raise IOError("INPUT MUST BE PROVIDED WITH THE -i nomenclature")
  if '-o' not in sys.argv:
    raise IOError("OUTPUT MUST BE PROVIDED WITH THE -o nomenclature")
  if '-text' in sys.argv:
    textOutput = True
  inFile = sys.argv[sys.argv.index('-i')+1]
  outFile = sys.argv[sys.argv.index('-o')+1]
  #construct the input
  Input = {}
  for line in open(inFile,'r'):
    arg, val = (a.strip() for a in line.split('='))
    Input[arg] = float(val)
  #run the code
  res = main(Input)
  #write output
  outName = outFile+ ('.txt' if textOutput else '.csv')
  delm = ' ' if textOutput else ','
  with open(outName, 'w') as outFile:
    outFile.writelines(delm.join(in_vars) + delm + delm.join(out_vars) + '\n')
    template = delm.join('{{}}'.format(v) for v in in_vars + out_vars) + '\n'
    for i in range(len(res['t'])):
      this = [(res[v][i] if len(np.shape(res[v])) else res[v]) for v in in_vars + out_vars]
      outFile.writelines(template.format(*this))
    if textOutput:
      outFile.write('---------------------------------------------------------------------------\n')
      outFile.write('SUCCESS\n')
  print('Wrote results to "{}"'.format(outName))
