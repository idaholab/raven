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

def range(v,th,y0=0,g=9.8):
    return v*np.cos(th)/g * (v*np.sin(th)*np.sqrt(v*v*np.sin(th)**2+2.*g*y0))

def time_to_ground(v,th,y0=0,g=9.8):
    return v*np.sin(th)/g + np.sqrt(v*v*np.sin(th)**2+2.*g*y0)/g

def x_pos(x0,v,t):
  return x0 + v*t

def y_pos(y0,v,t):
    return y0 + v*t - 4.9*t*t

def run(self,Input):
  x0 = Input.get('x0',0.0)
  y0 = Input.get('y0',0.0)
  v0 = Input.get('v0',1.0)
  ang = Input.get('angle',45.)*np.pi/180.

  ts = np.linspace(0,1,10) #time_to_ground(v0,ang,y0),10)

  vx0 = np.cos(ang)*v0
  vy0 = np.sin(ang)*v0
  r = range(v0,ang,y0)
  
  # this is an example of a penalty function applied on the loss function.
  # the v0 should converge around 40.
  if v0 > 40.0:
    r -= 100.0*(v0-40.0)**2
  # end of penalty function
  self.x = np.zeros(len(ts))
  self.y = np.zeros(len(ts))
  self.r = np.zeros(len(ts))
  for i,t in enumerate(ts):
    self.x[i] = x_pos(x0,vx0,t)
    self.y[i] = y_pos(y0,vy0,t)
    self.r[i] = r
  self.time = ts

