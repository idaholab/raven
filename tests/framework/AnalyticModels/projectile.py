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
  self.x0 = x0
  self.y0 = y0
  self.v0 = v0
  self.ang = ang

  ts = np.linspace(0,1,10) #time_to_ground(v0,ang,y0),10)

  vx0 = np.cos(ang)*v0
  vy0 = np.sin(ang)*v0
  r = prange(v0,ang,y0)

  self.x = np.zeros(len(ts))
  self.y = np.zeros(len(ts))
  self.r = np.zeros(len(ts))
  for i,t in enumerate(ts):
    self.x[i] = x_pos(x0,vx0,t)
    self.y[i] = y_pos(y0,vy0,t)
    self.r[i] = r
  self.time = ts

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
    pass
  io = IO()
  #run the code
  run(io,Input)
  #write output
  outFile = open(outFile+'.csv','w')
  outFile.writelines('x0,y0,v0,ang,r,t,x,y\n')
  inpstr = ','.join(str(i) for i in (io.x0,io.y0,io.v0,io.ang))
  for i in range(len(io.time)):
    outFile.writelines(inpstr+',%f,%f,%f,%f\n' %(io.r[i],io.x[i],io.y[i],io.time[i]))
  outFile.close()



