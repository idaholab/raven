import numpy as np

def run(self,Input):
  v0 = self.v0
  y0 = self.y0
  ang = 45.*np.pi/180.
  times = np.linspace(0,2,5)
  mode = self.mode
  if mode == 'stepper':
    y = stepper(v0,y0,ang,times)
  elif mode == 'analytic':
    y = analytic(v0,y0,ang,times)
  else:
    raise IOError('Unrecognized mode:',mode)
  self.y = np.atleast_1d(y)
  self.t = np.atleast_1d(times)
  self.restartID = np.array([2]*len(times))

def analytic(v0,y0,ang,times):
  ys = []
  # initial y velocity
  vy0 = v0[0] * np.sin(ang)
  for t in times:
    # calculate analytic height
    y = y0[0] + vy0*t - 0.5*9.8*t*t
    # calculate analytic velocity magnitude
    v = np.sqrt(v0[0]*v0[0] + 2.0*(9.8)*(y0[0]-y))
    ys.append(y)
  return ys

def stepper(v0,y0,ang,times):
  # initial x position
  x = 0.0
  y = y0[0]
  # initial x,y velocity
  vx = v0 * np.cos(ang)
  vy = v0 * np.sin(ang)
  dt = times[1] - times[0]
  # tracker
  ys = []
  for _ in times:
    # store current values
    #v = np.sqrt(vx*vx + vy*vy)
    ys.append(y)
    # update velocity
    vx = vx
    vy = vy - 9.8*dt
    # update position
    x = x + vx*dt
    y = y + vy*dt
  return ys

class data:
  def __init__(self,v0,y0,mode):
    self.v0 = v0
    self.y0 = y0
    self.mode = mode
    self.y = None
    self.t = None

if __name__=='__main__':
  import matplotlib.pyplot as plt
  # initialize
  y0 = 1.0
  v0 = 15.0
  # test
  rdata = {}
  for mode in ['stepper','analytic']:
    # set up input class
    dat = data(v0,y0,mode)
    rdata[mode] = dat
    # run
    run(dat,None)
    # plot
    plt.plot(dat.t,dat.y,'-o',label=mode)
  plt.legend(loc=0)
  plt.xlabel('time')
  plt.ylabel('height')
  plt.show()



