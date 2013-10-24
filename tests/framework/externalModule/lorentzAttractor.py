
''' from wikipedia: dx/dt = sigma*(y-x)  ; dy/dt = x*(rho-z)-y  dz/dt = x*y-beta*z  ; '''

import numpy as np
#import pylab as pyl
#import random
#import mpl_toolkits.mplot3d.axes3d as p3

def initialize(self,runInfoDict,inputFiles):
  self.SampledVars = None
  
  self.sigma = 10.0
  self.rho   = 28.0
  self.beta  = 8/3
  
    #self.fig=pyl.figure()
    #self.ax = p3.Axes3D(self.fig)
    #self.ax.set_xlabel('X')
    #self.ax.set_ylabel('Y')
    #self.ax.set_zlabel('Z')
    #self.fig.add_axes(self.ax)
  
  return


def createNewInput(self,myInput,samplerType,**Kwargs):
  self.SampledVars = Kwargs['SampledVars']
  return None

def readMoreXML(self,xmlNode):
  return None

def run(self,Input,jobHandler):

  max_time = 0.03
  t_step = 0.01
  
  numberTimeSteps = int(max_time/t_step)
  
  self.x = np.zeros(numberTimeSteps)
  self.y = np.zeros(numberTimeSteps)
  self.z = np.zeros(numberTimeSteps)
  self.time = np.zeros(numberTimeSteps)
  
  self.x0 = self.SampledVars['x0'] 
  self.y0 = self.SampledVars['y0']  
  self.z0 = self.SampledVars['z0'] 
  
  self.x[0] = self.SampledVars['x0'] 
  self.y[0] = self.SampledVars['y0']  
  self.z[0] = self.SampledVars['z0'] 
  self.time[0]= 0
  
  for t in range (numberTimeSteps-1):
    self.time[t+1] = self.time[t] + t_step
    self.x[t+1]    = self.x[t] + self.sigma*(self.y[t]-self.x[t]) * t_step
    self.y[t+1]    = self.y[t] + (self.x[t]*(self.rho-self.z[t])-self.y[t]) * t_step
    self.z[t+1]    = self.z[t] + (self.x[t]*self.y[t]-self.beta*self.z[t]) * t_step
    
    #self.ax.plot3D(self.x,self.y,self.z)

    #pyl.savefig('test.png')


