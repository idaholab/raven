
import numpy
import copy
#import pylab as pyl
#import random
#import mpl_toolkits.mplot3d.axes3d as p3

def initialize(self,runInfoDict,inputFiles):
  print('There is snow in my memories ...there is always snow...and my brian becomes white if I do not stop remembering...')
  self.z               = 0
  return

#def createNewInput(self,myInput,samplerType,**Kwargs):
#  return Kwargs['SampledVars']

def run(self,Input):
  #self.z = Input['x0']+Input['y0']
  self.z = self.x0 + self.y0

