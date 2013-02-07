'''
Created on Feb 6, 2013

@author: crisr
'''

class MonteCarloSampler:
  '''
  class running a Monte Carlo sampling
  '''
  def __init__(self,name):
    '''
    Constructor
    '''
    self.name               = name  #name of the instance of the Monte Carlo 
    self.perturbedParamters = []    #list of char with the name of the parameters perturbed
    self.outputNames        = []    #list of char with the name of the phase space variable recorded
    self.save               = ''    #possible option are: timeHistory, History, PointSet, Point
    self.timesnaps          = []    #list of time coordinate for the snaps (if presents)
    self.time               = True  #this is used to identify if the points are in time or in under other criteria

  def run(self):
    '''
    run himself to generate the output data
    '''
    return
  
  def read(self):
    '''
    from the input file reads the information to fill himself
    '''
#    self.outputNames
#    self.save

    return
  def getOutput(self):
    '''
    from the output  file reads the information to fill himself
    '''
    self.output = data
      

 