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
  Simulated Annealing class for global optimization.

  Created 2020-02
  @author: Mohammad Abdo
"""
import numpy as np
import math
import matplotlib.pyplot as plt

class simulatedAnnealing():
  """
  This class performs simulated annealing optimization
  @ In, currentPoint,
  @ In, onjectiveFunction,
  @ In, acceptanceCriterion,
  @ In, temperature,
  @ In, cooling schedule,
  @ In, maxIter,
  @ Out, xOpt,
  """
  def __init__(self, currentPoint=None, lb=None, ub=None, objectiveFunction=None, acceptanceCriterion=None, temperature=None, coolingSchedule=None, maxIter=10000):
    #super().__init__()
    self._currentPoint = currentPoint
    self._objectiveFunction = objectiveFunction
    self._acceptanceCriterion = acceptanceCriterion
    self._temperature = temperature
    self._coolingSchedule = coolingSchedule
    self._maxIter = maxIter
    self._lb = lb
    self._ub = ub

  def __str__(self):
    return  "Initial Guess: " + self._currentPoint + "\nObjective Function: " + self._objectiveFunction + "\nAcceptence Criterion: " + self._acceptanceCriterion + "\nTemperature: " + self._temperature + "\nCooling Schedule: " + self._collingSchedule + "\n Max. Number of iterations: " + self._maxIter    

  def __repr__(self):
    # This is for debugging puposes for developers not users
    # if o is the class instance then:
    # o == eval(__repr__(o)) should be true
    return "simulatedAnnealing('" + str(self._initialGuess) + "', " +  self._objectiveFunction + "', " + self._acceptanceCriterion + "', " + self._temperatue + "', " + self._coolingSchedule + "', " + self._maxIterations +")"
  
  @property
  def currentPoint(self):
    return self._currentPoint

  @currentPoint.setter
  def currentPoint(self, x0):
    self._currentPoint = x0
    
  @property
  def lb(self):
    return self._lb

  @lb.setter
  def lb(self, lb):
    self._lb = lb
    
  @property
  def ub(self):
    return self._ub

  @ub.setter
  def ub(self, ub):
    self._ub = ub      

  ##################
  # Static Methods #
  ##################
  @staticmethod
  def funcname(parameter_list):
    pass

  #################
  # Class Methods #
  #################
  @classmethod
  def funcname(parameter_list):
    pass

  ###########
  # Methods #
  ###########
  def objectiveFunction(self,x,model):
    """
    Method to compute the black-box objective function
    In the literature of simulated annealing
    the cose function (objective function) is often called
    System energy E(x)
    @ In, x, vector, design vector (configuaration in SA literature)
    @ Out, E(x), scalar, energy at x 
    """   
    #return self.E(x)
    return model(x)

  def acceptanceCriterion(self, oldObjective, newObjective, T, kB=1.380657799e-23):
    if newObjective < oldObjective:
      return 1
    else:
      deltaE = (newObjective - oldObjective)
      prob = np.exp(-deltaE/(kB*T))
      return prob  

  def temperature(self, fraction):
    return max(0.01,min(1,1-fraction))

  def coolingSchedule(self, iter, T0, type='Geometric', alpha = 0.94, beta = 0.1,d=10):
    if type == 'linear':
      return T0 - iter * beta
    elif type == 'Geometric':
      return alpha ** iter * T0
    elif type == 'Logarithmic':
      return T0/(np.log10(iter + d))
    # elif type == 'exponential':
    #   return  (T1/T0) ** iter * T0 # Add T1  
    else:
      raise NotImplementedError('Type not implemented.')
  

  def nextNeighbour(self, x, lb, ub, fraction=1):
    """ Pertrub x to find the next random neighbour"""
    nonNormalizedAmp = ((fraction)**-1) / 10 
    amp = (ub - lb) * ((fraction)**-1) / 10
    nonDelta = (-nonNormalizedAmp/2)+nonNormalizedAmp*np.random.rand(len(x))
    delta = (-amp/2)+amp*np.random.rand(len(x))
    xnew = x + delta
    for i in range(len(xnew)):
      xnew[i] = max(min(xnew[i],ub),lb)
    return xnew

  def anneal(self,model,Tol):
    state = self._currentPoint
    cost = self.objectiveFunction(state,model)
    states, costs = [state], [cost]
    for step in range(1,self._maxIter):
      if cost <= Tol:
        continue
      fraction = step / float(self._maxIter)
      #T0 = self.temperature(fraction)
      T0 = 1e4
      T = self.coolingSchedule(step, T0, type='Geometric', alpha = 0.94, beta = 0.1,d=10)
      new_state = self.nextNeighbour(state,self.lb,self.ub,fraction)
      new_cost = self.objectiveFunction(new_state,model)
      if self.acceptanceCriterion(cost, new_cost, T,1) > np.random.rand():
        state, cost = new_state, new_cost
        states.append(state)
        costs.append(cost)
        print("  ==> Accepted step {}, temperature {}, obj: {}!".format(step,T,cost))
      else:
        print("  ==> Rejected step {}, temperature {}, obj: {}!".format(step,T,cost))
    return state, self.objectiveFunction(state,model), states, costs  

  ##############
  # Destructor #      
  ##############
  def __del__(self):
    print('simulatedAnnealing() has been destroyed')

def E(x):
  """
  $$(\vec(x)-5)^{T}\vec(x)-5$$
  """
  return (x-5) @ (x-5)

def Q(x):
  x1 = x[0]
  x2 = x[1]
  obj = 0.2 + x1**2 + x2**2 - 0.1*math.cos(6.0*3.1415*x1) - 0.1*math.cos(6.0*3.1415*x2)
  return obj

def beale(x):
  return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]*x[1])**2 + (2.625 - x[0] + x[0]*x[1]*x[1]*x[1])**2

if __name__ == "__main__":
  S1 = simulatedAnnealing()
  d = 2
  S1.currentPoint = [-2,-2]#np.random.randn(d)#np.array([0.5,1.5])
  lb,ub = -4.5,4.5
  S1.lb,S1.ub = lb,ub
  model = beale
  Tol = 1e-8
  S1._maxIter = 1000
  state, obj, states, costs = S1.anneal(model,Tol)
  
  #state, obj, states, costs = simulatedAnnealing(np.array([0.5,1.5]),lb,ub,objectiveFunction = model(np.array([0.5,1.5]))).anneal(model,Tol)
  print(state,obj)
  hist = np.array(states).reshape(-1,d)
  
  # Create a contour plot
  plt.figure()
  # Specify contour lines
  #lines = range(0,int(max(costs)),5)
  # Plot contours
  # Start location
  x_start = hist[0,:]

  # Design variables at mesh points
  i1 = np.arange(lb, ub, 0.01)
  i2 = np.arange(lb, ub, 0.01)
  x1m, x2m = np.meshgrid(i1, i2)
  costm = np.zeros(x1m.shape)
  for i in range(x1m.shape[0]):
      for j in range(x1m.shape[1]):
          xm = np.array([x1m[i][j],x2m[i][j]])
          costm[i][j] = model(xm)
  CS = plt.contour(x1m, x2m, costm)#,lines)
  # Label contours
  plt.clabel(CS, inline=1, fontsize=10)
  # Add some text to the plot
  plt.title('Contour Plot for Objective Functions')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.plot(hist[:,0],hist[:,1],'m-x')
  plt.grid()
  plt.savefig('contourPlot.png')
  plt.show()

  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  ax1.semilogy(costs,'r.-')
  ax1.legend(['Objective'])
  ax1.set_xlabel('# iterations')
  ax1.grid()
  ax2 = fig.add_subplot(212)
  ax2.plot(hist[:,0],'r.')
  ax2.plot(hist[:,1],'b-')
  ax2.legend(['x1','x2'])
  ax2.set_xlabel('# iterations')
  ax2.grid()
  plt.show()
  # Save the figure as a PNG
  plt.savefig('iterationHistory.png')
