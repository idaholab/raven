# Function designed by D. Maljovec to have sinusoidally-shaped minimum region
#
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(0,0) = 0
# parameter range is -4.5 <= x,y <= 4.5
import numpy as np

def evaluate(x,y):
  return (np.cos(0.7*(x+y))-(y-x))**2 + 0.1*(x+y)**2

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

def precond(y):
  self.x = y

