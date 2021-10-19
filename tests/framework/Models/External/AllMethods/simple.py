import numpy as np
import math

def minimum_value(c1,c2,x1,x2):
  """
    Calculates minimum value for function c1*x1+c2*x2 based on constraints.
    @ In, c1, float, coefficeint of x1
    @ In, x1, float, function value 1
    @ In, c2, float, coefficient of x2
    @ In, x2, float, function value 2
    @ Out, float, minimum value
  """
  return np.float(c1*x1+c2*x2)

def constraint_value(a1,a2,x1,x2):
  """
    Calculates canstraint value for function x1 and x2.
    @ In, a1, float, coefficeint of x1
    @ In, x1, float, function value 1
    @ In, a2, float, coefficient of x2
    @ In, x2, float, function value 2
    @ Out, float, constraint value
  """
  return a1*x1+a2*x2

def test_constraint(constraint,b):
  """
    Tests if the selection of x1 and x2 satisfies the constraints set by "b".
    @ In, constraint, float, constraint value
    @ In, b, float, contraint parameter
    @ Out, logic, decision for validity
  """
  if constraint >= b:
    test_condition = True
  else:
    test_condition = False
  return test_condition

def run(self,Input): # Input as dict
  constraint = constraint_value(self.a1,self.a2,self.x1,self.x2)
  test = test_constraint(constraint,self.b)
  if test is True and self.x1 >= 0 and self.x2 >= 0:
    self.ans = minimum_value(self.c1,self.c2,self.x1,self.x2)
  else:
    self.ans = np.float(10**6)
