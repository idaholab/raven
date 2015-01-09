'''
Created on 12/23/14

@author: maljdan
'''
import numpy as np
import math

def Delta(self):
  return self.X - self.Y

def Sum(self):
  return self.X + self.Y

def SumAB(A,B):
  return A + B

def DiffAB(A,B):
  return A - B

def Norm(A,B):
  return np.sqrt(A**2 + B**2)
