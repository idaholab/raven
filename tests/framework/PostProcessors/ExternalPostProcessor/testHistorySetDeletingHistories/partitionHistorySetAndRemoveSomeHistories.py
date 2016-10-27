
import numpy as np
import math
# this test shows how to perform a partitioning of an historySet removing some histories
# since we are changing the structure of the historySet we need to acustom also the input variables that are not touched by this functions

def time(self):
  newTime = []
  x0      = [] # since we are changing the structure of the historySet we need to acustom also the input variables that are not touched by this functions
  y0      = [] # since we are changing the structure of the historySet we need to acustom also the input variables that are not touched by this functions
  z0      = [] # since we are changing the structure of the historySet we need to acustom also the input variables that are not touched by this functions
  for history in range(len(self.time)):
    for ts in range(len(self.time[history])):
      if self.time[history][ts] >= 0.001:
        break
    if history >1:
      # just to show how to skip a history, we skip the first two ones
      newTime.append(self.time[history][ts:])
      x0.append(self.x0[history])
      y0.append(self.y0[history])
      z0.append(self.z0[history])
  self.x0 = x0
  self.y0 = y0
  self.z0 = z0
  return newTime

def x(self):
  newX = []
  for history in range(len(self.time)):
    for ts in range(len(self.time[history])):
      if self.time[history][ts] >= 0.001:
        break
    if history >1:
      # just to show how to skip a history, we skip the first two ones
      newX.append(self.x[history][ts:])
  return newX

def y(self):
  newY = []
  for history in range(len(self.time)):
    for ts in range(len(self.time[history])):
      if self.time[history][ts] >= 0.001:
        break
    if history >1:
      # just to show how to skip a history, we skip the first two ones
      newY.append(self.y[history][ts:])
  return newY

def z(self):
  newZ = []
  for history in range(len(self.time)):
    for ts in range(len(self.time[history])):
      if self.time[history][ts] >= 0.001:
        break
    if history >1:
      # just to show how to skip a history, we skip the first two ones
      newZ.append(self.z[history][ts:])
  return newZ

