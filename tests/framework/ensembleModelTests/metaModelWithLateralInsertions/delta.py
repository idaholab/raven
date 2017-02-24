import numpy as np
import copy
import time
# CAUTION HERE #
# IT IS IMPORTANT THAT THE USER IS AWARE THAT THE EXTERNAL MODEL (RUN METHOD)
# NEEDS TO BE THREAD-SAFE!!!!!
# IN CASE IT IS NOT, THE USER NEEDS TO SET A LOCK SYSTEM!!!!!
import threading
localLock = threading.RLock()

def initialize(self, runInfo, inputs):
  pass

def run(self, Input):
  self.F = self.D + 1.0
  print('delta is finished '  +str(self.F) + ' ' +str(self.D))

