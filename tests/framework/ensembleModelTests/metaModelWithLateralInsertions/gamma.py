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
  self.E = self.C + 1.0
  print('Gamma is finished '  +str(self.E) + ' ' +str(self.C))

