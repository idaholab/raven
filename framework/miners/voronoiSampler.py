'''
Created on Mar 4, 2013

@author: mandd
'''

import numpy as np
from scipy.spatial import Delaunay

def voronoiSampler(distributions, numberOfSamples):
  # generates points using LHS
  
  #