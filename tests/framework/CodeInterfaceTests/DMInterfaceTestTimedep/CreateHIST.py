# -*- coding: utf-8 -*-
"""
created 11/16/16
$
@author: A. S. Epiney
"""

# This file assembles a couple of points into a vector

import numpy as np


def initialize(self, runInfoDict, inputFiles):

    pass


def run(self, Inputs):

    print "=============== Inside CreateHIST ================="
    self.G_vect = np.array([Inputs['G_a'], Inputs['G_b'], Inputs['G_c']])
    print "=============== End CreateHIST ================="





