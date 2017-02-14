# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 08:23:59 2016

@author: bakete - Ted Baker, ted.baker@inl.gov/tedb314@gmail.com
"""

# SMRCostModel
import numpy as np

hour = np.linspace(0,8759, 8760, endpoint = True)


def initialize(self,runInfoDict,inputFiles):

    # load demand data from file
    fileName = '2014_ERCOT_Hourly_Load_Data.csv'  # assign file name
    file = open(fileName)  # open file
    fileLen = sum(1 for line in file)  # determine length of file (should be 365*24)
    demandData = np.zeros(fileLen)  # initialize array to store demand data
    file = open(fileName)  # open file
    for line in xrange(fileLen):
        demandData[line] = file.readline()  # store each line in array
    self.demandDataOriginal = demandData

def run(self, Inputs):
    # for testing, we perturb the demandData
    self.demandData = self.demandDataOriginal*self.perturbDemand
