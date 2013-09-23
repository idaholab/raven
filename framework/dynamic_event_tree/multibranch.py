'''
Created on April 21, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
    xrange = range

# multiBranch function. It must be used by the user who wants
# to initiate a multi-branch
# @param newValuesList: list of new branching values
# @param associatedProbabilities: list of associated state probabilities
def multi_branch(newValuesList,associatedProbabilities):
  multiBranchString = ''
  for i in range(len(newValuesList)):
    if(i != len(newValuesList) -1):
      ending = str('/')
    else:
      ending = str('')
    multiBranchString = multiBranchString + (str(newValuesList[i])) + str('_') + (str(associatedProbabilities[i])) + ending
  
  return multiBranchString.encode()