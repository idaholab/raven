'''
Created on April 21, 2013

@author: alfoa
'''
# multiBranch function. It must be used by the user who wants
# to initiate a multi-branch
# @param newValuesList: list of new branching values
# @param associatedProbabilities: list of associated state probabilities
def multiBranch(newValuesList,associatedProbabilities):
  multiBranchString = ""

  for i in xrange(len(newValuesList)):
    if(i != len(newValuesList) -1):
      ending = "/"
    else:
      ending = ""
    multiBranchString = multiBranchString + str(newValuesList[i]) + "_" + str(associatedProbabilities[i]) + ending
  
  return multiBranchString