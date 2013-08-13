'''
Created on Feb 6, 2013

@author: mandd
'''

class classifierInterface:
  '''
  Container for Classifiers (it is a dictionary)
  '''
  def __init__(classifierName):
    '''
    Constructor
    '''
    self.myType = classifierName
    self.classifierDictClasses = {}
    self.classifierDictClasses ['SVM'] = SVMclassifier
    self.classifierDictClasses ['KNN'] = KNNclassifier
    self.classifierDictClasses ['DensityBased'] = DENSITYclassifier
    
    if classifierName in self.classifierDictClasse:
      self.train            = self.classifierDictClasses[self.myType].train
      self.checkConvergence = self.classifierDictClasses[self.myType].checkConvergence
      self.fittedGrid       = self.calssifierDictClasses[self.myType].fittedGrid
      self.pickNewSample    = self.classifierDictClasses[self.myType].pickNewSample
      self.predict          = self.classifierDictClasses[self.myType].predict
    else:
      print "Classifier not found"