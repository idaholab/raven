'''
Created on Feb 6, 2013

@author: mandd
'''

import numpy as np
from scikits.learn import svm

class SVMclassifier():
  '''
  this is the classifier based on the SVM algorithm
  '''
  def __init__(self, kernel, samples, outcomes, features, discretizationPoints):
    self.kernel = kernel
    self.samples = samples
    self.features = features
    self.outcomes = outcomes
    self.discretizationPoints = discretizationPoints     # number of points for each future equqlly spaced 
  
  def train(self):
    classifier = svm.SVC(self.kernel)
    classifier.fit(self.samples,self.outcomes)
    return classifier
  
  def predicit(self, classifier, pivot):
    return classifier.predict(pivot)
        
  def findPointsOnLimitSurface(self):   # o___o___o___o___o    o = discretization point 
    
    numberOfGridPoints=1
    
    for i in range(self.discretizationPoints):   # Determine how many points fit the grid
      numberOfGridPoints = numberOfGridPoints * discretizationPoints[i]   
      
    numberOfLimitPoints=0  
    
    for i in range(numberOfGridPoints):
      pivot = intToCoordinateConverter(i)
      isLimitPoint=isOnLimitSurface(pivot)
      
      if isLimitPoint>0:
        if numberOfLimitPoints==0:
          limitPoints=pivot
        else:
          append(limitPoints,pivot,axis=1)
        numberOfLimitPoints = numberOfLimitPoints +1

    return limitPoints
  
  def intToCoordinateConverter(self, pivotInt):
    weights[0]=1
    for d in range(self.features-1):
      weigths[d+1]=weigths[d]*self.discretizationPoints[d]
      
      
    for d in range(self.features):
      coordinate[self.features-d]=pivotInt % weights[d]
      pivotInt=pivotInt-coordinate[self.features-d]*weights[d]
    
    return coordinate
  
  def coordinateToInt(self, pivotCoordinate):
    weights[0]=1
    
    for d in range(self.features-1):
      weigths[d+1]=weigths[d]*self.discretizationPoints[d]
    
    pivotInt=0
    for d in range(self.features):
      pivotInt=pivotInt+weigths[d]*pivotCoordinate[d]
      
    return pivotInt
  
  def findNeighbors(self,pivot):
    neighborCounter=0
    
    for d in range(self.features):
      if (pivot[d]!=0 or pivot[d]!=discretizationPoints[d]):
        newNeighbor=pivot
        newNeighbor[d]=newNeighbor[d]-1
        neighborCounter+=1
        if neighborCounter==0:
          neighbors=newNeighbor
        else:
          append(neighbors,newNeighbort,axis=1)
    
    return neighbors
  
  def isOnLimitSurface(self,pivot):
    neighbors=self.findNeighbors(pivot)
    
    onLimitSurface=0
    
    numberOfNeighbors = np.shape(neighbors)[0]
    
    for i in range(numberOfNeighbors):
        if neighbors[i][features+1]==pivot[features+1]:
          onLimitSurface=onLimitSurface
        else:
          onLimitSurface=onLimitSurface+1
          
    return onLimitSurface
  
  def pickNextSamples(self, pointsOnLimitSurface,numberOfSamplesToPick):
    for i in range(pointsOnLimitSurface):
      closestSample[1,i]=i
      closestSample[2,i]=-1
      closestSampleDistance=1000000000
      for j in range(self.samples):
        distance=L2distance(fromGridToCartCoordinate(pointsOnLimitSurface[i,:]),samples[j,:])
        if distance<closestSampleDistance:
          closestSampleDistance=distance
          closestSample[2,i]=j
          
    closestSample[np.argsort(closestSample[:,1])]    # arrange point from closest to farthest
    
    if numberOfSamplesToPick > np.shape():
      return closestSample
    else:
      numberOfPointsOnLimitSurface=np.shape(pointsOnLimitSurface)[0]
      for i in range(numberOfSamplesToPick):
        subsetOfClosestSample[i]=closestSample[:,numberOfPointsOnLimitSurface-i]
      return subsetOfClosestSample
  
  def L2distance(self,point1,point2):
    distance=0
    for d in range(self.features):
      distance=distance+(point1[d]-point2[d])*(point1[d]-point2[d])
    distance=np.sqrt(distance)
    
    return distance
  
  def fromGridToCartCoordinate(self,point1onGrid):
    for d in range(self.features):
      pointOnCartCoordinate[d]=point1onGrid[d]/self.discretizationPoints[d]
    
    return pointOnCartCoordinate
        