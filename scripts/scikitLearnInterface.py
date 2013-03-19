'''
Created on Mar 16, 2013

@author: crisr
'''
from sklearn import svm
import Datas
import numpy



class Supervised():
  def train(self):
    return
  def reSet(self):
    return
  def evaluate(self):
    return

class SVMinterface(svm.SVC,Supervised):
  def parList(self):
    return svm.SVC._get_param_names()




classDictionary = {}
classDictionary['SVM']=SVMinterface