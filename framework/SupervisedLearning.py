'''
Module containing all supported type of ROM aka Surrogate Models etc
here we intend ROM as super-visioned learning,
where we try to understand the underlying model by a set of labeled sample
a sample is composed by (feature,label) that is easy translated in (input,output)
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from sklearn import linear_model
from sklearn import svm
from sklearn import multiclass
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import qda
from sklearn import tree
from sklearn import lda
from sklearn import gaussian_process
import sys
import numpy as np
import numpy
import abc
import ast
import pickle as pk
from operator import itemgetter
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import TreeStructure
interpolationND = utils.find_interpolationND()
#Internal Modules End--------------------------------------------------------------------------------

class superVisedLearning(utils.metaclass_insert(abc.ABCMeta)):
  '''
  This is the general interface to any superVisedLearning learning method.
  Essentially it contains a train, and evaluate methods
  '''
  returnType      = '' #this describe the type of information generated the possibility are 'boolean', 'integer', 'float'
  qualityEstType  = [] #this describe the type of estimator returned known type are 'distance', 'probability'. The values are returned by the self.__confidenceLocal__(Features)
  ROMtype         = '' #the broad class of the interpolator

  @staticmethod
  def checkArrayConsistency(arrayin):
    '''
    This method checks the consistency of the in-array
    @ In, object... It should be an array
    @ Out, tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    '''
    if type(arrayin) != numpy.ndarray: return (False,' The object is not a numpy array')
    if len(arrayin.shape) > 1: return(False, ' The array must be 1-d')
    return (True,'')

  def __init__(self,**kwargs):
    self.printTag = utils.returnPrintTag('SuperVised')
    #booleanFlag that controls the normalization procedure. If true, the normalization is performed. Default = True
    if kwargs != None: self.initOptionDict = kwargs
    else             : self.initOptionDict = {}
    if 'Features' not in self.initOptionDict.keys(): utils.raiseAnError(IOError,self,'Feature names not provided')
    if 'Target'   not in self.initOptionDict.keys(): utils.raiseAnError(IOError,self,'Target name not provided')
    self.features = self.initOptionDict['Features'].split(',')
    self.target   = self.initOptionDict['Target'  ]
    self.initOptionDict.pop('Target')
    self.initOptionDict.pop('Features')
    if self.features.count(self.target) > 0: utils.raiseAnError(IOError,self,'The target and one of the features have the same name!')
    #average value and sigma are used for normalization of the feature data
    #a dictionary where for each feature a tuple (average value, sigma)
    self.muAndSigmaFeatures = {}
    #these need to be declared in the child classes!!!!
    self.amITrained         = False

  def initialize(self,idict):
    pass #Overloaded by (at least) GaussPolynomialRom

  def train(self,tdict):
    '''
      Method to perform the training of the superVisedLearning algorithm
      NB.the superVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, training dictionary
      @ Out, None
    '''
    if type(tdict) != dict: utils.raiseAnError(TypeError,self,'In method "train", the training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = list(tdict.keys()), list(tdict.values())
    if self.target in names: targetValues = values[names.index(self.target)]
    else                   : utils.raiseAnError(IOError,self,'The output sought '+self.target+' is not in the training set')
    # check if the targetValues are consistent with the expected structure
    resp = self.checkArrayConsistency(targetValues)
    if not resp[0]: utils.raiseAnError(IOError,self,'In training set for target '+self.target+':'+resp[1])
    # construct the evaluation matrixes
    featureValues = np.zeros(shape=(targetValues.size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: utils.raiseAnError(IOError,self,'The feature sought '+feat+' is not in the training set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: utils.raiseAnError(IOError,self,'In training set for feature '+feat+':'+resp[1])
        if values[names.index(feat)].size != featureValues[:,0].size: utils.raiseAnError(IOError,self,'In training set, the number of values provided for feature '+feat+' are != number of target outcomes!')
        self._localNormalizeData(values,names,feat)
        if self.muAndSigmaFeatures[feat][1]==0: self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0],np.max(np.absolute(values[names.index(feat)])))
        if self.muAndSigmaFeatures[feat][1]==0: self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0],1.0)
        featureValues[:,cnt] = (values[names.index(feat)] - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
    self.__trainLocal__(featureValues,targetValues)
    self.amITrained = True

  def _localNormalizeData(self,values,names,feat):
    '''
    Method to normalize data based on the mean and standard deviation.  If undesired for a particular ROM,
    this method can be overloaded to simply pass (see, e.g., GaussPolynomialRom).
    @ In, values, list of feature values (from tdict)
    @ In, names, names of features (from tdict)
    @ In, feat, list of features (from ROM)
    @ Out, None
    '''
    self.muAndSigmaFeatures[feat] = (np.average(values[names.index(feat)]),np.std(values[names.index(feat)]))

  def confidence(self,edict):
    '''
    This call is used to get an estimate of the confidence in the prediction.
    The base class self.confidence will translate a dictionary into numpy array, then call the local confidence
    '''
    if type(edict) != dict: utils.raiseAnError(IOError,self,'method "confidence". The inquiring set needs to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values   = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: utils.raiseAnError(IOError,self,'In evaluate request for feature '+names[index]+':'+resp[1])
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: utils.raiseAnError(IOError,self,'The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: utils.raiseAnError(IOError,self,'In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = values[names.index(feat)]
    return self.__confidenceLocal__(featureValues)

  def evaluate(self,edict):
    '''
    Method to perform the evaluation of a point or a set of points through the previous trained superVisedLearning algorithm
    NB.the superVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
    the interface with the kernels requires.
    @ In, tdict, evaluation dictionary
    @ Out, numpy array of evaluated points
    '''
    if type(edict) != dict: utils.raiseAnError(IOError,self,'method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values  = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: utils.raiseAnError(IOError,self,'In evaluate request for feature '+names[index]+':'+resp[1])
    # construct the evaluation matrix
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: utils.raiseAnError(IOError,self,'The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: utils.raiseAnError(IOError,self,'In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = ((values[names.index(feat)] - self.muAndSigmaFeatures[feat][0]))/self.muAndSigmaFeatures[feat][1]
    return self.__evaluateLocal__(featureValues)

  def reset(self):
    '''override this method to re-instance the ROM'''
    self.amITrained = False
    self.__resetLocal__()

  def returnInitialParameters(self):
    '''override this method to return the fix set of parameters of the ROM'''
    iniParDict = dict(self.initOptionDict.items() + {'returnType':self.__class__.returnType,'qualityEstType':self.__class__.qualityEstType,'Features':self.features,
                                             'Target':self.target,'returnType':self.__class__.returnType}.items() + self.__returnInitialParametersLocal__().items())
    return iniParDict

  def returnCurrentSetting(self):
    '''return the set of parameters of the ROM that can change during simulation'''
    return dict({'Trained':self.amITrained}.items() + self.__CurrentSettingDictLocal__().items())

  def printXML(self,rootnode,options=None):
    '''
      Allows the SVE to put whatever it wants into an XML to print to file.
      @ In, options, dict of string-based options to use, including filename, things to print, etc
      @ Out, treedict, dict of strings to be printed
    '''
    node = TreeStructure.Node(self.target)
    rootnode.appendBranch(node)
    self._localPrintXML(node,options)

  def _localPrintXML(self,node,options=None):
    '''
      Specific local method for printing anything desired to xml file.  Overwrite in inheriting classes.
      @ In, options, dict of string-based options to use, including filename, things to print, etc
      @ Out, treedict, dict of strings to be printed
    '''
    #if treedict=={}: treedict={'PrintOptions':'ROM of type '+str(self.printTag.strip())+' has no special output options.'}
    node.addText('ROM of type '+str(self.printTag.strip())+' has no special output options.')

  @abc.abstractmethod
  def __trainLocal__(self,featureVals,targetVals):
    '''@ In, featureVals, 2-D numpy array [n_samples,n_features]'''

  @abc.abstractmethod
  def __confidenceLocal__(self,featureVals):
    '''
    This should return an estimation of the quality of the prediction.
    This could be distance or probability or anything else, the type needs to be declared in the variable cls.qualityEstType
    @ In, featureVals, 2-D numpy array [n_samples,n_features]
    '''

  @abc.abstractmethod
  def __evaluateLocal__(self,featureVals):
    '''
    @ In,  featureVals, 2-D numpy array [n_samples,n_features]
    @ Out, targetVals , 1-D numpy array [n_samples]
    '''

  @abc.abstractmethod
  def __resetLocal__(self,featureVals):
    '''After this method the ROM should be described only by the initial parameter settings'''

  @abc.abstractmethod
  def __returnInitialParametersLocal__(self):
    '''this should return a dictionary with the parameters that could be possible not in self.initOptionDict'''

  @abc.abstractmethod
  def __returnCurrentSettingLocal__(self):
    '''override this method to pass the set of parameters of the ROM that can change during simulation'''
#
#
#
class NDinterpolatorRom(superVisedLearning):
  def __init__(self,**kwargs):
    superVisedLearning.__init__(self,**kwargs)
    self.interpolator = None
    self.printTag = utils.returnPrintTag('ND Interpolation ROM')

  def __trainLocal__(self,featureVals,targetVals):
    """
    Perform training on samples in X with responses y.
    For an one-class model, +1 or -1 is returned.
    Parameters: featureVals : {array-like, sparse matrix}, shape = [n_samples, n_features]
    Returns   : targetVals : array, shape = [n_samples]
    """
    featv = interpolationND.vectd2d(featureVals[:][:])
    targv = interpolationND.vectd(targetVals)
    self.interpolator.fit(featv,targv)

  def __confidenceLocal__(self,featureVals):
    utils.raiseAnError(NotImplementedError,self,'NDinterpRom   : __confidenceLocal__ method must be implemented!')

  def __evaluateLocal__(self,featureVals):
    '''
    Perform regression on samples in featureVals.
    For an one-class model, +1 or -1 is returned.
    @ In, numpy.array 2-D, features
    @ Out, numpy.array 1-D, predicted values
    '''
    prediction = np.zeros(featureVals.shape[0])
    for n_sample in range(featureVals.shape[0]):
      featv = interpolationND.vectd(featureVals[n_sample][:])
      prediction[n_sample] = self.interpolator.interpolateAt(featv)
      utils.raiseAMessage(self,'NDinterpRom   : Prediction by ' + self.__class__.ROMtype + '. Predicted value is ' + str(prediction[n_sample]))
    return prediction

  def __returnInitialParametersLocal__(self):
    '''there are no possible default parameters to report'''
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    utils.raiseAnError(NotImplementedError,self,'NDinterpRom   : __returnCurrentSettingLocal__ method must be implemented!')

class GaussPolynomialRom(NDinterpolatorRom):
  def __confidenceLocal__(self,edict):pass #TODO

  def __resetLocal__(self):
    pass

  def __returnCurrentSettingLocal__(self):pass #TODO

  def __init__(self,**kwargs):
    superVisedLearning.__init__(self,**kwargs)
    self.interpolator  = None #FIXME what's this?
    self.printTag      = utils.returnPrintTag('GAUSSgpcROM('+self.target+')')
    self.indexSetType  = None #string of index set type, TensorProduct or TotalDegree or HyperbolicCross
    self.maxPolyOrder  = None #integer of relative maximum polynomial order to use in any one dimension
    self.itpDict       = {}   #dict of quad,poly,weight choices keyed on varName
    self.norm          = None #combined distribution normalization factors (product)
    self.sparseGrid    = None #Quadratures.SparseGrid object, has points and weights
    self.distDict      = None #dict{varName: Distribution object}, has point conversion methods based on quadrature
    self.quads         = None #dict{varName: Quadrature object}, has keys for distribution's point conversion methods
    self.polys         = None #dict{varName: OrthoPolynomial object}, has polynomials for evaluation
    self.indexSet      = None #array of tuples, polynomial order combinations
    self.polyCoeffDict = None #dict{index set point, float}, polynomial combination coefficients for each combination
    self.itpDict       = {}   #dict{varName: dict{attribName:value} }

    for key,val in kwargs.items():
      if key=='IndexSet': self.indexSetType = val
      if key=='PolynomialOrder': self.maxPolyOrder = val
      if key=='Interpolation':
        for var,val in val.items():
          self.itpDict[var]={'poly'  :'DEFAULT',
                             'quad'  :'DEFAULT',
                             'weight':'1'}
          for atrName,atrVal in val.items():
            if atrName in ['poly','quad','weight']: self.itpDict[var][atrName]=atrVal
            else: utils.raiseAnError(IOError,self,'Unrecognized option: '+atrName)

    if not self.indexSetType:
      utils.raiseAnError(IOError,self,'No IndexSet specified!')
    if not self.maxPolyOrder:
      utils.raiseAnError(IOError,self,'No maxPolyOrder specified!')
    if self.maxPolyOrder < 1:
      utils.raiseAnError(IOError,self,'Polynomial order cannot be less than 1 currently.')

  def _localPrintXML(self,node,options=None):
    if not self.amITrained: utils.raiseAnError(RuntimeError,'ROM is not yet trained!')
    self.mean=None
    canDo = ['mean','variance']
    if 'what' in options.keys():
      requests = list(o.strip() for o in options['what'].split(','))
      if 'all' in requests: requests = canDo
      for request in requests:
        request=request.strip()
        newnode = TreeStructure.Node(request)
        if   request.lower() in ['mean','expectedvalue']:
          if self.mean == None: self.mean = self.__evaluateMoment__(1)
          newnode.setText(self.mean)
        elif request.lower() in ['variance']:
          if self.mean == None: self.mean = self.__evaluateMoment__(1)
          newnode.setText(self.__evaluateMoment__(2) - self.mean*self.mean)
        else:
          utils.raiseAWarning(self,'ROM does not know how to return '+request)
          newnode.setText('not found')
        node.appendBranch(newnode)

  def _localNormalizeData(self,values,names,feat):
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def interpolationInfo(self):
    return dict(self.itpDict)

  def initialize(self,idict):
    for key,value in idict.items():
      if   key == 'SG'   : self.sparseGrid = value
      elif key == 'dists': self.distDict   = value
      elif key == 'quads': self.quads      = value
      elif key == 'polys': self.polys      = value
      elif key == 'iSet' : self.indexSet   = value

  def _multiDPolyBasisEval(self,orders,pts):
    '''Evaluates each polynomial set at given orders and points, returns product.
    @ In orders, tuple(int), polynomial orders to evaluate
    @ In pts, tuple(float), values at which to evaluate polynomials
    @ Out, float, product of polynomial evaluations
    '''
    tot=1
    for i,(o,p) in enumerate(zip(orders,pts)):
      varName = self.sparseGrid.varNames[i]
      tot*=self.polys[varName](o,p)
    return tot

  def __trainLocal__(self,featureVals,targetVals):
    self.polyCoeffDict={}
    #check consistency of featureVals
    #there might be more points than we need, so don't error on this
    #if len(featureVals)!=len(self.sparseGrid):
    #  utils.raiseAnError(IOError,self,'ROM requires '+str(len(self.sparseGrid))+' points, but '+str(len(featureVals))+' provided!')
    #the dimensions of featureVals might be reordered from sparseGrid, so fix it here
    self.sparseGrid._remap(self.features)
    utils.raiseAMessage(self,'types: '+str(type(self.sparseGrid.points()))+' | '+str(type(featureVals)))
    #check equality of point space
    #fvs=sorted(fvs,key=itemgetter(*range(len(fvs[0]))))
    #sgs=sorted(sgs,key=itemgetter(*range(len(sgs[0]))))
    fvs = []
    tvs=[]
    sgs = self.sparseGrid.points()[:]
    missing=[]
    for pt in sgs:
      found,idx,point = utils.NDInArray(featureVals,pt)
      if found:
        fvs.append(point)
        tvs.append(targetVals[idx])
      else:
        missing.append(pt)
    if len(missing)>0:
      msg='\n'
      #if not np.allclose(fvs,sgs,rtol=1e-15):
      msg+='DEBUG missing feature vals:\n'
      for i in missing:
        msg+='  '+str(i)+'\n'
      utils.raiseAWarning(self,msg)
      utils.raiseAnError(IOError,self,'input values do not match required values!')
    #make translation matrix between lists
    translate={}
    for i in range(len(fvs)):
      translate[tuple(fvs[i])]=sgs[i]
    self.norm = np.prod(list(self.distDict[v].measureNorm(self.quads[v].type) for v in self.distDict.keys()))
    for i,idx in enumerate(self.indexSet):
      idx=tuple(idx)
      self.polyCoeffDict[idx]=0
      wtsum=0
      for pt,soln in zip(fvs,tvs):
        stdPt = np.zeros(len(pt))
        for i,p in enumerate(pt):
          varName = self.sparseGrid.varNames[i]
          stdPt[i] = self.distDict[varName].convertToQuad(self.quads[varName].type,p)
        wt = self.sparseGrid.weights(translate[tuple(pt)])
        self.polyCoeffDict[idx]+=soln*self._multiDPolyBasisEval(idx,stdPt)*wt
      self.polyCoeffDict[idx]*=self.norm
    self.amITrained=True
    #self.printPolyDict()

  def printPolyDict(self,printZeros=False):
    '''Human-readable version of the polynomial chaos expansion.
    @ In printZeros,boolean,optional flag for printing even zero coefficients
    @ Out, None, None
    '''
    data=[]
    for idx,val in self.polyCoeffDict.items():
      if val > 1e-14 or printZeros:
        data.append([idx,val])
    data.sort()
    msg='polyDict for ['+self.target+'] with inputs '+str(self.features)+': \n'
    for idx,val in data:
      msg+='    '+str(idx)+' '+str(val)+'\n'
    utils.raiseAMessage(msg)

  def __evaluateMoment__(self,r):
    '''Use the ROM's built-in method to calculate moments.
    @ In r, int, moment to calculate
    @ Out, float, evaluation of moment
    '''
    #TODO is there a faster way still to do this?
    tot=0
    for pt,wt in self.sparseGrid:
      tot+=self.__evaluateLocal__([pt])**r*wt
    tot*=self.norm
    return tot

  def __evaluateLocal__(self,featureVals):
    featureVals=featureVals[0]
    tot=0
    stdPt = np.zeros(len(featureVals))
    for p,pt in enumerate(featureVals):
      varName = self.sparseGrid.varNames[p]
      stdPt[p] = self.distDict[varName].convertToQuad(self.quads[varName].type,pt)
    for idx,coeff in self.polyCoeffDict.items():
      tot+=coeff*self._multiDPolyBasisEval(idx,stdPt)
    return tot

  def __returnInitialParametersLocal__(self):
    return {}#TODO 'IndexSet:':self.indexSetType,
             #'PolynomialOrder':self.maxPolyOrder,
             # 'Interpolation':interpolationInfo()}

class HDMRRom(GaussPolynomialRom):
  def __confidenceLocal__(self,edict):pass #TODO

  def __resetLocal__(self):
    pass

  def __returnCurrentSettingLocal__(self):pass #TODO

  def _localNormalizeData(self,values,names,feat):
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __init__(self,**kwargs):
    '''Initializes SupervisedEngine. See base class.'''
    superVisedLearning.__init__(self,**kwargs)
    self.printTag      = utils.returnPrintTag('HDMR_ROM('+self.target+')')
    self.sobolOrder    = None #depth of HDMR/Sobol expansion
    self.indexSetType  = None #string of index set type, TensorProduct or TotalDegree or HyperbolicCross
    self.maxPolyOrder  = None #integer of relative maximum polynomial order to use in any one dimension
    self.itpDict       = {}   #dict of quad,poly,weight choices keyed on varName
    self.ROMs          = {}   #dict of GaussPolyROM objects keyed by combination of vars that make them up
    self.sparseGrid    = None #Quadratures.SparseGrid object, has points and weights
    self.distDict      = None #dict{varName: Distribution object}, has point conversion methods based on quadrature
    self.quads         = None #dict{varName: Quadrature object}, has keys for distribution's point conversion methods
    self.polys         = None #dict{varName: OrthoPolynomial object}, has polynomials for evaluation
    self.indexSet      = None #array of tuples, polynomial order combinations
    self.polyCoeffDict = None #dict{index set point, float}, polynomial combination coefficients for each combination
    self.itpDict       = {}   #dict{varName: dict{attribName:value} }
    self.sdx           = None #dict of sobol sensitivity coeffs, keyed on order and tuple(varnames)
    self.mean          = None #mean, store to avoid recalculation
    self.variance      = None #variance, store to avoid recalculation

    for key,val in kwargs.items():
      if key=='SobolOrder': self.sobolOrder = int(val)
      if key=='IndexSet': self.indexSetType = val
      if key=='PolynomialOrder': self.maxPolyOrder = val
      if key=='Interpolation':
        for var,val in val.items():
          self.itpDict[var]={'poly'  :'DEFAULT',
                             'quad'  :'DEFAULT',
                             'weight':'1'}
          for atrName,atrVal in val.items():
            if atrName in ['poly','quad','weight']: self.itpDict[var][atrName]=atrVal
            else: raise IOError(self.printTag+' Unrecognized option: '+atrName)

    if self.indexSetType==None:
      raise IOError(self.printTag+' No IndexSet specified!')
    if self.maxPolyOrder==None:
      raise IOError(self.printTag+' No maxPolyOrder specified!')
    if self.maxPolyOrder < 1:
      raise IOError(self.printTag+' Polynomial order cannot be less than 1 currently.')

  def _localPrintXML(self,node,options=None):
    if not self.amITrained: utils.raiseAnError(RuntimeError,'ROM is not yet trained!')
    self.mean=None
    canDo = ['mean','variance','indices']
    if 'what' in options.keys():
      requests = list(o.strip() for o in options['what'].split(','))
      if 'all' in requests: requests = canDo
      for request in requests:
        request=request.strip()
        newnode = TreeStructure.Node(request)
        #node.appendBranch(newnode)
        if request.lower() in ['mean','expectedvalue']: newnode.setText(self.__mean__())
        elif request.lower() in ['variance']: newnode.setText(self.__variance__())
        elif request.lower() in ['indices']:
          pcts,totpct,totvar = self.getPercentSensitivities(returnTotal=True)
          vnode = TreeStructure.Node('total_variance')
          vnode.setText(totvar)
          newnode.appendBranch(vnode)
          entries=[]
          for combo,val in pcts.items():
            entries.append((combo,val))
          entries.sort(key=itemgetter(1),reverse=True)
          for combo,sens in entries:
            snode = TreeStructure.Node(str(combo))
            snode.setText(sens)
            newnode.appendBranch(snode)
        else:
          utils.raiseAWarning(self,'ROM does not know how to return '+request)
          newnode.setText('not found')
        node.appendBranch(newnode)

  def _localNormalizeData(self,values,names,feat):
    '''Overwrite normalization. See base class.'''
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def interpolationInfo(self):
    '''See base class.'''
    return dict(self.itpDict)

  def initialize(self,idict):
    '''Called by sampler to pass necessary information along.  See base class.'''
    for key,value in idict.items():
      if   key == 'ROMs' : self.ROMs       = value
      elif key == 'dists': self.distDict   = value
      elif key == 'quads': self.quads      = value
      elif key == 'polys': self.polys      = value
      elif key == 'refs' : self.references = value

  def __trainLocal__(self,featureVals,targetVals):
    '''
      Because HDMR rom is a collection of sub-roms, we call sub-rom "train" to do what we need it do.
      @ In, tdict, training dictionary
      @ Out, None
    '''
    ft={}
    for i in range(len(featureVals)):
      ft[tuple(featureVals[i])]=targetVals[i]
    #get the reference case
    self.refpt = tuple(self.__fillPointWithRef((),[]))
    self.refSoln = ft[tuple(self.refpt)]
    for combo,rom in self.ROMs.items():
      subtdict={}
      for c in combo: subtdict[c]=[]
      subtdict[self.target]=[]
      SG = rom.sparseGrid
      fvals=np.zeros([len(SG),len(combo)])
      tvals=np.zeros(len(SG))
      for i in range(len(SG)):
        getpt=tuple(self.__fillPointWithRef(combo,SG[i][0]))
        tvals[i] = ft[getpt]
        for fp,fpt in enumerate(SG[i][0]):
          fvals[i][fp] = fpt
      for i,c in enumerate(combo):
        subtdict[c] = fvals[:,i]
      subtdict[self.target] = tvals
      rom.train(subtdict)
      #rom.__trainLocal__(fvals,tvals)

    #make ordered list of combos for use later
    maxLevel = max(list(len(combo) for combo in self.ROMs.keys()))
    self.combos = []
    for i in range(maxLevel+1):
      self.combos.append([])
    for combo in self.ROMs.keys():
      self.combos[len(combo)].append(combo)

    self.amITrained = True

  def __fillPointWithRef(self,combo,pt):
    '''Given a "combo" subset of the full input space and a partially-filled
       point within that space, fills the rest of space with the reference
       cut values.
       @ In, combo, tuple of strings, names of subset dimensions
       @ In, pt, list of floats, values of points in subset dimension
       @ Out, newpt, full point in input dimension space on cut-hypervolume
    '''
    newpt=np.zeros(len(self.features))
    for v,var in enumerate(self.features):
      if var in combo:
        newpt[v] = pt[combo.index(var)]
      else:
        newpt[v] = self.references[var]
    return newpt

  def __mean__(self):
    '''The Cut-HDMR approximation can return its mean easily.'''
    if self.mean != None: return self.mean
    vals={'':self.refSoln}
    for i,c in enumerate(self.combos):
      for combo in c:
        rom = self.ROMs[combo]
        vals[combo] = rom.__evaluateMoment__(1) - vals['']
        for cl in range(i):
          for doneCombo in self.combos[cl]:
            if set(doneCombo).issubset(set(combo)):
              vals[combo] -= vals[doneCombo]
    tot = sum(vals.values())
    self.mean=tot
    return tot

  def __variance__(self):
    '''The Cut-HDMR approximation can return its variance easily.'''
    if self.variance != None: return self.variance
    vals={}
    for i,c in enumerate(self.combos):
      for combo in c:
        rom = self.ROMs[combo]
        mean = rom.__evaluateMoment__(1)
        vals[combo] = rom.__evaluateMoment__(2) - mean*mean
        for cl in range(i):
          for doneCombo in self.combos[cl]:
            if set(doneCombo).issubset(set(combo)):
              vals[combo] -= vals[doneCombo]
    tot = sum(vals.values())
    self.variance = tot
    return tot

  def __evaluateLocal__(self,featureVals):
    '''Evaluates ROM at given points.  See base class.'''
    #am I trained?
    if not self.amITrained: raise IOError(self.printTag+': '+utils.returnPrintPostTag('ERROR')+'-> Cannot evaluate, as ROM is not trained!')
    fvals=dict(zip(self.features,featureVals[0]))
    vals={'':self.refSoln}
    for i,c in enumerate(self.combos):
      for combo in c:
        myVals = [list(featureVals[0][self.features.index(j)] for j in combo)]
        rom = self.ROMs[combo]
        #check if rom is trained
        if not rom.amITrained: raise IOError('ROM for subset %s is not trained!' %combo)
        vals[combo] = rom.__evaluateLocal__(myVals) - vals['']
        for cl in range(i):
          for doneCombo in self.combos[cl]:
            if set(doneCombo).issubset(set(combo)):
              vals[combo] -= vals[doneCombo]
    tot = sum(vals.values())
    return tot

  def getSensitivities(self,maxLevel=None,kind='variance'):
    '''
      Generates dictionary of Sobol indices for the requested levels.
      Optionally the moment (r) to get sensitivity indices of can be requested.
      @ In, levels, list, levels to obtain indices for. Defaults to all available.
      @ In, kind, string, the metric to use when calculating sensitivity indices. Defaults to variance.
    '''
    if kind.lower().strip() not in ['mean','variance']:
      raise IOError(self.printTag+': '+utils.returnPrintPostTag('ERROR'),'-> Requested sensitivity benchmark is %s, but expected "mean" or "variance".' %kind)
    avail = max(list(len(combo) for combo in self.ROMs.keys()))
    if maxLevel==None: maxLevel = avail
    else:
      if maxLevels>avail: raise IOError(self.printTag+': '+utils.returnPrintPostTag('ERROR')+'-> Requested level %i for sensitivity analyis, but this composition is at most %i order!' %(maxLevel,avail) )

    self.sdx = {}
    for l in range(maxLevel+1):
      self.sdx[l]={}
    #put basic metric in
    for i,c in enumerate(self.combos):
      for combo in c:
        rom = self.ROMs[combo]
        mean = rom.__evaluateMoment__(1)
        self.sdx[i][combo] = rom.__evaluateMoment__(2) - mean*mean
        for cl in range(i):
          for doneCombo in self.combos[cl]:
            if set(doneCombo).issubset(set(combo)):
              self.sdx[i][combo]-=self.sdx[cl][doneCombo]

  def getPercentSensitivities(self,variance=None,returnTotal=False):
    '''Calculates percent sensitivities.
    If variance specified, uses it as the bnechmark variance, otherwise uses ROM to calculate total variance approximately.
    If returnTotal specified, also returns percent of total variance and the total variance value.
    @ In, variance, float to represent user-provided total variance
    @ In, returnTotal, boolean to turn on returning total percent and total variance
    @ Out, pcts, percent=based Sobol sensitivity indices
    '''
    if self.sdx == None or len(self.sdx)<1:
      self.getSensitivities()
    if variance==None or variance==0:
      variance = self.__variance__()
      variance = 0.0
      for c,combos in self.sdx.items():
        for combo in combos:
          variance+=self.sdx[c][combo]
    tot=0.0
    totvar=0.0
    pcts={}
    for c,combos in self.sdx.items():
      for combo in combos:
        totvar+=self.sdx[c][combo]
        pcts[combo]=self.sdx[c][combo]/variance
        tot+=pcts[combo]
    if returnTotal: return pcts,tot,totvar
    else: return pcts

#
#
#
class NDsplineRom(NDinterpolatorRom):
  ROMtype         = 'NDsplineRom'
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.printTag = utils.returnPrintTag('ND-SPLINE ROM')
    self.interpolator = interpolationND.NDspline()

  def __resetLocal__(self):
    ''' The reset here erase the Interpolator while keeping the instance'''
    self.interpolator.reset()
#
#
#
class NDinvDistWeigth(NDinterpolatorRom):
  ROMtype         = 'NDinvDistWeigth'
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.printTag = utils.returnPrintTag('ND-INVERSEWEIGHT ROM')
    if not 'p' in self.initOptionDict.keys(): utils.raiseAnError(IOError,self,'the <p> parameter must be provided in order to use NDinvDistWeigth as ROM!!!!')
    self.interpolator = interpolationND.InverseDistanceWeighting(float(self.initOptionDict['p']))

  def __resetLocal__(self):
    ''' The reset here erase the Interpolator while keeping the instance'''
    self.interpolator.reset(float(self.initOptionDict['p']))
#
#
#
class NDmicroSphere(NDinterpolatorRom):
  ROMtype         = 'NDmicroSphere'
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.printTag = utils.returnPrintTag('ND-MICROSPHERE ROM')
    if not 'p' in self.initOptionDict.keys(): utils.raiseAnError(IOError,self,'the <p> parameter must be provided in order to use NDmicroSphere as ROM!!!!')
    if not 'precision' in self.initOptionDict.keys(): utils.raiseAnError(IOError,self,'the <precision> parameter must be provided in order to use NDmicroSphere as ROM!!!!')
    self.interpolator = interpolationND.microSphere(float(self.initOptionDict['p']),int(self.initOptionDict['precision']))

  def __resetLocal__(self):
    self.interpolator.reset(float(self.initOptionDict['p']),int(self.initOptionDict['precision']))

class SciKitLearn(superVisedLearning):
  ROMtype = 'SciKitLearn'
  availImpl = {}
  availImpl['lda'] = {}
  availImpl['lda']['LDA'] = (lda.LDA, 'integer') #Quadratic Discriminant Analysis (QDA)
  availImpl['linear_model'] = {} #Generalized Linear Models
  availImpl['linear_model']['ARDRegression'               ] = (linear_model.ARDRegression               , 'float'  ) #Bayesian ARD regression.
  availImpl['linear_model']['BayesianRidge'               ] = (linear_model.BayesianRidge               , 'float'  ) #Bayesian ridge regression
  availImpl['linear_model']['ElasticNet'                  ] = (linear_model.ElasticNet                  , 'float'  ) #Linear Model trained with L1 and L2 prior as regularizer
  availImpl['linear_model']['ElasticNetCV'                ] = (linear_model.ElasticNetCV                , 'float'  ) #Elastic Net model with iterative fitting along a regularization path
  availImpl['linear_model']['Lars'                        ] = (linear_model.Lars                        , 'float'  ) #Least Angle Regression model a.k.a.
  availImpl['linear_model']['LarsCV'                      ] = (linear_model.LarsCV                      , 'float'  ) #Cross-validated Least Angle Regression model
  availImpl['linear_model']['Lasso'                       ] = (linear_model.Lasso                       , 'float'  ) #Linear Model trained with L1 prior as regularizer (aka the Lasso)
  availImpl['linear_model']['LassoCV'                     ] = (linear_model.LassoCV                     , 'float'  ) #Lasso linear model with iterative fitting along a regularization path
  availImpl['linear_model']['LassoLars'                   ] = (linear_model.LassoLars                   , 'float'  ) #Lasso model fit with Least Angle Regression a.k.a.
  availImpl['linear_model']['LassoLarsCV'                 ] = (linear_model.LassoLarsCV                 , 'float'  ) #Cross-validated Lasso, using the LARS algorithm
  availImpl['linear_model']['LassoLarsIC'                 ] = (linear_model.LassoLarsIC                 , 'float'  ) #Lasso model fit with Lars using BIC or AIC for model selection
  availImpl['linear_model']['LinearRegression'            ] = (linear_model.LinearRegression            , 'float'  ) #Ordinary least squares Linear Regression.
  availImpl['linear_model']['LogisticRegression'          ] = (linear_model.LogisticRegression          , 'float'  ) #Logistic Regression (aka logit, MaxEnt) classifier.
  availImpl['linear_model']['MultiTaskLasso'              ] = (linear_model.MultiTaskLasso              , 'float'  ) #Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer
  availImpl['linear_model']['MultiTaskElasticNet'         ] = (linear_model.MultiTaskElasticNet         , 'float'  ) #Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer
  availImpl['linear_model']['OrthogonalMatchingPursuit'   ] = (linear_model.OrthogonalMatchingPursuit   , 'float'  ) #Orthogonal Mathching Pursuit model (OMP)
  availImpl['linear_model']['OrthogonalMatchingPursuitCV' ] = (linear_model.OrthogonalMatchingPursuitCV , 'float'  ) #Cross-validated Orthogonal Mathching Pursuit model (OMP)
  availImpl['linear_model']['PassiveAggressiveClassifier' ] = (linear_model.PassiveAggressiveClassifier , 'integer') #Passive Aggressive Classifier
  availImpl['linear_model']['PassiveAggressiveRegressor'  ] = (linear_model.PassiveAggressiveRegressor  , 'float'  ) #Passive Aggressive Regressor
  availImpl['linear_model']['Perceptron'                  ] = (linear_model.Perceptron                  , 'float'  ) #Perceptron
  availImpl['linear_model']['RandomizedLasso'             ] = (linear_model.RandomizedLasso             , 'float'  ) #Randomized Lasso.
  availImpl['linear_model']['RandomizedLogisticRegression'] = (linear_model.RandomizedLogisticRegression, 'float'  ) #Randomized Logistic Regression
  availImpl['linear_model']['Ridge'                       ] = (linear_model.Ridge                       , 'float'  ) #Linear least squares with l2 regularization.
  availImpl['linear_model']['RidgeClassifier'             ] = (linear_model.RidgeClassifier             , 'float'  ) #Classifier using Ridge regression.
  availImpl['linear_model']['RidgeClassifierCV'           ] = (linear_model.RidgeClassifierCV           , 'integer') #Ridge classifier with built-in cross-validation.
  availImpl['linear_model']['RidgeCV'                     ] = (linear_model.RidgeCV                     , 'float'  ) #Ridge regression with built-in cross-validation.
  availImpl['linear_model']['SGDClassifier'               ] = (linear_model.SGDClassifier               , 'integer') #Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
  availImpl['linear_model']['SGDRegressor'                ] = (linear_model.SGDRegressor                , 'float'  ) #Linear model fitted by minimizing a regularized empirical loss with SGD
  availImpl['linear_model']['lars_path'                   ] = (linear_model.lars_path                   , 'float'  ) #Compute Least Angle Regression or Lasso path using LARS algorithm [1]
  availImpl['linear_model']['lasso_path'                  ] = (linear_model.lasso_path                  , 'float'  ) #Compute Lasso path with coordinate descent
  availImpl['linear_model']['lasso_stability_path'        ] = (linear_model.lasso_stability_path        , 'float'  ) #Stabiliy path based on randomized Lasso estimates
  availImpl['linear_model']['orthogonal_mp_gram'          ] = (linear_model.orthogonal_mp_gram          , 'float'  ) #Gram Orthogonal Matching Pursuit (OMP)

  availImpl['svm'] = {} #support Vector Machines
  availImpl['svm']['LinearSVC'] = (svm.LinearSVC, 'boolean')
  availImpl['svm']['SVC'      ] = (svm.SVC      , 'boolean')
  availImpl['svm']['NuSVC'    ] = (svm.NuSVC    , 'boolean')
  availImpl['svm']['SVR'      ] = (svm.SVR      , 'boolean')

  availImpl['multiClass'] = {} #Multiclass and multilabel classification
  availImpl['multiClass']['OneVsRestClassifier' ] = (multiclass.OneVsRestClassifier , 'integer') # One-vs-the-rest (OvR) multiclass/multilabel strategy
  availImpl['multiClass']['OneVsOneClassifier'  ] = (multiclass.OneVsOneClassifier  , 'integer') # One-vs-one multiclass strategy
  availImpl['multiClass']['OutputCodeClassifier'] = (multiclass.OutputCodeClassifier, 'integer') # (Error-Correcting) Output-Code multiclass strategy
  availImpl['multiClass']['fit_ovr'             ] = (multiclass.fit_ovr             , 'integer') # Fit a one-vs-the-rest strategy.
  availImpl['multiClass']['predict_ovr'         ] = (multiclass.predict_ovr         , 'integer') # Make predictions using the one-vs-the-rest strategy.
  availImpl['multiClass']['fit_ovo'             ] = (multiclass.fit_ovo             , 'integer') # Fit a one-vs-one strategy.
  availImpl['multiClass']['predict_ovo'         ] = (multiclass.predict_ovo         , 'integer') # Make predictions using the one-vs-one strategy.
  availImpl['multiClass']['fit_ecoc'            ] = (multiclass.fit_ecoc            , 'integer') # Fit an error-correcting output-code strategy.
  availImpl['multiClass']['predict_ecoc'        ] = (multiclass.predict_ecoc        , 'integer') # Make predictions using the error-correcting output-code strategy.

  availImpl['naiveBayes'] = {}
  availImpl['naiveBayes']['GaussianNB'   ] = (naive_bayes.GaussianNB   , 'float')
  availImpl['naiveBayes']['MultinomialNB'] = (naive_bayes.MultinomialNB, 'float')
  availImpl['naiveBayes']['BernoulliNB'  ] = (naive_bayes.BernoulliNB  , 'float')

  availImpl['neighbors'] = {}
  availImpl['neighbors']['NearestNeighbors']         = (neighbors.NearestNeighbors         , 'float'  )# Unsupervised learner for implementing neighbor searches.
  availImpl['neighbors']['KNeighborsClassifier']     = (neighbors.KNeighborsClassifier     , 'integer')# Classifier implementing the k-nearest neighbors vote.
  availImpl['neighbors']['RadiusNeighbors']          = (neighbors.RadiusNeighborsClassifier, 'integer')# Classifier implementing a vote among neighbors within a given radius
  availImpl['neighbors']['KNeighborsRegressor']      = (neighbors.KNeighborsRegressor      , 'float'  )# Regression based on k-nearest neighbors.
  availImpl['neighbors']['RadiusNeighborsRegressor'] = (neighbors.RadiusNeighborsRegressor , 'float'  )# Regression based on neighbors within a fixed radius.
  availImpl['neighbors']['NearestCentroid']          = (neighbors.NearestCentroid          , 'integer')# Nearest centroid classifier.
  availImpl['neighbors']['BallTree']                 = (neighbors.BallTree                 , 'float'  )# BallTree for fast generalized N-point problems
  availImpl['neighbors']['KDTree']                   = (neighbors.KDTree                   , 'float'  )# KDTree for fast generalized N-point problems

  availImpl['qda'] = {}
  availImpl['qda']['QDA'] = (qda.QDA, 'integer') #Quadratic Discriminant Analysis (QDA)

  availImpl['tree'] = {}
  availImpl['tree']['DecisionTreeClassifier'] = (tree.DecisionTreeClassifier, 'integer')# A decision tree classifier.
  availImpl['tree']['DecisionTreeRegressor' ] = (tree.DecisionTreeRegressor , 'float'  )# A tree regressor.
  availImpl['tree']['ExtraTreeClassifier'   ] = (tree.ExtraTreeClassifier   , 'integer')# An extremely randomized tree classifier.
  availImpl['tree']['ExtraTreeRegressor'    ] = (tree.ExtraTreeRegressor    , 'float'  )# An extremely randomized tree regressor.

  availImpl['GaussianProcess'] = {}
  availImpl['GaussianProcess']['GaussianProcess'] = (gaussian_process.GaussianProcess    , 'float'  )
  #test if a method to estimate the probability of the prediction is available
  qualityEstTypeDict = {}
  for key1, myDict in availImpl.items():
    qualityEstTypeDict[key1] = {}
    for key2 in myDict:
      qualityEstTypeDict[key1][key2] = []
      if  callable(getattr(myDict[key2][0], "predict_proba", None))  : qualityEstTypeDict[key1][key2] += ['probability']
      elif  callable(getattr(myDict[key2][0], "score"        , None)): qualityEstTypeDict[key1][key2] += ['score']
      else                                                           : qualityEstTypeDict[key1][key2] = False

  def __init__(self,**kwargs):
    superVisedLearning.__init__(self,**kwargs)
    self.printTag = utils.returnPrintTag('SCIKITLEARN')
    if 'SKLtype' not in self.initOptionDict.keys(): utils.raiseAnError(IOError,self,'to define a scikit learn ROM the SKLtype keyword is needed (from ROM '+self.name+')')
    SKLtype, SKLsubType = self.initOptionDict['SKLtype'].split('|')
    self.initOptionDict.pop('SKLtype')
    if not SKLtype in self.__class__.availImpl.keys(): utils.raiseAnError(IOError,self,'not known SKLtype ' + SKLtype +'(from ROM '+self.name+')')
    if not SKLsubType in self.__class__.availImpl[SKLtype].keys(): utils.raiseAnError(IOError,self,'not known SKLsubType ' + SKLsubType +'(from ROM '+self.name+')')
    self.__class__.returnType     = self.__class__.availImpl[SKLtype][SKLsubType][1]
    self.ROM                      = self.__class__.availImpl[SKLtype][SKLsubType][0]()
    self.__class__.qualityEstType = self.__class__.qualityEstTypeDict[SKLtype][SKLsubType]
    for key,value in self.initOptionDict.items():
      try:self.initOptionDict[key] = ast.literal_eval(value)
      except: pass
    self.ROM.set_params(**self.initOptionDict)

  def _readdressEvaluateConstResponse(self,edict):
    """
    Method to re-address the evaluate base class method in order to avoid wasting time
    in case the training set has an unique response (e.g. if 10 points in the training set,
    and the 10 outcomes are all == to 1, this method returns one without the need of an
    evaluation)
    @ In, prediction request, Not used in this method (kept the consistency with evaluate method)
    """
    return self.myNumber

  def _readdressEvaluateRomResponse(self,edict):
    """
    Method to re-address the evaluate base class method to its original method
    @ In, prediction request, used in this method (kept the consistency with evaluate method)
    """
    return self.__class__.evaluate(self,edict)

  def __trainLocal__(self,featureVals,targetVals):
    """
    Perform training on samples in featureVals with responses y.
    For an one-class model, +1 or -1 is returned.
    Parameters
    ----------
    featureVals : {array-like, sparse matrix}, shape = [n_samples, n_features]
    Returns
    -------
    targetVals : array, shape = [n_samples]
    """
    #If all the target values are the same no training is needed and the moreover the self.evaluate could be re-addressed to this value
    if len(np.unique(targetVals))>1:
      self.ROM.fit(featureVals,targetVals)
      self.evaluate = self._readdressEvaluateRomResponse
      #self.evaluate = lambda edict : self.__class__.evaluate(self,edict)
    else:
      self.myNumber = np.unique(targetVals)[0]
      self.evaluate = self._readdressEvaluateConstResponse

  def __confidenceLocal__(self,edict):
    if  'probability' in self.__class__.qualityEstType: return self.ROM.predict_proba(edict)
    else            : utils.raiseAnError(IOError,self,'the ROM '+str(self.name)+'has not the an method to evaluate the confidence of the prediction')

  def __evaluateLocal__(self,featureVals):
    return self.ROM.predict(featureVals)

  def __resetLocal__(self):
    self.ROM = self.ROM.__class__(**self.initOptionDict)

  def __returnInitialParametersLocal__(self):
    return self.ROM.get_params()

  def __returnCurrentSettingLocal__(self):
    utils.raiseAMessage(self,'here we need to collect some info on the ROM status','FIXME')
    localInitParam = {}
    return localInitParam
#
#
#
__interfaceDict                         = {}
__interfaceDict['NDspline'            ] = NDsplineRom
__interfaceDict['NDinvDistWeigth'     ] = NDinvDistWeigth
__interfaceDict['microSphere'         ] = NDmicroSphere
__interfaceDict['SciKitLearn'         ] = SciKitLearn
__interfaceDict['GaussPolynomialRom'  ] = GaussPolynomialRom
__interfaceDict['HDMRRom'             ] = HDMRRom
__base                                  = 'superVisedLearning'

def addToInterfaceDict(newDict):
  for key,val in newDict.items():
    __interfaceDict[key]=val

def returnInstance(ROMclass,**kwargs):
  '''This function return an instance of the request model type'''
  try: return __interfaceDict[ROMclass](**kwargs)
  except KeyError: utils.raiseAnError(NameError,'SUPERVISEDLEARNING','not known '+__base+' type '+str(ROMclass))

def returnClass(ROMclass):
  '''This function return an instance of the request model type'''
  try: return __interfaceDict[ROMclass]
  except KeyError: utils.raiseAnError(NameError,'SUPERVISEDLEARNING','not known '+__base+' type '+ROMclass)
