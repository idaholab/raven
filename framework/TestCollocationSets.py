#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import numpy as np
import sys, os

from utils import find_crow

find_crow(os.path.dirname(os.path.abspath(sys.argv[0])))


import Distributions
import Quadrature

print (Distributions)
def createElement(tag,attrib={},text={}):
  element = ET.Element(tag,attrib)
  element.text = text
  return element

results = {"pass":0,"fail":0}

def floatNotEqual(a,b):
  return abs(a - b) > 1e-10


def checkAnswer(comment,value,expected):
  if floatNotEqual(value, expected):
    print(comment,value,"!=",expected)
    results["fail"] += 1
  else:
    results["pass"] += 1

def checkObject(comment,value,expected):
  if value!=expected:
    print(comment,value,"!=",expected)
    results['fail']+=1
  else: results['pass']+=1

def testPoly(n,y):
  return y**n

## shared shape parameters ##
laguerre_alpha = 2.0

## generate quadratures ##
quads={}

legendreElement = ET.Element("legendre")
legendre = Quadrature.Legendre()
legendre._readMoreXML(legendreElement)
legendre.initialize()
quads['Legendre']=legendre

cdfElement = ET.Element("cdf")
cdf = Quadrature.CDF()
cdf._readMoreXML(cdfElement)
cdf.initialize()
quads['CDF']=cdf

ccElement = ET.Element("clenshawcurtis")
cc = Quadrature.ClenshawCurtis()
cc._readMoreXML(ccElement)
cc.initialize()
quads['ClenshawCurtis']=cc

hermiteElement = ET.Element('hermite')
hermite = Quadrature.Hermite()
hermite._readMoreXML(hermiteElement)
hermite.initialize()
quads['Hermite']=hermite

laguerreElement = ET.Element('laguerre')
laguerreElement.append(createElement("alpha",text="%f" %laguerre_alpha))
laguerre = Quadrature.Laguerre()
laguerre._readMoreXML(laguerreElement)
laguerre.initialize()
quads['Laguerre']=laguerre
## END make quadratures ##

###########################################
#            Tests for Uniform            #
###########################################

#make distribution
uniformElement = ET.Element("uniform")
L = 1.0
R = 5.0
uniformElement.append(createElement("low",text="%f" %L))
uniformElement.append(createElement("hi",text="%f" %R))

uniform = Distributions.Uniform()
uniform._readMoreXML(uniformElement)
uniform.initializeDistribution()

ptset = [(-1  , 1),
         (-0.5, 2),
         ( 0  , 3),
         ( 0.5, 4),
         ( 1  , 5)]

doQuads = ['Legendre','CDF','ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Uniform-%s...' %quadname)
  #link quadrature to distr
  uniform.setQuadrature(quad)
  checkObject("setting %s collocation in uniform" %quadname,uniform.quadratureSet(),quad)

  #test points and weights conversion
  for p,pt in enumerate(ptset):
    checkAnswer("uniform-%s std-to-act pt (%i)" %(quadname,pt[0]),uniform.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("uniform-%s act-to-std pt (%i)" %(quadname,pt[1]),uniform.convertDistrPointsToStd(pt[1]),pt[0])

  #test quadrature integration
  for i in range(1,6):
    pts,wts = quad(i)
    pts = uniform.convertStdPointsToDistr(pts)
    totu=0
    if i>=2  :tot2=0
    if i>=3  :tot5=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*uniform.probabilityNorm()
      if i>=2  :tot2+=testPoly(2,pt)*wts[p]*uniform.probabilityNorm()
      if i>=3  :tot5+=testPoly(5,pt)*wts[p]*uniform.probabilityNorm()
    checkAnswer("%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    if i>=2  :checkAnswer("uniform-%s integrate y^2 P(y)dy with O(%i)" %(quadname,i),tot2,31./3.)
    if i>=3  :checkAnswer("uniform-%s integrate y^5 P(y)dy with O(%i)" %(quadname,i),tot5,3906./6.)

## END Uniform tests ##

##########################################
#            Tests for Normal            #
##########################################

#make distrubtion
normalElement = ET.Element("normal")
mean=1.0
stdv=2.0
normalElement.append(createElement("mean",text="%f"%mean))
normalElement.append(createElement("sigma",text="%f"%stdv))
normalElement.append(createElement("lowerBound",text="-1e50"))
normalElement.append(createElement("upperBound",text= "1e50"))

normal = Distributions.Normal()
normal._readMoreXML(normalElement)
normal.initializeDistribution()

ptset = [(-2,-3),
         (-1,-1),
         ( 0, 1),
         ( 1, 3),
         ( 2, 5)]

doQuads = ['Hermite','CDF']#,'ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Normal-%s...' %quadname)
  #link quadrature to distr
  normal.setQuadrature(quad)
  checkObject("setting %s collocation in normal",normal.quadratureSet(),quad)

  #test points and weights conversion
  if quadname not in ['CDF','ClenshawCurtis']:
    #TODO CDF and CC use 0..1 point set, so can't do this check
    #  Maybe should each quadrature know its own test set?
    for p,pt in enumerate(ptset):
      checkAnswer("normal-%s std-to-act pt (%i)" %(quadname,pt[0]),normal.convertStdPointsToDistr(pt[0]),pt[1])
      checkAnswer("normal-%s act-to-std pt (%i)" %(quadname,pt[1]),normal.convertDistrPointsToStd(pt[1]),pt[0])

  #test quadrature integration
  for i in range(1,6):
    pts,wts = quad(i)
    pts = normal.convertStdPointsToDistr(pts)
    #print('\norder',i)
    #for p,pt in enumerate(pts):
    #  print (pt,wts[p])
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot4=0
    tot6=0
    for p,pt in enumerate(pts):
      #print('pt',pt)
      totu+=wts[p]*normal.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*normal.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*normal.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*normal.probabilityNorm()
      tot4+=testPoly(3,pt)*wts[p]*normal.probabilityNorm()
      tot6+=testPoly(4,pt)*wts[p]*normal.probabilityNorm()
    checkAnswer(        "normal-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "normal-%s integrate x^0*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot0,1)
    if i>=1:checkAnswer("normal-%s integrate x^1*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot1,1)
    #TODO CDF case, this converges but isn't exact
    #if i>=2:checkAnswer("normal-%s integrate x^2*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot2,5)
    #if i>=3:checkAnswer("normal-%s integrate x^3*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot4,13)
    #if i>=4:checkAnswer("normal-%s integrate x^4*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot6,73)

##################################################
#            Tests for Truncated Norm            #
##################################################
truncNormalElement = ET.Element("truncnorm")
truncNormalElement.append(createElement("mean",text="1.0"))
truncNormalElement.append(createElement("sigma",text="2.0"))
truncNormalElement.append(createElement("lowerBound",text="-1.0"))
truncNormalElement.append(createElement("upperBound",text="3.0"))

truncNormal = Distributions.Normal()
truncNormal._readMoreXML(truncNormalElement)
truncNormal.initializeDistribution()

ptset = [(-2,-3),
         (-1,-1),
         ( 0, 1),
         ( 1, 3),
         ( 2, 5)]

doQuads = ['Hermite','CDF','ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing TruncNormal-%s...' %quadname)
  #link quadrature to distr
  truncNormal.setQuadrature(quad)
  checkObject("setting %s collocation in truncNormal",truncNormal.quadratureSet(),quad)

  #test points and weights conversion
  if quadname not in ['CDF','ClenshawCurtis']:
    #TODO CDF and CC use 0..1 point set, so can't do this check
    #  Maybe should each quadrature know its own test set?
    for p,pt in enumerate(ptset):
      checkAnswer("truncNormal-%s std-to-act pt (%i)" %(quadname,pt[0]),truncNormal.convertStdPointsToDistr(pt[0]),pt[1])
      checkAnswer("truncNormal-%s act-to-std pt (%i)" %(quadname,pt[1]),truncNormal.convertDistrPointsToStd(pt[1]),pt[0])

  #test quadrature integration
  for i in range(1,6):
    pts,wts = quad(i)
    pts = truncNormal.convertStdPointsToDistr(pts)
    #print('\norder',i)
    #for p,pt in enumerate(pts):
    #  print (pt,wts[p])
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot4=0
    tot6=0
    for p,pt in enumerate(pts):
      #print('pt',pt)
      totu+=wts[p]*truncNormal.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*truncNormal.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*truncNormal.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*truncNormal.probabilityNorm()
      tot4+=testPoly(3,pt)*wts[p]*truncNormal.probabilityNorm()
      tot6+=testPoly(4,pt)*wts[p]*truncNormal.probabilityNorm()
    checkAnswer(        "truncNormal-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "truncNormal-%s integrate x^0*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot0,1)
    if i>=1:checkAnswer("truncNormal-%s integrate x^1*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot1,0)
    #TODO ClenshawCurtis gets 0.797884560803, which looks like the actual integral of the truncated portion
    #if i>=2:checkAnswer("truncNormal-%s integrate x^2*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot2,5)
    #if i>=3:checkAnswer("truncNormal-%s integrate x^3*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot4,13)
    #if i>=4:checkAnswer("truncNormal-%s integrate x^4*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot6,73)

#########################################
#            Tests for Gamma            #
#########################################

#make distribution
gammaElement = ET.Element("gamma")
low = -2.0
beta = 3.0
gammaElement.append(createElement("low",text="%f" %low))
gammaElement.append(createElement("alpha",text="%f" %laguerre_alpha))
gammaElement.append(createElement("beta",text="%f" %beta))

gamma = Distributions.Gamma()
gamma._readMoreXML(gammaElement)
gamma.initializeDistribution()

ptset = [( 0,-2),
         ( 3,-1),
         ( 6, 0),
         ( 9, 1),
         (12, 2)]

doQuads=['Laguerre','CDF']#,'ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads:continue
  print('Testing Gamma-%s...' %quadname)
  #set quadrature to distr
  gamma.setQuadrature(quad)
  checkObject("setting %s collocation in gamma" %quadname,gamma.quadratureSet(),quad)

  #test points and weights conversion
  if quadname not in ['CDF','ClenshawCurtis']:
    for p,pt in enumerate(ptset):
      checkAnswer("Gamma-%s std-to-act pt (%i)" %(quadname,pt[0]),gamma.convertStdPointsToDistr(pt[0]),pt[1])
      checkAnswer("Gamma-%s act-to-std pt (%i)" %(quadname,pt[1]),gamma.convertDistrPointsToStd(pt[1]),pt[0])

  #test quadrature integration
  for i in [int(1e1)]:
    pts,wts = quad(i)
    pts = gamma.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*gamma.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*gamma.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*gamma.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*gamma.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*gamma.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*gamma.probabilityNorm()
    checkAnswer(        "Gamma-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Gamma-%s integrate x^0*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot0,1)
    if i>=1:checkAnswer("Gamma-%s integrate x^1*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot1,-4./3.)
    if i>=2:checkAnswer("Gamma-%s integrate x^2*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot2,2)
    if i>=3:checkAnswer("Gamma-%s integrate x^3*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot3,-28./9.)
    if i>=4:checkAnswer("Gamma-%s integrate x^4*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot4,136./27.)

##############################################
#            Test Jacobi for Beta            #
##############################################
#Test Beta

#betaElement = ET.Element("beta")
#betaElement.append(createElement("low",text="0.0"))
#betaElement.append(createElement("hi",text="1.0"))
#betaElement.append(createElement("alpha",text="5.0"))
#betaElement.append(createElement("beta",text="2.0"))
#
#beta = Distributions.Beta()
#beta._readMoreXML(betaElement)
#beta.initializeDistribution()
#
#checkCrowDist("beta",beta,{'scale': 1.0, 'beta': 2.0, 'xMax': 1.0, 'xMin': 0.0, 'alpha': 5.0, 'type': 'BetaDistribution'})
#
#checkAnswer("beta cdf(0.1)",beta.cdf(0.1),5.5e-05)
#checkAnswer("beta cdf(0.5)",beta.cdf(0.5),0.109375)
#checkAnswer("beta cdf(0.9)",beta.cdf(0.9),0.885735)
#
#checkAnswer("beta ppf(0.1)",beta.ppf(0.1),0.489683693449)
#checkAnswer("beta ppf(0.5)",beta.ppf(0.5),0.735550016704)
#checkAnswer("beta ppf(0.9)",beta.ppf(0.9),0.907404741087)
#
#checkAnswer("beta mean()",beta.untruncatedMean(),5.0/(5.0+2.0))
#checkAnswer("beta median()",beta.untruncatedMedian(),0.735550016704)
#checkAnswer("beta mode()",beta.untruncatedMode(),(5.0-1)/(5.0+2.0-2))
#
#checkAnswer("beta pdf(0.25)",beta.pdf(0.25),0.087890625)
#checkAnswer("beta cdfComplement(0.25)",beta.untruncatedCdfComplement(0.25),0.995361328125)
#checkAnswer("beta hazard(0.25)",beta.untruncatedHazard(0.25),0.0883002207506)
#
#print(beta.rvs(5),beta.rvs())
#
##Test Beta Scaled
#
#betaElement = ET.Element("beta")
#betaElement.append(createElement("low",text="0.0"))
#betaElement.append(createElement("hi",text="4.0"))
#betaElement.append(createElement("alpha",text="5.0"))
#betaElement.append(createElement("beta",text="1.0"))
#
#beta = Distributions.Beta()
#beta._readMoreXML(betaElement)
#beta.initializeDistribution()
#
#checkCrowDist("scaled beta",beta,{'scale': 4.0, 'beta': 1.0, 'xMax': 4.0, 'xMin': 0.0, 'alpha': 5.0, 'type': 'BetaDistribution'})
#
#checkAnswer("scaled beta cdf(0.1)",beta.cdf(0.1),9.765625e-09)
#checkAnswer("scaled beta cdf(0.5)",beta.cdf(0.5),3.0517578125e-05)
#checkAnswer("scaled beta cdf(0.9)",beta.cdf(0.9),0.000576650390625)
#
#checkAnswer("scaled beta ppf(0.1)",beta.ppf(0.1),2.52382937792)
#checkAnswer("scaled beta ppf(0.5)",beta.ppf(0.5),3.48220225318)
#checkAnswer("scaled beta ppf(0.9)",beta.ppf(0.9),3.91659344944)
#
#print(beta.rvs(5),beta.rvs())

##############################################
#            Tests for Triangular            #
##############################################

#make distribution
triangularElement = ET.Element("triangular")
a=1.0
b=5.0
c=4.0
triangularElement.append(createElement("min",text="%f" %a))
triangularElement.append(createElement("apex",text="%f" %c))
triangularElement.append(createElement("max",text="%f" %b))

triangular = Distributions.Triangular()
triangular._readMoreXML(triangularElement)
triangular.initializeDistribution()

ptset = [(-1, 1),
         (-0.5, 2.73205080757),
         (0, 3.44948974278),
         (0.5, 4),
         (1, 5)]

doQuads = ['CDF','ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Triangular-%s...' %quadname)
  #link quad to distr
  triangular.setQuadrature(quad)
  checkObject("setting %s collocation in triangular" %quadname,triangular.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("triangular-%s std-to-act pt (%f)" %(quadname,pt[0]),triangular.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("triangular-%s act-to-std pt (%i)" %(quadname,pt[1]),triangular.convertDistrPointsToStd(pt[1]),pt[0])

  for i in range(14,15):
    pts,wts = quad(i)
    pts = triangular.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*triangular.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*triangular.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*triangular.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*triangular.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*triangular.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*triangular.probabilityNorm()
    checkAnswer(        "Triangular-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Triangular-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    if i>=1:checkAnswer("Triangular-%s integrate x^1 with O(%i)" %(quadname,i),tot1,10./3.) #CC needs p(12)==O(2**12+1)
    #print('Triangular-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Triangular-%s integrate x^2 with O(%i)" %(quadname,i),tot2,71./6.) #CC needs p(13)
    if i>=1:checkAnswer("Triangular-%s integrate x^3 with O(%i)" %(quadname,i),tot3,1760./40.) #CC needs p(13)
    if i>=1:checkAnswer("Triangular-%s integrate x^4 with O(%i)" %(quadname,i),tot4,847./5.) #CC needs p(14)

###########################################
#            Tests for Poisson            #
###########################################
#make distribution
poissonElement = ET.Element("poisson")
poissonElement.append(createElement("mu",text="4.0"))
poisson = Distributions.Poisson()
poisson._readMoreXML(poissonElement)
poisson.initializeDistribution()

#TODO confirm these
ptset = [(-1  , 0),
         (-0.5, 2),
         ( 0  , 4),
         ( 0.5, 5),
         ( 0.9, 8)] #1.0 is system max

doQuads = []#'CDF','ClenshawCurtis']
#doQuads = ['CDF']#,'ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Poisson-%s...' %quadname)
  #link quad to distr
  poisson.setQuadrature(quad)
  checkObject("setting %s collocation in poisson" %quadname,poisson.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("poisson-%s std-to-act pt (%f)" %(quadname,pt[0]),poisson.convertStdPointsToDistr(pt[0]),pt[1])
    #checkAnswer("poisson-%s act-to-std pt (%i)" %(quadname,pt[1]),poisson.convertDistrPointsToStd(pt[1]),pt[0])

  for i in [20]:#range(1,6):
    pts,wts = quad(i)
    pts = poisson.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*poisson.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*poisson.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*poisson.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*poisson.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*poisson.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*poisson.probabilityNorm()
    checkAnswer(        "Poisson-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Poisson-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    #if i>=1:checkAnswer("Poisson-%s integrate x^1 with O(%i)" %(quadname,i),tot1,4.0) #CC needs p(12)==O(2**12+1)
    #print('Poisson-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Poisson-%s integrate x^2 with O(%i)" %(quadname,i),tot2,20.) #CC needs p(13)
    #if i>=1:checkAnswer("Poisson-%s integrate x^3 with O(%i)" %(quadname,i),tot3,116.) #CC needs p(13)
    #if i>=1:checkAnswer("Poisson-%s integrate x^4 with O(%i)" %(quadname,i),tot4,756.) #CC needs p(14)
    #TODO fix these

############################################
#            Tests for Binomial            #
############################################
#make distribution
binomialElement = ET.Element("binomial")
binomialElement.append(createElement("n",text="10"))
binomialElement.append(createElement("p",text="0.25"))
binomial = Distributions.Binomial()
binomial._readMoreXML(binomialElement)
binomial.initializeDistribution()

#TODO fix these
ptset = [(-1, 1),
         (-0.5, 2.73205080757),
         (0, 3.44948974278),
         (0.5, 4),
         (1, 5)]

doQuads = []#'CDF','ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Binomial-%s...' %quadname)
#link quad to distr
  binomial.setQuadrature(quad)
  checkObject("setting %s collocation in binomial" %quadname,binomial.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("binomial-%s std-to-act pt (%f)" %(quadname,pt[0]),binomial.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("binomial-%s act-to-std pt (%i)" %(quadname,pt[1]),binomial.convertDistrPointsToStd(pt[1]),pt[0])

  for i in range(1,6):
    pts,wts = quad(i)
    pts = binomial.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    print('pts:',pts)
    for p,pt in enumerate(pts):
      totu+=wts[p]*binomial.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*binomial.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*binomial.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*binomial.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*binomial.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*binomial.probabilityNorm()
    checkAnswer(        "Binomial-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Binomial-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    if i>=1:checkAnswer("Binomial-%s integrate x^1 with O(%i)" %(quadname,i),tot1,10./3.) #CC needs p(12)==O(2**12+1)
    #print('Binomial-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Binomial-%s integrate x^2 with O(%i)" %(quadname,i),tot2,71./6.) #CC needs p(13)
    if i>=1:checkAnswer("Binomial-%s integrate x^3 with O(%i)" %(quadname,i),tot3,1760./40.) #CC needs p(13)
    if i>=1:checkAnswer("Binomial-%s integrate x^4 with O(%i)" %(quadname,i),tot4,847./5.) #CC needs p(14)
    #TODO fix these


#############################################
#            Tests for Bernoulli            #
#############################################
#make distribution
bernoulliElement = ET.Element("bernoulli")
bernoulliElement.append(createElement("p",text="0.4"))
bernoulli = Distributions.Bernoulli()
bernoulli._readMoreXML(bernoulliElement)
bernoulli.initializeDistribution()

#TODO fix these
ptset = [(-1, 1),
         (-0.5, 2.73205080757),
         (0, 3.44948974278),
         (0.5, 4),
         (1, 5)]

doQuads = []#'CDF','ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Bernoulli-%s...' %quadname)
#link quad to distr
  bernoulli.setQuadrature(quad)
  checkObject("setting %s collocation in bernoulli" %quadname,bernoulli.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("bernoulli-%s std-to-act pt (%f)" %(quadname,pt[0]),bernoulli.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("bernoulli-%s act-to-std pt (%i)" %(quadname,pt[1]),bernoulli.convertDistrPointsToStd(pt[1]),pt[0])

  for i in range(1,6):
    pts,wts = quad(i)
    pts = bernoulli.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    print('pts:',pts)
    for p,pt in enumerate(pts):
      totu+=wts[p]*bernoulli.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*bernoulli.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*bernoulli.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*bernoulli.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*bernoulli.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*bernoulli.probabilityNorm()
    checkAnswer(        "Bernoulli-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Bernoulli-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    if i>=1:checkAnswer("Bernoulli-%s integrate x^1 with O(%i)" %(quadname,i),tot1,10./3.) #CC needs p(12)==O(2**12+1)
    #print('Bernoulli-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Bernoulli-%s integrate x^2 with O(%i)" %(quadname,i),tot2,71./6.) #CC needs p(13)
    if i>=1:checkAnswer("Bernoulli-%s integrate x^3 with O(%i)" %(quadname,i),tot3,1760./40.) #CC needs p(13)
    if i>=1:checkAnswer("Bernoulli-%s integrate x^4 with O(%i)" %(quadname,i),tot4,847./5.) #CC needs p(14)
    #TODO fix these

#############################################
#            Tests for Logistic            #
#############################################
#make distribution
logisticElement = ET.Element("logistic")
logisticElement.append(createElement("location",text="4.0"))
logisticElement.append(createElement("scale",text="1.0"))
logistic = Distributions.Logistic()
logistic._readMoreXML(logisticElement)
logistic.initializeDistribution()

#TODO fix these
ptset = [(-0.9, 1.05556102083),
         (-0.5, 2.90138771133),
         (   0, 4.0          ),
         ( 0.5, 5.09861228867),
         ( 0.9, 6.94443897917)]

doQuads = ['CDF']#,'ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Logistic-%s...' %quadname)
#link quad to distr
  logistic.setQuadrature(quad)
  checkObject("setting %s collocation in logistic" %quadname,logistic.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("logistic-%s std-to-act pt (%f)" %(quadname,pt[0]),logistic.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("logistic-%s act-to-std pt (%i)" %(quadname,pt[1]),logistic.convertDistrPointsToStd(pt[1]),pt[0])

  for i in range(1,16,2):
    pts,wts = quad(i)
    pts = logistic.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*logistic.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*logistic.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*logistic.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*logistic.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*logistic.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*logistic.probabilityNorm()
    checkAnswer(        "Logistic-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Logistic-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    if i>=1:checkAnswer("Logistic-%s integrate x^1 with O(%i)" %(quadname,i),tot1,4.)
    #print('Logistic-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Logistic-%s integrate x^2 with O(%i)" %(quadname,i),tot2,19.2898681336964528729448303) #CC needs p(13)
    if i>=1:checkAnswer("Logistic-%s integrate x^3 with O(%i)" %(quadname,i),tot3,103.478417604357434475337964) #CC needs p(13)
    if i>=1:checkAnswer("Logistic-%s integrate x^4 with O(%i)" %(quadname,i),tot4,617.284916650727279846375867) #CC needs p(14)
    #TODO fix these

###############################################
#            Tests for Exponential            #
###############################################
#TODO this is a subset of gamma, so use laguerre quad
#make distribution
exponentialElement = ET.Element("exponential")
exponentialElement.append(createElement("lambda",text="5.0"))
exponential = Distributions.Exponential()
exponential._readMoreXML(exponentialElement)
exponential.initializeDistribution()

#TODO fix these
ptset = [(-1, 1),
         (-0.5, 2.73205080757),
         (0, 3.44948974278),
         (0.5, 4),
         (1, 5)]

#TODO need to implement shifted exponential with self.low like I did for gamma
#doQuads = ['Laguerre','CDF']#,'ClenshawCurtis']
doQuads = []#'Laguerre','CDF']#,'ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Exponential-%s...' %quadname)
#link quad to distr
  exponential.setQuadrature(quad)
  checkObject("setting %s collocation in exponential" %quadname,exponential.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("exponential-%s std-to-act pt (%f)" %(quadname,pt[0]),exponential.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("exponential-%s act-to-std pt (%i)" %(quadname,pt[1]),exponential.convertDistrPointsToStd(pt[1]),pt[0])

  for i in range(1,6):
    pts,wts = quad(i)
    pts = exponential.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    print('pts:',pts)
    for p,pt in enumerate(pts):
      totu+=wts[p]*exponential.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*exponential.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*exponential.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*exponential.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*exponential.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*exponential.probabilityNorm()
    checkAnswer(        "Exponential-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Exponential-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    if i>=1:checkAnswer("Exponential-%s integrate x^1 with O(%i)" %(quadname,i),tot1,10./3.) #CC needs p(12)==O(2**12+1)
    #print('Exponential-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Exponential-%s integrate x^2 with O(%i)" %(quadname,i),tot2,71./6.) #CC needs p(13)
    if i>=1:checkAnswer("Exponential-%s integrate x^3 with O(%i)" %(quadname,i),tot3,1760./40.) #CC needs p(13)
    if i>=1:checkAnswer("Exponential-%s integrate x^4 with O(%i)" %(quadname,i),tot4,847./5.) #CC needs p(14)
    #TODO fix these


#############################################
#            Tests for Lognormal            #
#############################################
#make distribution
lognormalElement = ET.Element("logNormal")
lognormalElement.append(createElement("mean",text="3.0"))
lognormalElement.append(createElement("sigma",text="2.0"))
lognormal = Distributions.LogNormal()
lognormal._readMoreXML(lognormalElement)
lognormal.initializeDistribution()

#TODO fix these
ptset = [(-1  ,  0),
         (-0.5,  5.2122962603),
         (   0, 20.0855369232),
         ( 0.5, 77.3994365143)]

doQuads = ['CDF']#,'ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Lognormal-%s...' %quadname)
#link quad to distr
  lognormal.setQuadrature(quad)
  checkObject("setting %s collocation in lognormal" %quadname,lognormal.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("lognormal-%s std-to-act pt (%f)" %(quadname,pt[0]),lognormal.convertStdPointsToDistr(pt[0]),pt[1])
    #checkAnswer("lognormal-%s act-to-std pt (%i)" %(quadname,pt[1]),lognormal.convertDistrPointsToStd(pt[1]),pt[0])

  for i in range(1,21,2):
    pts,wts = quad(i)
    pts = lognormal.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*lognormal.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*lognormal.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*lognormal.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*lognormal.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*lognormal.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*lognormal.probabilityNorm()
    checkAnswer(        "Lognormal-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Lognormal-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    if i>=1:checkAnswer("Lognormal-%s integrate x^1 with O(%i)" %(quadname,i),tot1,np.exp(5.)) #CC needs p(12)==O(2**12+1)
    #print('Lognormal-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Lognormal-%s integrate x^2 with O(%i)" %(quadname,i),tot2,np.exp(14.)) #CC needs p(13)
    if i>=1:checkAnswer("Lognormal-%s integrate x^3 with O(%i)" %(quadname,i),tot3,np.exp(27.)) #CC needs p(13)
    if i>=1:checkAnswer("Lognormal-%s integrate x^4 with O(%i)" %(quadname,i),tot4,np.exp(44.)) #CC needs p(14)
    #TODO fix these



###########################################
#            Tests for Weibull            #
###########################################
#make distribution
weibullElement = ET.Element("weibull")
weibullElement.append(createElement("k", text="1.5"))
weibullElement.append(createElement("lambda", text="1.0"))
weibull = Distributions.Weibull()
weibull._readMoreXML(weibullElement)
weibull.initializeDistribution()

#TODO fix these
ptset = [(-1, 1),
         (-0.5, 2.73205080757),
         (0, 3.44948974278),
         (0.5, 4),
         (1, 5)]

doQuads = ['CDF']#,'ClenshawCurtis']

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  print('Testing Weibull-%s...' %quadname)
#link quad to distr
  weibull.setQuadrature(quad)
  checkObject("setting %s collocation in weibull" %quadname,weibull.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("weibull-%s std-to-act pt (%f)" %(quadname,pt[0]),weibull.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("weibull-%s act-to-std pt (%i)" %(quadname,pt[1]),weibull.convertDistrPointsToStd(pt[1]),pt[0])

  for i in range(1,32,5):
    pts,wts = quad(i)
    pts = weibull.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*weibull.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*weibull.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*weibull.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*weibull.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*weibull.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*weibull.probabilityNorm()
    checkAnswer(        "Weibull-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Weibull-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    if i>=1:checkAnswer("Weibull-%s integrate x^1 with O(%i)" %(quadname,i),tot1,0.902745292951)
    #print('Weibull-CDF, Quad order %i, err: %1.5e' %(i,tot1-10./3.))
    if i>=1:checkAnswer("Weibull-%s integrate x^2 with O(%i)" %(quadname,i),tot2,1.190639348759)
    if i>=1:checkAnswer("Weibull-%s integrate x^3 with O(%i)" %(quadname,i),tot3,2.)
    if i>=1:checkAnswer("Weibull-%s integrate x^4 with O(%i)" %(quadname,i),tot4,4.01220130200415)
    #TODO fix these

sys.exit(results["fail"])
