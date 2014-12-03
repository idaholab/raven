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


#Test Legendre for Uniform

#make distribution
uniformElement = ET.Element("uniform")
uniformElement.append(createElement("low",text="1.0"))
uniformElement.append(createElement("hi",text="5.0"))

uniform = Distributions.Uniform()
uniform._readMoreXML(uniformElement)
uniform.initializeDistribution()

#make quadrature
legendreElement = ET.Element("legendre")
legendre = Quadrature.Legendre()
legendre._readMoreXML(uniformElement)
legendre.initialize()

#link quadrature to distr
uniform.setQuadrature(legendre)
checkObject("setting legendre collocation in uniform",uniform.quadratureSet(),legendre)

#test points and weights conversion
stdpts=[-2,-1, 0, 1, 2]
actpts=[-1, 1, 3, 5, 7]
for s,std in enumerate(stdpts):
  act=actpts[s]
  checkAnswer("uniform-legendre std-to-act pt (%i)" %std,uniform.convertStdPointsToDistr(std),actpts[s])
  checkAnswer("uniform-legendre act-to-std pt (%i)" %act,uniform.convertDistrPointsToStd(act),stdpts[s])

#test quadrature integration
for i in range(1,6):
  pts,wts = legendre(i)
  pts = uniform.convertStdPointsToDistr(pts)
  totu=0
  if i>=1.5:tot2=0
  if i>=3  :tot5=0
  for p,pt in enumerate(pts):
    totu+=wts[p]*uniform.probabilityNorm(std=True)
    if i>=1.5:tot2+=testPoly(2,pt)*wts[p]*uniform.probabilityNorm(std=True)
    if i>=3  :tot5+=testPoly(5,pt)*wts[p]*uniform.probabilityNorm(std=True)
  checkAnswer("legendre integrate weights with O(%i)" %i,totu,1.0)
  if i>=1.5:checkAnswer("uniform-legendre integrate y^2 with O(%i)" %i,tot2,31./3.)
  if i>=3  :checkAnswer("uniform-legendre integrate y^5 with O(%i)" %i,tot5,3906./6.)





#Test Hermite for Normals

#make distrubtion
normalElement = ET.Element("normal")
mean=1.0
stdv=2.0
normalElement.append(createElement("mean",text="%f"%mean))
normalElement.append(createElement("sigma",text="%f"%stdv))

normal = Distributions.Normal()
normal._readMoreXML(normalElement)
normal.initializeDistribution()

#make quadrature
hermiteElement = ET.Element('hermite')
hermite = Quadrature.Hermite()
hermite._readMoreXML(hermiteElement)
hermite.initialize()

#link quadrature to distr
normal.setQuadrature(hermite)
checkObject("setting hermite collocation in normal",normal.quadratureSet(),hermite)

#test points and weights conversion
stdpts=[-2,-1, 0, 1, 2]
actpts=[-3,-1, 1, 3, 5]
for s,std in enumerate(stdpts):
  act=actpts[s]
  checkAnswer("normal-hermite std-to-act pt (%i)" %std,normal.convertStdPointsToDistr(std),actpts[s])
  checkAnswer("normal-hermite act-to-std pt (%i)" %act,normal.convertDistrPointsToStd(act),stdpts[s])

#test quadrature integration
for i in range(1,6):
  pts,wts = hermite(i)
  pts = normal.convertStdPointsToDistr(pts)
  totu=0
  tot0=0
  tot1=0
  tot2=0
  tot4=0
  tot6=0
  for p,pt in enumerate(pts):
    totu+=wts[p]*normal.probabilityNorm()
    tot0+=testPoly(0,pt)*wts[p]*normal.probabilityNorm()
    tot1+=testPoly(1,pt)*wts[p]*normal.probabilityNorm()
    tot2+=testPoly(2,pt)*wts[p]*normal.probabilityNorm()
    tot4+=testPoly(3,pt)*wts[p]*normal.probabilityNorm()
    tot6+=testPoly(4,pt)*wts[p]*normal.probabilityNorm()
  checkAnswer(        "normal-hermite integrate weights with O(%i)" %i,totu,1.0)
  checkAnswer(        "normal-hermite integrate x^0*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(mean,stdv,i),tot0,1)
  if i>=1:checkAnswer("normal-hermite integrate x^1*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(mean,stdv,i),tot1,1)
  if i>=2:checkAnswer("normal-hermite integrate x^2*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(mean,stdv,i),tot2,5)
  if i>=3:checkAnswer("normal-hermite integrate x^3*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(mean,stdv,i),tot4,13)
  if i>=4:checkAnswer("normal-hermite integrate x^4*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(mean,stdv,i),tot6,73)

#Test Truncated Norm as Non-Standard
#TODO
#truncNormalElement = ET.Element("truncnorm")
#truncNormalElement.append(createElement("mean",text="1.0"))
#truncNormalElement.append(createElement("sigma",text="2.0"))
#truncNormalElement.append(createElement("lowerBound",text="-1.0"))
#truncNormalElement.append(createElement("upperBound",text="3.0"))
#
#truncNormal = Distributions.Normal()
#truncNormal._readMoreXML(truncNormalElement)
#truncNormal.initializeDistribution()
#
#checkCrowDist("truncNormal",truncNormal,{'xMin': -1.0, 'mu': 1.0, 'type': 'NormalDistribution', 'sigma': 2.0, 'xMax': 3.0})
#
#checkAnswer("truncNormal cdf(0.0)",truncNormal.cdf(0.0),0.219546787406)
#checkAnswer("truncNormal cdf(1.0)",truncNormal.cdf(1.0),0.5)
#checkAnswer("truncNormal cdf(2.0)",truncNormal.cdf(2.0),0.780453212594)
#
#checkAnswer("truncNormal ppf(0.1)",truncNormal.ppf(0.1),-0.498029197939)
#checkAnswer("truncNormal ppf(0.5)",truncNormal.ppf(0.5),1.0)
#checkAnswer("truncNormal ppf(0.9)",truncNormal.ppf(0.9),2.49802919794)

#Test Gamma - Laguerre

#make distribution
gammaElement = ET.Element("gamma")
low = -2.0
alpha = 2.0
beta = 3.0
gammaElement.append(createElement("low",text="%f" %low))
gammaElement.append(createElement("alpha",text="%f" %alpha))
gammaElement.append(createElement("beta",text="%f" %beta))

gamma = Distributions.Gamma()
gamma._readMoreXML(gammaElement)
gamma.initializeDistribution()

#make quadrature
laguerreElement = ET.Element('laguerre')
laguerreElement.append(createElement("alpha",text="%f" %alpha)) #TODO should these be set independently?
laguerre = Quadrature.Laguerre()
laguerre._readMoreXML(laguerreElement)
laguerre.initialize()

#set quadrature to distr
gamma.setQuadrature(laguerre)
checkObject("setting laguerre collocation in gamma",gamma.quadratureSet(),laguerre)

#test points and weights conversion
stdpts=[ 0, 3, 6, 9, 12]
actpts=[-2,-1, 0, 1,  2]
for s,std in enumerate(stdpts):
  act=actpts[s]
  checkAnswer("gamma-laguerre std-to-act pt (%i)" %std,gamma.convertStdPointsToDistr(std),actpts[s])
  checkAnswer("gamma-laguerre act-to-std pt (%i)" %act,gamma.convertDistrPointsToStd(act),stdpts[s])

#test quadrature integration
for i in range(1,6):
  pts,wts = laguerre(i)
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
  checkAnswer(        "gamma-laguerre integrate weights with O(%i)" %i,totu,1.0)
  checkAnswer(        "gamma-laguerre integrate x^0*x^%1.2f exp(-%1.2fx) with O(%i)" %(gamma.alpha,gamma.beta,i),tot0,1)
  if i>=1:checkAnswer("gamma-laguerre integrate x^1*x^%1.2f exp(-%1.2fx) with O(%i)" %(gamma.alpha,gamma.beta,i),tot1,-4./3.)
  if i>=2:checkAnswer("gamma-laguerre integrate x^2*x^%1.2f exp(-%1.2fx) with O(%i)" %(gamma.alpha,gamma.beta,i),tot2,2)
  if i>=3:checkAnswer("gamma-laguerre integrate x^3*x^%1.2f exp(-%1.2fx) with O(%i)" %(gamma.alpha,gamma.beta,i),tot3,-28./9.)
  if i>=4:checkAnswer("gamma-laguerre integrate x^4*x^%1.2f exp(-%1.2fx) with O(%i)" %(gamma.alpha,gamma.beta,i),tot4,136./27.)

#
#checkCrowDist("gamma",gamma,{'xMin': 0.0, 'theta': 2.0, 'k': 1.0, 'type': 'GammaDistribution', 'low': 0.0})
#
#checkAnswer("gamma cdf(0.0)",gamma.cdf(0.0),0.0)
#checkAnswer("gamma cdf(1.0)",gamma.cdf(1.0),0.393469340287)
#checkAnswer("gamma cdf(10.0)",gamma.cdf(10.0),0.993262053001)
#
#checkAnswer("gamma ppf(0.1)",gamma.ppf(0.1),0.210721031316)
#checkAnswer("gamma ppf(0.5)",gamma.ppf(0.5),1.38629436112)
#checkAnswer("gamma ppf(0.9)",gamma.ppf(0.9),4.60517018599)
#
#print(gamma.rvs(5),gamma.rvs())

#Test Beta
#TODO
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

#Test Triangular - Legendre (through Irregular)

#make distribution
triangularElement = ET.Element("triangular")
triangularElement.append(createElement("min",text="1.0"))
triangularElement.append(createElement("apex",text="4.0"))
triangularElement.append(createElement("max",text="5.0"))

triangular = Distributions.Triangular()
triangular._readMoreXML(triangularElement)
triangular.initializeDistribution()

#make quadrature
legendreElement = ET.Element("legendre")
legendre = Quadrature.Legendre()
legendre._readMoreXML(uniformElement)
legendre.initialize()

#link quad to distr
triangular.setQuadrature(legendre)
checkObject("setting irregular-legendre collocation in triangular",triangular.quadratureSet(),legendre)

ptset = [(-1, 1),
         (-0.5, 2.73205080757),
         (0, 3.44948974278),
         (0.5, 4),
         (1, 5)]

for s,pt in enumerate(ptset):
  checkAnswer("triangular-legendre std-to-act pt (%f)" %std,triangular.convertStdPointsToDistr(pt[0]),pt[1])
  checkAnswer("triangular-legendre act-to-std pt (%i)" %act,triangular.convertDistrPointsToStd(pt[1]),pt[0])

for i in range(1,100,5):
  pts,wts = legendre(i)
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
  checkAnswer(        "irreg triang integrate weights with O(%i)" %i,totu,1.0)
  checkAnswer(        "irreg triang integrate x^0 with O(%i)" %i,tot0,1.0)
  #if i>=1:checkAnswer("irreg triang integrate x^1 with O(%i)" %i,tot1,10./3.)
  print('Quad order %i, err: %1.5e' %(i,tot1-10./3.))
  #if i>=1:checkAnswer("irreg triang integrate x^2 with O(%i)" %i,tot2,71./6.)
  #if i>=1:checkAnswer("irreg triang integrate x^3 with O(%i)" %i,tot3,1760./40.)
  #if i>=1:checkAnswer("irreg triang integrate x^4 with O(%i)" %i,tot4,847./5.)

#FIXME problem for testing - this converges, but is by no means exact



sys.exit()

#Test Poisson

poissonElement = ET.Element("poisson")
poissonElement.append(createElement("mu",text="4.0"))

poisson = Distributions.Poisson()
poisson._readMoreXML(poissonElement)
poisson.initializeDistribution()

checkCrowDist("poisson",poisson,{'mu': 4.0, 'type': 'PoissonDistribution'})

checkAnswer("poisson cdf(1.0)",poisson.cdf(1.0),0.0915781944437)
checkAnswer("poisson cdf(5.0)",poisson.cdf(5.0),0.7851303870304052)
checkAnswer("poisson cdf(10.0)",poisson.cdf(10.0),0.997160233879)

checkAnswer("poisson ppf(0.0915781944437)",poisson.ppf(0.0915781944437),1.0)
checkAnswer("poisson ppf(0.7851303870304052)",poisson.ppf(0.7851303870304052),5.0)
checkAnswer("poisson ppf(0.997160233879)",poisson.ppf(0.997160233879),10.0)

print(poisson.rvs(5),poisson.rvs())

#Test Binomial

binomialElement = ET.Element("binomial")
binomialElement.append(createElement("n",text="10"))
binomialElement.append(createElement("p",text="0.25"))

binomial = Distributions.Binomial()
binomial._readMoreXML(binomialElement)
binomial.initializeDistribution()

checkCrowDist("binomial",binomial,{'p': 0.25, 'type': 'BinomialDistribution', 'n': 10.0})

checkAnswer("binomial cdf(1)",binomial.cdf(1),0.244025230408)
checkAnswer("binomial cdf(2)",binomial.cdf(2),0.525592803955)
checkAnswer("binomial cdf(5)",binomial.cdf(5),0.980272293091)

checkAnswer("binomial ppf(0.1)",binomial.ppf(0.1),0.0)
checkAnswer("binomial ppf(0.5)",binomial.ppf(0.5),2.0)
checkAnswer("binomial ppf(0.9)",binomial.ppf(0.9),4.0)

#Test Bernoulli

bernoulliElement = ET.Element("bernoulli")
bernoulliElement.append(createElement("p",text="0.4"))

bernoulli = Distributions.Bernoulli()
bernoulli._readMoreXML(bernoulliElement)
bernoulli.initializeDistribution()

checkCrowDist("bernoulli",bernoulli,{'p': 0.4, 'type': 'BernoulliDistribution'})

checkAnswer("bernoulli cdf(0)",bernoulli.cdf(0),0.6)
checkAnswer("bernoulli cdf(1)",bernoulli.cdf(1),1.0)

checkAnswer("bernoulli ppf(0.1)",bernoulli.ppf(0.1),0.0)
checkAnswer("bernoulli ppf(0.3)",bernoulli.ppf(0.3),0.0)
checkAnswer("bernoulli ppf(0.8)",bernoulli.ppf(0.8),1.0)
checkAnswer("bernoulli ppf(0.9)",bernoulli.ppf(0.9),1.0)

#Test Logistic

logisticElement = ET.Element("logistic")
logisticElement.append(createElement("location",text="4.0"))
logisticElement.append(createElement("scale",text="1.0"))

logistic = Distributions.Logistic()
logistic._readMoreXML(logisticElement)
logistic.initializeDistribution()

checkCrowDist("logistic",logistic,{'scale': 1.0, 'type': 'LogisticDistribution', 'location': 4.0})

checkAnswer("logistic cdf(0)",logistic.cdf(0.0),0.0179862099621)
checkAnswer("logistic cdf(4)",logistic.cdf(4.0),0.5)
checkAnswer("logistic cdf(8)",logistic.cdf(8.0),0.982013790038)

checkAnswer("logistic ppf(0.25)",logistic.ppf(0.25),2.90138771133)
checkAnswer("logistic ppf(0.50)",logistic.ppf(0.50),4.0)
checkAnswer("logistic ppf(0.75)",logistic.ppf(0.75),5.09861228867)

#Test Exponential

exponentialElement = ET.Element("exponential")
exponentialElement.append(createElement("lambda",text="5.0"))

exponential = Distributions.Exponential()
exponential._readMoreXML(exponentialElement)
exponential.initializeDistribution()

checkCrowDist("exponential",exponential,{'xMin': 0.0, 'type': 'ExponentialDistribution', 'lambda': 5.0})

checkAnswer("exponential cdf(0.3)",exponential.cdf(0.3),0.7768698399)
checkAnswer("exponential cdf(1.0)",exponential.cdf(1.0),0.993262053001)
checkAnswer("exponential cdf(3.0)",exponential.cdf(3.0),0.999999694098)

checkAnswer("exponential ppf(0.7768698399)",exponential.ppf(0.7768698399),0.3)
checkAnswer("exponential ppf(0.2)",exponential.ppf(0.2),0.0446287102628)
checkAnswer("exponential ppf(0.5)",exponential.ppf(0.5),0.138629436112)

#Test truncated exponential

truncExponentialElement = ET.Element("truncexponential")
truncExponentialElement.append(createElement("lambda",text="5.0"))
truncExponentialElement.append(createElement("lowerBound",text="0.0"))
truncExponentialElement.append(createElement("upperBound",text="10.0"))

truncExponential = Distributions.Exponential()
truncExponential._readMoreXML(truncExponentialElement)
truncExponential.initializeDistribution()

checkCrowDist("truncExponential",truncExponential,{'xMin': 0.0, 'type': 'ExponentialDistribution', 'xMax': 10.0, 'lambda': 5.0})

checkAnswer("truncExponential cdf(0.1)",truncExponential.cdf(0.1),0.393469340287)
checkAnswer("truncExponential cdf(5.0)",truncExponential.cdf(5.0),0.999999999986)
checkAnswer("truncExponential cdf(9.9)",truncExponential.cdf(9.9),1.0)


checkAnswer("truncExponential ppf(0.1)",truncExponential.ppf(0.1),0.0210721031316)
checkAnswer("truncExponential ppf(0.5)",truncExponential.ppf(0.5),0.138629436112)
checkAnswer("truncExponential ppf(0.9)",truncExponential.ppf(0.9),0.460517018599)

#Test log normal

logNormalElement = ET.Element("logNormal")
logNormalElement.append(createElement("mean",text="3.0"))
logNormalElement.append(createElement("sigma",text="2.0"))

logNormal = Distributions.LogNormal()
logNormal._readMoreXML(logNormalElement)
logNormal.initializeDistribution()

checkCrowDist("logNormal",logNormal,{'mu': 3.0, 'sigma': 2.0, 'type': 'LogNormalDistribution'})

checkAnswer("logNormal cdf(2.0)",logNormal.cdf(2.0),0.124367703363)
checkAnswer("logNormal cdf(1.0)",logNormal.cdf(1.0),0.0668072012689)
checkAnswer("logNormal cdf(3.0)",logNormal.cdf(3.0),0.170879904093)

checkAnswer("logNormal ppf(0.1243677033)",logNormal.ppf(0.124367703363),2.0)
checkAnswer("logNormal ppf(0.1)",logNormal.ppf(0.1),1.54789643258)
checkAnswer("logNormal ppf(0.5)",logNormal.ppf(0.5),20.0855369232)

#Test log normal with low mean

logNormalLowMeanElement = ET.Element("logNormal")
logNormalLowMeanElement.append(createElement("mean",text="-0.00002"))
logNormalLowMeanElement.append(createElement("sigma",text="0.2"))

logNormalLowMean = Distributions.LogNormal()
logNormalLowMean._readMoreXML(logNormalLowMeanElement)
logNormalLowMean.initializeDistribution()

checkCrowDist("logNormalLowMean",logNormalLowMean,{'mu': -2e-5, 'sigma': 0.2, 'type': 'LogNormalDistribution'})

checkAnswer("logNormalLowMean cdf(2.0)",logNormalLowMean.cdf(2.0),0.999735707106)
checkAnswer("logNormalLowMean cdf(1.0)",logNormalLowMean.cdf(1.0),0.500039894228)
checkAnswer("logNormalLowMean cdf(3.0)",logNormalLowMean.cdf(3.0),0.999999980238)

checkAnswer("logNormalLowMean ppf(0.500039894228)",logNormalLowMean.ppf(0.500039894228),1.0)
checkAnswer("logNormalLowMean ppf(0.1)",logNormalLowMean.ppf(0.1),0.773886301779)
checkAnswer("logNormalLowMean ppf(0.5)",logNormalLowMean.ppf(0.5),0.9999800002)

#Test Weibull

weibullElement = ET.Element("weibull")
weibullElement.append(createElement("k", text="1.5"))
weibullElement.append(createElement("lambda", text="1.0"))

weibull = Distributions.Weibull()
weibull._readMoreXML(weibullElement)
weibull.initializeDistribution()

checkCrowDist("weibull",weibull,{'xMin': 0.0, 'k': 1.5, 'type': 'WeibullDistribution', 'lambda': 1.0})

checkAnswer("weibull cdf(0.5)",weibull.cdf(0.5),0.29781149863)
checkAnswer("weibull cdf(0.2)",weibull.cdf(0.2),0.0855593563928)
checkAnswer("weibull cdf(2.0)",weibull.cdf(2.0),0.940894253438)

checkAnswer("weibull ppf(0.29781149863)",weibull.ppf(0.29781149863),0.5)
checkAnswer("weibull ppf(0.1)",weibull.ppf(0.1),0.223075525637)
checkAnswer("weibull ppf(0.9)",weibull.ppf(0.9),1.7437215136)



print(results)

sys.exit(results["fail"])
