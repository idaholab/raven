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
import OrthoPolynomials



## Flag for by-section print statements ##
#debug = False
debug = True

## XML tools ##
def createElement(tag,attrib={},text={}):
  element = ET.Element(tag,attrib)
  element.text = text
  return element

## Record for tests ##
results = {"pass":0,"fail":0}

## Compare floats ##
def floatNotEqual(a,b):
  if b<1e1 and a<1e1:
    denom=1.0
  elif b==0:
    if a==0:
      return False
    else:
      denom=a
  else:
    denom=b
  return abs(a - b)/denom > 1e-10

## Compare expected/obtained results ##
def checkAnswer(comment,value,expected):
  if floatNotEqual(value, expected):
    print(comment,value,"!=",expected)
    results["fail"] += 1
  else:
    results["pass"] += 1

## Compare expected/obtained objects ##
def checkObject(comment,value,expected):
  if value!=expected:
    print(comment,value,"!=",expected)
    results['fail']+=1
  else: results['pass']+=1

## Polynomial for moment testing ##
def testPoly(n,y):
  return y**n

#
# OUTLINE
#  - Test quadrature integration of monomials
#  - Test orthogonality of polynomials using quadratures
#

## shared shape parameters ##
laguerre_alpha = 2.0
jacobi_alpha = 5.0
jacobi_beta  = 2.0

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

jacobiElement = ET.Element('jacobi')
jacobiElement.append(createElement("alpha",text="%f" %jacobi_alpha))
jacobiElement.append(createElement("beta",text="%f" %jacobi_beta))
jacobi = Quadrature.Jacobi()
jacobi._readMoreXML(jacobiElement)
jacobi.initialize()
quads['Jacobi']=jacobi

## END make quadratures ##

## generate polynomials ##
polys={}

plegendreElement = ET.Element("plegendre")
plegendre = OrthoPolynomials.Legendre()
plegendre._readMoreXML(plegendreElement)
plegendre.initialize()
polys['Legendre']=plegendre

phermiteElement = ET.Element('phermite')
phermite = OrthoPolynomials.Hermite()
phermite._readMoreXML(phermiteElement)
phermite.initialize()
polys['Hermite']=phermite

plaguerreElement = ET.Element('plaguerre')
plaguerreElement.append(createElement("alpha",text="%f" %laguerre_alpha))
plaguerre = OrthoPolynomials.Laguerre()
plaguerre._readMoreXML(plaguerreElement)
plaguerre.initialize()
polys['Laguerre']=plaguerre

pjacobiElement = ET.Element('pjacobi')
pjacobiElement.append(createElement("alpha",text="%f" %jacobi_alpha))
pjacobiElement.append(createElement("beta",text="%f" %jacobi_beta))
pjacobi = OrthoPolynomials.Jacobi()
pjacobi._readMoreXML(pjacobiElement)
pjacobi.initialize()
polys['Jacobi']=pjacobi
## END make polynomials ##

###########################################
#            Tests for Uniform            #
###########################################

#make distribution
if debug: print('Testing Uniform...')
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
uniform.setPolynomials(polys['Legendre'],1)
checkObject("setting Legendre as poly in uniform",uniform.polynomialSet(),polys['Legendre'])

for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quadrature to distr
  uniform.setQuadrature(quad)
  checkObject("setting %s collocation in uniform" %quadname,uniform.quadratureSet(),quad)

  #test points and weights conversion
  for p,pt in enumerate(ptset):
    checkAnswer("uniform-%s std-to-act pt (%i)" %(quadname,pt[0]),uniform.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("uniform-%s act-to-std pt (%i)" %(quadname,pt[1]),uniform.convertDistrPointsToStd(pt[1]),pt[0])

  #test quadrature integration
  for i in range(1,6):
    opts,wts = quad(i)
    pts = uniform.convertStdPointsToDistr(opts)
    totu=0
    if i>=2:
      tot2=0
      orth1_1=0
      orth1_2=0
    if i>=3:
      tot5=0
      orth2_2=0
      orth1_3=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*uniform.probabilityNorm()
      if i>=2:
        tot2+=testPoly(2,pt)*wts[p]*uniform.probabilityNorm()
        orth1_1+=polys['Legendre'](1,opts[p])*polys['Legendre'](1,opts[p])*wts[p]*uniform.probabilityNorm()
        orth1_2+=polys['Legendre'](1,opts[p])*polys['Legendre'](2,opts[p])*wts[p]*uniform.probabilityNorm()
      if i>=3:
        orth2_2+=polys['Legendre'](2,opts[p])*polys['Legendre'](2,opts[p])*wts[p]*uniform.probabilityNorm()
        orth1_3+=polys['Legendre'](1,opts[p])*polys['Legendre'](3,opts[p])*wts[p]*uniform.probabilityNorm()
        tot5+=testPoly(5,pt)*wts[p]*uniform.probabilityNorm()
    checkAnswer("%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    if i>=2:
      checkAnswer("uniform-%s integrate y^2 P(y)dy with O(%i)" %(quadname,i),tot2,31./3.)
      checkAnswer("uniform-%s integrate Legendre polys (1,1) with O(%i)" %(quadname,i),orth1_1,1)
      checkAnswer("uniform-%s integrate Legendre polys (1,2) with O(%i)" %(quadname,i),orth1_2,0)
    if i>=3:
      checkAnswer("uniform-%s integrate y^5 P(y)dy with O(%i)" %(quadname,i),tot5,3906./6.)
      checkAnswer("uniform-%s integrate Legendre polys (2,2) with O(%i)" %(quadname,i),orth1_1,1)
      checkAnswer("uniform-%s integrate Legendre polys (1,3) with O(%i)" %(quadname,i),orth1_2,0)

del uniform


##########################################
#            Tests for Normal            #
##########################################

#make distrubtion
if debug: print('Testing Normal...')
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

doQuads = ['Hermite','CDF']
solns={}
solns['Hermite']=[1,5,13,73]
solns['CDF']    =[1,4.95557166321,12.8667149896,69.1169629063]

orthslns={}
orthslns['Hermite']={1:0,
                     2:0,
                     3:0,
                     (1,1):1,
                     (1,2):0,
                     (1,3):0,
                     (2,2):1,
                     (3,3):1}
orthslns['CDF']={1:0,
                 2:0,
                 3:0,
                 (1,1):0.997455547486,
                 (1,2):0,
                 (1,3):-0.0238010251601,
                 (2,2):0.969577590244,
                 (3,3):0.850051194319} #see how badly this is converging?
## perform tests ##
# make sure CC fails on untruncated normal
try:
  normal.setQuadrature(quads['ClenshawCurtis'])
  prevented=False
except IOError:
  prevented=True
checkAnswer('Prevent full Normal from using ClenshawCurtis',prevented,True)

normal.setPolynomials(polys['Hermite'],1)
checkObject("setting Hermite as poly in normal",normal.polynomialSet(),polys['Hermite'])

for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quadrature to distr
  normal.setQuadrature(quad)
  checkObject("setting %s collocation in normal",normal.quadratureSet(),quad)

  #test points and weights conversion
  if quadname not in ['CDF','ClenshawCurtis']:
    for p,pt in enumerate(ptset):
      checkAnswer("normal-%s std-to-act pt (%i)" %(quadname,pt[0]),normal.convertStdPointsToDistr(pt[0]),pt[1])
      checkAnswer("normal-%s act-to-std pt (%i)" %(quadname,pt[1]),normal.convertDistrPointsToStd(pt[1]),pt[0])

  #test quadrature integration
  for i in [14]:
    pts,wts = quad(i)
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
    checkAnswer("normal-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer("normal-%s integrate x^0*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot0,1)
    checkAnswer("normal-%s integrate x^1*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot1,solns[quadname][0])
    checkAnswer("normal-%s integrate x^2*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot2,solns[quadname][1])
    checkAnswer("normal-%s integrate x^3*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot4,solns[quadname][2])
    checkAnswer("normal-%s integrate x^4*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot6,solns[quadname][3])
    
  for i in [30]:
    pts,wts = quad(i)
    orth1=0
    orth2=0
    orth3=0
    orth1_1=0
    orth1_2=0
    orth2_2=0
    orth1_3=0
    orth3_3=0
    for p,pt in enumerate(pts):
      orth1+=polys['Hermite'](1,pts[p])*wts[p]
      orth2+=polys['Hermite'](2,pts[p])*wts[p]
      orth3+=polys['Hermite'](3,pts[p])*wts[p]
      orth1_1+=polys['Hermite'](1,pts[p])*polys['Hermite'](1,pts[p])*wts[p]
      orth1_2+=polys['Hermite'](1,pts[p])*polys['Hermite'](2,pts[p])*wts[p]
      orth2_2+=polys['Hermite'](2,pts[p])*polys['Hermite'](2,pts[p])*wts[p]
      orth1_3+=polys['Hermite'](1,pts[p])*polys['Hermite'](3,pts[p])*wts[p]
      orth3_3+=polys['Hermite'](3,pts[p])*polys['Hermite'](3,pts[p])*wts[p]
    orth1*=normal.probabilityNorm()
    orth2*=normal.probabilityNorm()
    orth3*=normal.probabilityNorm()
    orth1_1*=normal.probabilityNorm()
    orth1_2*=normal.probabilityNorm()
    orth2_2*=normal.probabilityNorm()
    orth1_3*=normal.probabilityNorm()
    orth3_3*=normal.probabilityNorm()
    checkAnswer("normal-%s integrate Hermite poly(1) with O(%i)" %(quadname,i),orth1,orthslns[quadname][1])
    checkAnswer("normal-%s integrate Hermite poly(2) with O(%i)" %(quadname,i),orth1,orthslns[quadname][2])
    checkAnswer("normal-%s integrate Hermite poly(3) with O(%i)" %(quadname,i),orth1,orthslns[quadname][3])
    checkAnswer("normal-%s integrate Hermite polys (1,1) with O(%i)" %(quadname,i),orth1_1,orthslns[quadname][(1,1)])
    checkAnswer("normal-%s integrate Hermite polys (1,2) with O(%i)" %(quadname,i),orth1_2,orthslns[quadname][(1,2)])
    checkAnswer("normal-%s integrate Hermite polys (2,2) with O(%i)" %(quadname,i),orth2_2,orthslns[quadname][(2,2)])
    checkAnswer("normal-%s integrate Hermite polys (1,3) with O(%i)" %(quadname,i),orth1_3,orthslns[quadname][(1,3)])
    checkAnswer("normal-%s integrate Hermite polys (3,3) with O(%i)" %(quadname,i),orth3_3,orthslns[quadname][(3,3)])
    
  #test orthogonal polynomials TODO
del normal

##################################################
#            Tests for Truncated Norm            #
##################################################
if debug: print('Testing Truncated Normal...')
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

solns={}
solns['Hermite']=       [1,5            ,13           ,73]
solns['CDF']=           [1,2.16450037909,4.49350113726,10.6190083398]
solns['ClenshawCurtis']=[1,2.16450037909,4.49350113727,10.6190083398]

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quadrature to distr
  truncNormal.setQuadrature(quad)
  checkObject("setting %s collocation in truncNormal",truncNormal.quadratureSet(),quad)

  #test points and weights conversion
  if quadname not in ['CDF','ClenshawCurtis']:
    for p,pt in enumerate(ptset):
      checkAnswer("truncNormal-%s std-to-act pt (%i)" %(quadname,pt[0]),truncNormal.convertStdPointsToDistr(pt[0]),pt[1])
      checkAnswer("truncNormal-%s act-to-std pt (%i)" %(quadname,pt[1]),truncNormal.convertDistrPointsToStd(pt[1]),pt[0])

  #test quadrature integration of monomials
  for i in [14]:
    pts,wts = quad(i)
    pts = truncNormal.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot4=0
    tot6=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*truncNormal.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*truncNormal.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*truncNormal.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*truncNormal.probabilityNorm()
      tot4+=testPoly(3,pt)*wts[p]*truncNormal.probabilityNorm()
      tot6+=testPoly(4,pt)*wts[p]*truncNormal.probabilityNorm()
    checkAnswer(        "truncNormal-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "truncNormal-%s integrate x^0*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot0,1)
    if i>=1:checkAnswer("truncNormal-%s integrate x^1*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot1,solns[quadname][0])
    if i>=2:checkAnswer("truncNormal-%s integrate x^2*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot2,solns[quadname][1])
    if i>=3:checkAnswer("truncNormal-%s integrate x^3*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot4,solns[quadname][2])
    if i>=4:checkAnswer("truncNormal-%s integrate x^4*exp(-(x-%i)^2/2*%i^2) with O(%i)" %(quadname,mean,stdv,i),tot6,solns[quadname][3])

del truncNormal

#########################################
#            Tests for Gamma            #
#########################################

if debug: print('Testing Gamma...')
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

doQuads=['Laguerre','CDF']
solns={}
solns['Laguerre']=[-4./3.,2.,-28./9.,136./27.]
solns['CDF']=[-1.33541433856,1.99568160194,-3.11623304332,5.02760429874]

orthslns={}
orthslns['CDF']={1:0.000518652611711,
                 2:0.000518652611711,
                 3:0.000518652611711,
                 (1,1):0.993141128539,
                 (1,2):0.0327146322173,
                 (1,3):-0.070476302904,
                 (2,2):0.868802995096,
                 (3,3):0.535147488966}
orthslns['Laguerre']={1:0,
                      2:0,
                      3:0,
                      (1,1):1,
                      (1,2):0,
                      (1,3):0,
                      (2,2):1,
                      (3,3):1}
# make sure CC fails
try:
  gamma.setQuadrature(quads['ClenshawCurtis'])
  prevented=False
except IOError:
  prevented=True
checkAnswer('Prevent full Gamma from using ClenshawCurtis',prevented,True)

gamma.setPolynomials(polys['Laguerre'],1)
checkObject("setting Laguerre as poly in Gamma",gamma.polynomialSet(),polys['Laguerre'])

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads:continue
  if debug: print('    ...testing %s...' %quadname)
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
    checkAnswer("Gamma-%s integrate x^1*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot1,solns[quadname][0])
    checkAnswer("Gamma-%s integrate x^2*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot2,solns[quadname][1])
    checkAnswer("Gamma-%s integrate x^3*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot3,solns[quadname][2])
    checkAnswer("Gamma-%s integrate x^4*x^%1.2f exp(-%1.2fx) with O(%i)" %(quadname,gamma.alpha,gamma.beta,i),tot4,solns[quadname][3])

  for i in [30]:
    pts,wts = quad(i)
    orth1=0
    orth2=0
    orth3=0
    orth1_1=0
    orth1_2=0
    orth2_2=0
    orth1_3=0
    orth3_3=0
    for p,pt in enumerate(pts):
      orth1+=polys['Laguerre'](1,pts[p])*wts[p]
      orth2+=polys['Laguerre'](2,pts[p])*wts[p]
      orth3+=polys['Laguerre'](3,pts[p])*wts[p]
      orth1_1+=polys['Laguerre'](1,pts[p])*polys['Laguerre'](1,pts[p])*wts[p]
      orth1_2+=polys['Laguerre'](1,pts[p])*polys['Laguerre'](2,pts[p])*wts[p]
      orth2_2+=polys['Laguerre'](2,pts[p])*polys['Laguerre'](2,pts[p])*wts[p]
      orth1_3+=polys['Laguerre'](1,pts[p])*polys['Laguerre'](3,pts[p])*wts[p]
      orth3_3+=polys['Laguerre'](3,pts[p])*polys['Laguerre'](3,pts[p])*wts[p]
    orth1*=gamma.probabilityNorm()
    orth2*=gamma.probabilityNorm()
    orth3*=gamma.probabilityNorm()
    orth1_1*=gamma.probabilityNorm()
    orth1_2*=gamma.probabilityNorm()
    orth2_2*=gamma.probabilityNorm()
    orth1_3*=gamma.probabilityNorm()
    orth3_3*=gamma.probabilityNorm()
    #checkAnswer("gamma-%s integrate Laguerre poly(1) with O(%i)" %(quadname,i),orth1,orthslns[quadname][1])
    #checkAnswer("gamma-%s integrate Laguerre poly(2) with O(%i)" %(quadname,i),orth1,orthslns[quadname][2])
    #checkAnswer("gamma-%s integrate Laguerre poly(3) with O(%i)" %(quadname,i),orth1,orthslns[quadname][3])
    #checkAnswer("gamma-%s integrate Laguerre polys (1,1) with O(%i)" %(quadname,i),orth1_1,orthslns[quadname][(1,1)])
    #checkAnswer("gamma-%s integrate Laguerre polys (1,2) with O(%i)" %(quadname,i),orth1_2,orthslns[quadname][(1,2)])
    #checkAnswer("gamma-%s integrate Laguerre polys (2,2) with O(%i)" %(quadname,i),orth2_2,orthslns[quadname][(2,2)])
    #checkAnswer("gamma-%s integrate Laguerre polys (1,3) with O(%i)" %(quadname,i),orth1_3,orthslns[quadname][(1,3)])
    checkAnswer("gamma-%s integrate Laguerre polys (3,3) with O(%i)" %(quadname,i),orth3_3,orthslns[quadname][(3,3)])
  #test orthogonal polynomials TODO
del gamma

#pol = polys['Laguerre']
#print('DEBUG alpha',laguerre_alpha)
#print('DEBUG polynorm(2)^2',pol.norm(2)**2)
#print('DEBUG polynorm(5)^2',pol.norm(5)**2)
#print('DEBUG polynorm(8)^2',pol.norm(8)**2)
#pol.setMeasures(quads['CDF'])
#print('DEBUG poly(1)\n',pol[1])
#print('DEBUG ppf',pol._getDistr().ppf(0.211325))
#print('DEBUG ppf',pol._getDistr().ppf(0.788675))

##############################################
#            Test Jacobi for Beta            #
##############################################
if debug: print('Testing Beta...')
betaElement = ET.Element("beta")
betaElement.append(createElement("low",text="-1.0"))
betaElement.append(createElement("hi",text="5.0"))
betaElement.append(createElement("alpha",text="%f" %jacobi_alpha))
betaElement.append(createElement("beta" ,text="%f" %jacobi_beta))

beta = Distributions.Beta()
beta._readMoreXML(betaElement)
beta.initializeDistribution()

ptset = [(-1  ,-1  ),
         (-0.5, 0.5),
         ( 0  , 2  ),
         ( 0.5, 3.5),
         ( 1  , 5  )]

doQuads = ['Jacobi']#,'CDF','ClenshawCurtis']

solns={}
solns['Jacobi']=[23./7., 82./7., 44., 1201./7.]
solns['CDF']=[3.28627676273,11.713670989,43.9958588608,171.545083074]
solns['ClenshawCurtis']=[3.28571431915,11.7142873238,44.0000445579,171.572749292]

orthslns={}
orthslns['CDF']={1:0,
                 2:0,
                 3:0,
                 (1,1):1,
                 (1,2):0,
                 (1,3):0,
                 (2,2):1,
                 (3,3):1}
orthslns['Jacobi']={1:0,
                    2:0,
                    3:0,
                    (1,1):1,
                    (1,2):0,
                    (1,3):0,
                    (2,2):1,
                    (3,3):1}

beta.setPolynomials(polys['Jacobi'],1)
checkObject("setting Jacobi as poly in Beta",beta.polynomialSet(),polys['Jacobi'])

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quad to distr
  beta.setQuadrature(quad)
  checkObject("setting %s collocation in beta" %quadname,beta.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("beta-%s std-to-act pt (%f)" %(quadname,pt[0]),beta.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("beta-%s act-to-std pt (%i)" %(quadname,pt[1]),beta.convertDistrPointsToStd(pt[1]),pt[0])

  for i in [14]:
    pts,wts = quad(i)
    pts = beta.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*beta.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*beta.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*beta.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*beta.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*beta.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*beta.probabilityNorm()
    checkAnswer("Beta-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer("Beta-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    checkAnswer("Beta-%s integrate x^1 with O(%i)" %(quadname,i),tot1,solns[quadname][0])
    checkAnswer("Beta-%s integrate x^2 with O(%i)" %(quadname,i),tot2,solns[quadname][1])
    checkAnswer("Beta-%s integrate x^3 with O(%i)" %(quadname,i),tot3,solns[quadname][2])
    checkAnswer("Beta-%s integrate x^4 with O(%i)" %(quadname,i),tot4,solns[quadname][3])

  for i in [4]:
    pts,wts = quad(i)
    orth1=0
    orth2=0
    orth3=0
    orth1_1=0
    orth1_2=0
    orth2_2=0
    orth1_3=0
    orth3_3=0
    for p,pt in enumerate(pts):
      orth1+=polys['Jacobi'](1,pts[p])*wts[p]
      orth2+=polys['Jacobi'](2,pts[p])*wts[p]
      orth3+=polys['Jacobi'](3,pts[p])*wts[p]
      orth1_1+=polys['Jacobi'](1,pts[p])*polys['Jacobi'](1,pts[p])*wts[p]
      orth1_2+=polys['Jacobi'](1,pts[p])*polys['Jacobi'](2,pts[p])*wts[p]
      orth2_2+=polys['Jacobi'](2,pts[p])*polys['Jacobi'](2,pts[p])*wts[p]
      orth1_3+=polys['Jacobi'](1,pts[p])*polys['Jacobi'](3,pts[p])*wts[p]
      orth3_3+=polys['Jacobi'](3,pts[p])*polys['Jacobi'](3,pts[p])*wts[p]
    orth1*=beta.probabilityNorm()
    orth2*=beta.probabilityNorm()
    orth3*=beta.probabilityNorm()
    orth1_1*=beta.probabilityNorm()
    orth1_2*=beta.probabilityNorm()
    orth2_2*=beta.probabilityNorm()
    orth1_3*=beta.probabilityNorm()
    orth3_3*=beta.probabilityNorm()
    checkAnswer("beta-%s integrate Jacobi poly(1) with O(%i)" %(quadname,i),orth1,orthslns[quadname][1])
    checkAnswer("beta-%s integrate Jacobi poly(2) with O(%i)" %(quadname,i),orth1,orthslns[quadname][2])
    checkAnswer("beta-%s integrate Jacobi poly(3) with O(%i)" %(quadname,i),orth1,orthslns[quadname][3])
    checkAnswer("beta-%s integrate Jacobi polys (1,1) with O(%i)" %(quadname,i),orth1_1,orthslns[quadname][(1,1)])
    checkAnswer("beta-%s integrate Jacobi polys (1,2) with O(%i)" %(quadname,i),orth1_2,orthslns[quadname][(1,2)])
    checkAnswer("beta-%s integrate Jacobi polys (2,2) with O(%i)" %(quadname,i),orth2_2,orthslns[quadname][(2,2)])
    checkAnswer("beta-%s integrate Jacobi polys (1,3) with O(%i)" %(quadname,i),orth1_3,orthslns[quadname][(1,3)])
    checkAnswer("beta-%s integrate Jacobi polys (3,3) with O(%i)" %(quadname,i),orth3_3,orthslns[quadname][(3,3)])
#print('DEBUG org point:',2./3.)
#print('DEBUG poly:     \n',polys['Jacobi']._poly(2,1,4))
#print('DEBUG norm:     ',polys['Jacobi'].norm(2))
#print('DEBUG polynorm:\n',polys['Jacobi']._poly(2,1,4)*polys['Jacobi'].norm(1))
#print('DEBUG poly1:    ',polys['Jacobi']._poly(2,1,4)(2./3.)*polys['Jacobi'].norm(1))
#print('DEBUG poly2:    ',polys['Jacobi']._evPoly(2,1,4,2./3.)*polys['Jacobi'].norm(1))
#print('')
#print('DEBUG jacobi(2)(0)   ',polys['Jacobi'](2,0))
#print('DEBUG jacobi(2)(2/3) ',polys['Jacobi'](2,2./3.))
#print('DEBUG jacobi(0)^2    ',polys['Jacobi'](2,0)**2)
#print('DEBUG jacobi(2/3)^2  ',polys['Jacobi'](2,2./3.)**2)
#print('')
#print('DEBUG beta norm:',beta.probabilityNorm())
del beta

##############################################
#            Tests for Triangular            #
##############################################

if debug: print('Testing Triangular...')
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

solns={}
solns['CDF']=[3.33335785995,11.8326958121,43.9941325182,169.360731767]
solns['ClenshawCurtis']=[10./3.,71./6.,1760./40.,847./5.]

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quad to distr
  triangular.setQuadrature(quad)
  checkObject("setting %s collocation in triangular" %quadname,triangular.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("triangular-%s std-to-act pt (%f)" %(quadname,pt[0]),triangular.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("triangular-%s act-to-std pt (%i)" %(quadname,pt[1]),triangular.convertDistrPointsToStd(pt[1]),pt[0])

  for i in [14]:
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
    checkAnswer("Triangular-%s integrate x^1 with O(%i)" %(quadname,i),tot1,solns[quadname][0])#10./3.) #CC needs p(12)==O(2**12+1)
    checkAnswer("Triangular-%s integrate x^2 with O(%i)" %(quadname,i),tot2,solns[quadname][1])#71./6.) #CC needs p(13)
    checkAnswer("Triangular-%s integrate x^3 with O(%i)" %(quadname,i),tot3,solns[quadname][2])#1760./40.) #CC needs p(13)
    checkAnswer("Triangular-%s integrate x^4 with O(%i)" %(quadname,i),tot4,solns[quadname][3])#847./5.) #CC needs p(14)
del triangular

###########################################
#            Tests for Poisson            #
###########################################
if debug: print('Testing Poisson...')
#make distribution
poissonElement = ET.Element("poisson")
poissonElement.append(createElement("mu",text="4.0"))
poisson = Distributions.Poisson()
poisson._readMoreXML(poissonElement)
poisson.initializeDistribution()

ptset = [(-1  , 0, -0.963368722223),
         (-0.5, 2, -0.523793388893),
         ( 0  , 4,  0.25767387036 ),
         ( 0.5, 5,  0.570260774061),
         ( 0.9, 8,  0.957273131024)]

#doQuads = []
doQuads = ['CDF']

# make sure CC fails
try:
  poisson.setQuadrature(quads['ClenshawCurtis'])
  prevented=False
except IOError:
  prevented=True
checkAnswer('Prevent full Poisson from using ClenshawCurtis',prevented,True)


## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quad to distr
  poisson.setQuadrature(quad)
  checkObject("setting %s collocation in poisson" %quadname,poisson.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("poisson-%s std-to-act pt (%f)" %(quadname,pt[0]),poisson.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("poisson-%s act-to-std pt (%i)" %(quadname,pt[1]),poisson.convertDistrPointsToStd(pt[1]),pt[2])

  for i in [20]:
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
    checkAnswer("Poisson-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer("Poisson-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    checkAnswer("Poisson-%s integrate x^1 with O(%i)" %(quadname,i),tot1,3.52377197144) #analytic 4
    checkAnswer("Poisson-%s integrate x^2 with O(%i)" %(quadname,i),tot2,18.0141760836) #20
    checkAnswer("Poisson-%s integrate x^3 with O(%i)" %(quadname,i),tot3,108.342947358) #116
    checkAnswer("Poisson-%s integrate x^4 with O(%i)" %(quadname,i),tot4,726.114429637) #756
del poisson

############################################
#            Tests for Binomial            #
############################################
if debug: print('Testing Binomial...')
#make distribution
binomialElement = ET.Element("binomial")
binomialElement.append(createElement("n",text="10"))
binomialElement.append(createElement("p",text="0.25"))
binomial = Distributions.Binomial()
binomial._readMoreXML(binomialElement)
binomial.initializeDistribution()

ptset = [(-1  ,  0, -0.887372970581),
         (-0.5,  1, -0.511949539185),
         ( 0  ,  2, 0.0511856079102),
         ( 0.5,  3, 0.551750183105),
         ( 1  , 10, 1)]

doQuads = ['CDF','ClenshawCurtis']
solns={}
solns['CDF']=[1.98242930517,6.80113465281,26.1131185809,108.236455079]
solns['ClenshawCurtis']=[2.05622115673,7.16871201528,28.6411699694,125.008305221]

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quad to distr
  binomial.setQuadrature(quad)
  checkObject("setting %s collocation in binomial" %quadname,binomial.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("binomial-%s std-to-act pt (%f)" %(quadname,pt[0]),binomial.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("binomial-%s act-to-std pt (%i)" %(quadname,pt[1]),binomial.convertDistrPointsToStd(pt[1]),pt[2])

  for i in [14]:
    pts,wts = quad(i)
    pts = binomial.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*binomial.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*binomial.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*binomial.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*binomial.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*binomial.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*binomial.probabilityNorm()
    checkAnswer(        "Binomial-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer(        "Binomial-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    checkAnswer("Binomial-%s integrate x^1 with O(%i)" %(quadname,i),tot1,solns[quadname][0])
    checkAnswer("Binomial-%s integrate x^2 with O(%i)" %(quadname,i),tot2,solns[quadname][1])
    checkAnswer("Binomial-%s integrate x^3 with O(%i)" %(quadname,i),tot3,solns[quadname][2])
    checkAnswer("Binomial-%s integrate x^4 with O(%i)" %(quadname,i),tot4,solns[quadname][3])

del binomial

#############################################
#            Tests for Bernoulli            #
#############################################
if debug: print('Testing Bernoulli...')
#make distribution
bernoulliElement = ET.Element("bernoulli")
bernoulliElement.append(createElement("p",text="0.4"))
bernoulli = Distributions.Bernoulli()
bernoulli._readMoreXML(bernoulliElement)
bernoulli.initializeDistribution()

ptset = [(-1  , 0, 0.2),
         (-0.5, 0, 0.2),
         ( 0  , 0, 0.2),
         ( 0.5, 1, 1  ),
         ( 1  , 1, 1  )]

doQuads = ['CDF','ClenshawCurtis']
solns={}
solns['CDF']=0.392368073268
solns['ClenshawCurtis']=0.39996427116

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
#link quad to distr
  bernoulli.setQuadrature(quad)
  checkObject("setting %s collocation in bernoulli" %quadname,bernoulli.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("bernoulli-%s std-to-act pt (%f)" %(quadname,pt[0]),bernoulli.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("bernoulli-%s act-to-std pt (%f)" %(quadname,pt[1]),bernoulli.convertDistrPointsToStd(pt[1]),pt[2])

  for i in [14]:
    pts,wts = quad(i)
    pts = bernoulli.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*bernoulli.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*bernoulli.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*bernoulli.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*bernoulli.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*bernoulli.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*bernoulli.probabilityNorm()
    checkAnswer("Bernoulli-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer("Bernoulli-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    checkAnswer("Bernoulli-%s integrate x^1 with O(%i)" %(quadname,i),tot1,solns[quadname]) # analytic nth moment is p for n>=1
    checkAnswer("Bernoulli-%s integrate x^2 with O(%i)" %(quadname,i),tot2,solns[quadname])
    checkAnswer("Bernoulli-%s integrate x^3 with O(%i)" %(quadname,i),tot3,solns[quadname])
    checkAnswer("Bernoulli-%s integrate x^4 with O(%i)" %(quadname,i),tot4,solns[quadname])
del bernoulli

#############################################
#            Tests for Logistic            #
#############################################
if debug: print('Testing Logistic...')
#make distribution
logisticElement = ET.Element("logistic")
logisticElement.append(createElement("location",text="4.0"))
logisticElement.append(createElement("scale",text="1.0"))
logistic = Distributions.Logistic()
logistic._readMoreXML(logisticElement)
logistic.initializeDistribution()

ptset = [(-0.9, 1.05556102083),
         (-0.5, 2.90138771133),
         (   0, 4.0          ),
         ( 0.5, 5.09861228867),
         ( 0.9, 6.94443897917)]

doQuads = ['CDF']

# make sure CC fails
try:
  logistic.setQuadrature(quads['ClenshawCurtis'])
  prevented=False
except IOError:
  prevented=True
checkAnswer('Prevent full Logistic from using ClenshawCurtis',prevented,True)

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
#link quad to distr
  logistic.setQuadrature(quad)
  checkObject("setting %s collocation in logistic" %quadname,logistic.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("logistic-%s std-to-act pt (%f)" %(quadname,pt[0]),logistic.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("logistic-%s act-to-std pt (%i)" %(quadname,pt[1]),logistic.convertDistrPointsToStd(pt[1]),pt[0])

  for i in [14]:
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
    checkAnswer("Logistic-%s integrate x^1 with O(%i)" %(quadname,i),tot1,4.)
    checkAnswer("Logistic-%s integrate x^2 with O(%i)" %(quadname,i),tot2,19.202490836)
    checkAnswer("Logistic-%s integrate x^3 with O(%i)" %(quadname,i),tot3,102.429890032)
    checkAnswer("Logistic-%s integrate x^4 with O(%i)" %(quadname,i),tot4,599.777557399)
del logistic

###############################################
#            Tests for Exponential            #
###############################################
if debug: print('Testing Exponential...')
#TODO this is a subset of gamma, so use laguerre quad
#make distribution
exponentialElement = ET.Element("exponential")
exponentialElement.append(createElement("lambda",text="5.0"))
exponentialElement.append(createElement("low",text="2.0"))
exponential = Distributions.Exponential()
exponential._readMoreXML(exponentialElement)
exponential.initializeDistribution()

#TODO fix these
ptset = [(-1  , 2),
         (-0.5, 2.0575364144904),
         ( 0  , 2.138629436112),
         ( 0.5, 2.277258872224),
         ( 0.9, 2.599146454711)]

#TODO need to implement shifted exponential with self.low like I did for gamma
#doQuads = ['Laguerre','CDF']#,'ClenshawCurtis']
doQuads = ['CDF']#,'ClenshawCurtis']

# make sure CC fails on untruncated normal
try:
  exponential.setQuadrature(quads['ClenshawCurtis'])
  prevented=False
except IOError:
  prevented=True
checkAnswer('Prevent full Exponential from using ClenshawCurtis',prevented,True)

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
#link quad to distr
  exponential.setQuadrature(quad)
  checkObject("setting %s collocation in exponential" %quadname,exponential.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("exponential-%s std-to-act pt (%f)" %(quadname,pt[0]),exponential.convertStdPointsToDistr(pt[0]),pt[1])
   # checkAnswer("exponential-%s act-to-std pt (%i)" %(quadname,pt[1]),exponential.convertDistrPointsToStd(pt[1]),pt[0])

  for i in [14]:
    pts,wts = quad(i)
    pts = exponential.convertStdPointsToDistr(pts)
    totu=0
    tot0=0
    tot1=0
    tot2=0
    tot3=0
    tot4=0
    for p,pt in enumerate(pts):
      totu+=wts[p]*exponential.probabilityNorm()
      tot0+=testPoly(0,pt)*wts[p]*exponential.probabilityNorm()
      tot1+=testPoly(1,pt)*wts[p]*exponential.probabilityNorm()
      tot2+=testPoly(2,pt)*wts[p]*exponential.probabilityNorm()
      tot3+=testPoly(3,pt)*wts[p]*exponential.probabilityNorm()
      tot4+=testPoly(4,pt)*wts[p]*exponential.probabilityNorm()
    checkAnswer("Exponential-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer("Exponential-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    checkAnswer("Exponential-%s integrate x^1 with O(%i)" %(quadname,i),tot1,2.19939887826) #analytic 11/5
    checkAnswer("Exponential-%s integrate x^2 with O(%i)" %(quadname,i),tot2,4.87584842263) #122/25
    checkAnswer("Exponential-%s integrate x^3 with O(%i)" %(quadname,i),tot3,10.9065252234) #1366/125
    checkAnswer("Exponential-%s integrate x^4 with O(%i)" %(quadname,i),tot4,24.6437115563) #15464/625

del exponential

#############################################
#            Tests for Lognormal            #
#############################################
if debug: print('Testing Lognormal...')
#make distribution
lognormalElement = ET.Element("logNormal")
lognormalElement.append(createElement("mean",text="3.0"))
lognormalElement.append(createElement("sigma",text="2.0"))
lognormal = Distributions.LogNormal()
lognormal._readMoreXML(lognormalElement)
lognormal.initializeDistribution()

ptset = [(-1  ,  0),
         (-0.5,  5.2122962603),
         (   0, 20.0855369232),
         ( 0.5, 77.3994365143)]

doQuads = ['CDF']

solns={}
solns['CDF']=[123.878109959,140453.114261,274870509.467,597275620136.0]

# make sure CC fails
try:
  lognormal.setQuadrature(quads['ClenshawCurtis'])
  prevented=False
except IOError:
  prevented=True
checkAnswer('Prevent full Lognormal from using ClenshawCurtis',prevented,True)

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quad to distr
  lognormal.setQuadrature(quad)
  checkObject("setting %s collocation in lognormal" %quadname,lognormal.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("lognormal-%s std-to-act pt (%f)" %(quadname,pt[0]),lognormal.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("lognormal-%s act-to-std pt (%i)" %(quadname,pt[1]),lognormal.convertDistrPointsToStd(pt[1]),pt[0])

  for i in [12]:
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
    checkAnswer("Lognormal-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer("Lognormal-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    checkAnswer("Lognormal-%s integrate x^1 with O(%i)" %(quadname,i),tot1,solns[quadname][0]) #analytic=np.exp(5.))
    checkAnswer("Lognormal-%s integrate x^2 with O(%i)" %(quadname,i),tot2,solns[quadname][1]) #np.exp(14.))
    checkAnswer("Lognormal-%s integrate x^3 with O(%i)" %(quadname,i),tot3,solns[quadname][2]) #np.exp(27.))
    checkAnswer("Lognormal-%s integrate x^4 with O(%i)" %(quadname,i),tot4,solns[quadname][3]) #np.exp(44.))

del lognormal


###########################################
#            Tests for Weibull            #
###########################################
if debug: print('Testing Weibull...')
#make distribution
weibullElement = ET.Element("weibull")
weibullElement.append(createElement("k", text="1.5"))
weibullElement.append(createElement("lambda", text="1.0"))
weibull = Distributions.Weibull()
weibull._readMoreXML(weibullElement)
weibull.initializeDistribution()

ptset = [(-1  , 0),
         (-0.5, 0.435787931703),
         ( 0  , 0.783219768775),
         ( 0.5, 1.24328388488),
         ( 0.9, 2.07811063753)]

doQuads = ['CDF']

# make sure CC fails on untruncated normal
try:
  weibull.setQuadrature(quads['ClenshawCurtis'])
  prevented=False
except IOError:
  prevented=True
checkAnswer('Prevent full Weibull from using ClenshawCurtis',prevented,True)

## perform tests ##
for quadname,quad in quads.iteritems():
  if quadname not in doQuads: continue
  if debug: print('    ...testing %s...' %quadname)
  #link quad to distr
  weibull.setQuadrature(quad)
  checkObject("setting %s collocation in weibull" %quadname,weibull.quadratureSet(),quad)

  for s,pt in enumerate(ptset):
    checkAnswer("weibull-%s std-to-act pt (%f)" %(quadname,pt[0]),weibull.convertStdPointsToDistr(pt[0]),pt[1])
    checkAnswer("weibull-%s act-to-std pt (%i)" %(quadname,pt[1]),weibull.convertDistrPointsToStd(pt[1]),pt[0])

  for i in [14]:
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
    checkAnswer("Weibull-%s integrate weights with O(%i)" %(quadname,i),totu,1.0)
    checkAnswer("Weibull-%s integrate x^0 with O(%i)" %(quadname,i),tot0,1.0)
    checkAnswer("Weibull-%s integrate x^1 with O(%i)" %(quadname,i),tot1,0.901725320721) #analytic=0.902745292951
    checkAnswer("Weibull-%s integrate x^2 with O(%i)" %(quadname,i),tot2,1.18286480485) #analytic=1.190639348759
    checkAnswer("Weibull-%s integrate x^3 with O(%i)" %(quadname,i),tot3,1.95632274001) #analytic = 2.0
    checkAnswer("Weibull-%s integrate x^4 with O(%i)" %(quadname,i),tot4,3.79483016062) #analytic=4.01220130200415
del weibull

print(results)
sys.exit(results["fail"])
