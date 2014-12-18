from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from copy import copy as copy

from utils import find_crow

find_crow(os.path.dirname(os.path.abspath(sys.argv[0])))

import Distributions
import Quadrature
import OrthoPolynomials
import IndexSets

def createElement(tag,attrib={},text={}):
  element = ET.Element(tag,attrib)
  element.text = text
  return element


# Generate distributions
distros = {}

uniformElement = ET.Element("uniform")
uniformElement.append(createElement("low",text="-1"))
uniformElement.append(createElement("hi" ,text=" 1"))
uniform = Distributions.Uniform()
uniform._readMoreXML(uniformElement)
uniform.initializeDistribution()
distros['uniform']=uniform

# Generate quadrature
quads={}

legendreElement = ET.Element("legendre")
legendre = Quadrature.Legendre()
legendre._readMoreXML(legendreElement)
legendre.initialize()
quads['Legendre']=legendre

ccElement = ET.Element("clenshawcurtis")
cc = Quadrature.ClenshawCurtis()
cc._readMoreXML(ccElement)
cc.initialize()
quads['ClenshawCurtis']=cc

# Generate polynomials
polys={}

plegendreElement = ET.Element("plegendre")
plegendre = OrthoPolynomials.Legendre()
plegendre._readMoreXML(plegendreElement)
plegendre.initialize()
polys['Legendre']=plegendre

N=2
print('Calculating for N =',N)

def testfunc(vals):
  return np.prod(np.cos(vals))

actual = np.sin(1)**N #analytic solution

def doIntegration(myDists,sparseQuad):
  norm = np.prod(list(d.probabilityNorm() for d in myDists.values()))

  tot=0
  for i in range(len(sparseQuad)):
    pt,wt = sparseQuad[i]
    tot+=wt*testfunc(pt)*norm
  return tot

def testLevel(L):
  # INDEX SETS
  print('Starting level',L)
  myDists={}
  for n in range(N):
    y = copy(distros['uniform'])
    y.setQuadrature(quads['Legendre'])
    y.setPolynomials(polys['Legendre'],L)
    myDists[n]=y

  tpSet = IndexSets.TensorProduct()
  tpSet.initialize(myDists)

  tdSet = IndexSets.TotalDegree()
  tdSet.initialize(myDists)

  hcSet = IndexSets.HyperbolicCross()
  hcSet.initialize(myDists)

  soln={}
  # SPARSE GRID QUADRATURE, LEGENDRE
  print('...starting Legendre quadrature...')
  sparseQuad = Quadrature.SparseQuad()
  soln['GL']={}

  print('......tensor product...')
  sparseQuad.initialize(tpSet,Quadrature.GaussQuadRule,myDists)
  soln['GL']['TP'] = doIntegration(myDists,sparseQuad)
  soln['GL']['TPn'] = len(sparseQuad)

  print('......total degree...')
  sparseQuad.initialize(tdSet,Quadrature.GaussQuadRule,myDists)
  soln['GL']['TD'] = doIntegration(myDists,sparseQuad)
  soln['GL']['TDn'] = len(sparseQuad)

  print('......hyperbolic cross...')
  sparseQuad.initialize(hcSet,Quadrature.GaussQuadRule,myDists)
  soln['GL']['HC'] = doIntegration(myDists,sparseQuad)
  soln['GL']['HCn'] = len(sparseQuad)

  # C-C quadrature #
  print('...starting C-C quadrature...')
  soln['CC']={}

  for dist in myDists.values():
    dist.setQuadrature(quads['ClenshawCurtis'])

  #print('......tensor product...')
  #sparseQuad.initialize(tpSet,Quadrature.CCQuadRule,myDists)
  #soln['CC']['TP'] = doIntegration(myDists,sparseQuad)
  #soln['CC']['TPn'] = len(sparseQuad)

  print('......total degree...')
  sparseQuad.initialize(tdSet,Quadrature.CCQuadRule,myDists)
  soln['CC']['TD'] = doIntegration(myDists,sparseQuad)
  soln['CC']['TDn'] = len(sparseQuad)

  print('......hyperbolic cross...')
  sparseQuad.initialize(hcSet,Quadrature.CCQuadRule,myDists)
  soln['CC']['HC'] = doIntegration(myDists,sparseQuad)
  soln['CC']['HCn'] = len(sparseQuad)

  return soln

results={}
for L in range(5):
  results[L]=testLevel(L)

GLTP = list(results[i]['GL']['TP'] for i in results.keys())
GLTD = list(results[i]['GL']['TD'] for i in results.keys())
GLHC = list(results[i]['GL']['HC'] for i in results.keys())
#CCTP = list(results[i]['CC']['TP'] for i in results.keys())
CCTD = list(results[i]['CC']['TD'] for i in results.keys())
CCHC = list(results[i]['CC']['HC'] for i in results.keys())

eGLTP = list(abs(results[i]['GL']['TP']-actual)/actual for i in results.keys())
eGLTD = list(abs(results[i]['GL']['TD']-actual)/actual for i in results.keys())
eGLHC = list(abs(results[i]['GL']['HC']-actual)/actual for i in results.keys())
#eCCTP = list(abs(results[i]['CC']['TP']-actual)/actual for i in results.keys())
eCCTD = list(abs(results[i]['CC']['TD']-actual)/actual for i in results.keys())
eCCHC = list(abs(results[i]['CC']['HC']-actual)/actual for i in results.keys())

nGLTP = list(results[i]['GL']['TPn'] for i in results.keys())
nGLTD = list(results[i]['GL']['TDn'] for i in results.keys())
nGLHC = list(results[i]['GL']['HCn'] for i in results.keys())
#nCCTP = list(results[i]['CC']['TPn'] for i in results.keys())
nCCTD = list(results[i]['CC']['TDn'] for i in results.keys())
nCCHC = list(results[i]['CC']['HCn'] for i in results.keys())

import matplotlib.pyplot as plt
plt.figure()
plt.plot(nGLTP,GLTP,'-x',label='GLTP')
plt.plot(nGLTD,GLTD,'-x',label='GLTD')
plt.plot(nGLHC,GLHC,'-x',label='GLHC')
#plt.plot(nCCTP,CCTP,'-o',label='CCTP')
plt.plot(nCCTD,CCTD,'-o',label='CCTD')
plt.plot(nCCHC,CCHC,'-o',label='CCHC')
plt.gca().set_xscale('log')
plt.title('Integrating Non-Polynomial, N=%i' %N)
plt.xlabel('Quadrature Points')
plt.ylabel('Solution')
plt.legend()
#plt.axis([0,100,0.65,0.75])

plt.figure()
plt.loglog(nGLTP,eGLTP,'-x',label='GLTP')
plt.loglog(nGLTD,eGLTD,'-x',label='GLTD')
plt.loglog(nGLHC,eGLHC,'-x',label='GLHC')
#plt.loglog(nCCTP,eCCTP,'-o',label='CCTP')
plt.loglog(nCCTD,eCCTD,'-o',label='CCTD')
plt.loglog(nCCHC,eCCHC,'-o',label='CCHC')
plt.title('Integrating Non-Polynomial, N=%i' %N)
plt.xlabel('Quadrature Points')
plt.ylabel('Rel. Error')
plt.legend()
plt.show()
