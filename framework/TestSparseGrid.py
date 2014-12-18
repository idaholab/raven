from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import numpy as np
import sys, os
from copy import copy as copy

from utils import find_crow

find_crow(os.path.dirname(os.path.abspath(sys.argv[0])))

import Distributions
import Quadrature
import OrthoPolynomials
import IndexSets


debug = False

def createElement(tag,attrib={},text={}):
  element = ET.Element(tag,attrib)
  element.text = text
  return element

def checkObject(comment,value,expected):
  if value!=expected:
    print(comment,value,"!=",expected)
    results['fail']+=1
  else: results['pass']+=1

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

def checkIndexSet(comment,value,expected):
  #NOTE this test requires identical ordering for value and expected
  same=True
  if len(expected) != len(value):
    same=False
  else:
    for v,val in enumerate(value):
      if val!=expected[v]:
        same=False
  if not same:
    print(comment)
    results['fail']+=1
    for v,val in enumerate(value):
      try:
        print('    ',val,'|',expected[v])
      except IndexError:
        print('    ',val,'|  -',)
    for e in range(len(value),len(expected)):
      print('    ','-  |',expected[e])
  else: results['pass']+=1

results = {'pass':0,'fail':0}

# Generate distributions
distros = {}

uniformElement = ET.Element("uniform")
uniformElement.append(createElement("low",text="-1"))
uniformElement.append(createElement("hi" ,text=" 1"))
uniform = Distributions.Uniform()
uniform._readMoreXML(uniformElement)
uniform.initializeDistribution()
distros['uniform']=uniform

normalElement = ET.Element("normal")
normalElement.append(createElement("mean" ,text="0"))
normalElement.append(createElement("sigma",text="1"))
normal = Distributions.Normal()
normal._readMoreXML(normalElement)
normal.initializeDistribution()
distros['normal']=normal

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

hermiteElement = ET.Element('hermite')
hermite = Quadrature.Hermite()
hermite._readMoreXML(hermiteElement)
hermite.initialize()
quads['Hermite']=hermite

# Generate polynomials
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

# Test index set generation, N=2, L=4
if debug: print('Testing Index Set generation...')

def changePolyOrder(L,distros):
  for dist in distros.values():
    dist.setNewPolyOrder(L)

def makeAnIndexSet(dists,name):
  if   name=='tp': iset = IndexSets.TensorProduct()
  elif name=='td': iset = IndexSets.TotalDegree()
  elif name=='hc': iset = IndexSets.HyperbolicCross()
  iset.initialize(dists)
  return iset

N=2; L=4
myDists={}
y1 = copy(distros['uniform'])
y1.setQuadrature(quads['Legendre'])
y1.setPolynomials(polys['Legendre'],L)

y2 = copy(distros['uniform'])
y2.setQuadrature(quads['Legendre'])
y2.setPolynomials(polys['Legendre'],L)

myDists['y1']=y1
myDists['y2']=y2

tpSet = makeAnIndexSet(myDists,'tp')
correct = [(0,0),(0,1),(0,2),(0,3),(0,4),
           (1,0),(1,1),(1,2),(1,3),(1,4),
           (2,0),(2,1),(2,2),(2,3),(2,4),
           (3,0),(3,1),(3,2),(3,3),(3,4),
           (4,0),(4,1),(4,2),(4,3),(4,4)]
checkIndexSet('Tensor Product set points',tpSet.points,correct)

tdSet = makeAnIndexSet(myDists,'td')
correct = [(0,0),(0,1),(0,2),(0,3),(0,4),
           (1,0),(1,1),(1,2),(1,3),
           (2,0),(2,1),(2,2),
           (3,0),(3,1),
           (4,0)]
checkIndexSet('Total Degree set points',tdSet.points,correct)

hcSet = makeAnIndexSet(myDists,'hc')
correct = [(0,0),(0,1),(0,2),(0,3),(0,4),
           (1,0),(1,1),
           (2,0),
           (3,0),
           (4,0)]
checkIndexSet('Hyperbolic Cross set points',hcSet.points,correct)

# Test Anisotropic index set
wts=[1,2]

correct = [(0,0),(0,1),(0,2),(0,3),(0,4),
           (1,0),(1,1),(1,2),(1,3),(1,4),
           (2,0),(2,1),(2,2),(2,3),(2,4),
           (3,0),(3,1),(3,2),(3,3),(3,4),
           (4,0),(4,1),(4,2),(4,3),(4,4)]
tpSet.initialize(myDists,wts)
checkIndexSet('Tensor Product anisotropic',tpSet.points,correct) #TODO should I implment it so this changes?

correct = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
           (1,0),(1,1),(1,2),(1,3),(1,4),
           (2,0),(2,1),(2,2),
           (3,0)]
tdSet.initialize(myDists,wts)
checkIndexSet('Total Degree anisotropic',tdSet.points,correct)

correct = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),
           (1,0),(1,1),
           (2,0)]
hcSet.initialize(myDists,wts)
checkIndexSet('Hyperbolic Cross anisotropic',hcSet.points,correct)


# Test sparse grids #
if debug: print('Testing Sparse Quad generation...')

def makeASparseGrid(iset,rule,dists):
  sparseQuad = Quadrature.SparseQuad()
  sparseQuad.initialize(iset.points,rule,dists)
  return sparseQuad


def quadrule(i):
  return i

# TEST ability to integrate 2D simple poly
def poly(vals,exps):
  tot=1
  for v,val in enumerate(vals):
    tot*=val**exps[v]
  return tot

def doSparseTest(myDists,trials,testype,label):
  norm = 1
  for dist in myDists.values():
    norm*=dist.probabilityNorm()

  for L in trials.keys():
    changePolyOrder(L,myDists)
    iset = makeAnIndexSet(myDists,testtype)
    sparseQuad = makeASparseGrid(iset,quadrule,myDists)
    #print('DEBUG',label,sparseQuad._pointKey())
    for exp,soln in trials[L].iteritems():
      tot=0
      for i in range(len(sparseQuad)):
        pt,wt = sparseQuad[i]
        tot+=wt*poly(pt,exp)*norm
      checkAnswer(label+' '+testtype+' power '+str(exp)+' using L='+str(L),tot,soln)

# TEST 

#reset to unweighted
tpSet.initialize(myDists,[1,1])
tdSet.initialize(myDists,[1,1])
hcSet.initialize(myDists,[1,1])

testtype='tp'
trials={}
trials[4]={(0,0):1, #L=4, 3, 2 are exact
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./5.,
           (1,1):0,
           (2,2):1./9.,
           (2,4):1./15.,
           (4,4):1./25.,
           (4,3):0}
trials[3]=trials[4]
trials[2]=trials[4]
trials[1]={(0,0):1, #L=1 is not exact
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./9.,
           (1,1):0,
           (2,2):1./9.,
           (2,4):1./27.,
           (4,4):1./81.,
           (4,3):0}
doSparseTest(myDists,trials,'tp','Std Uniform Sparse integ. simple monos')

testtype='td'
trials={}
trials[4]={(0,0):1, #L=4 is exact
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./5.,
           (1,1):0,
           (2,2):1./9.,
           (2,4):1./15.,
           (4,4):1./25.,
           (4,3):0}
trials[3]={(0,0):1,
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./5.,
           (1,1):0,
           (2,2):1./9.,
           (2,4):1./15.,
           (4,4):0.0320987654321,
           (4,3):0}
trials[2]={(0,0):1,
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./5.,
           (1,1):0,
           (2,2):1./9.,
           (2,4):0.037037037037,
           (4,4):0.0123456790123,
           (4,3):0}
trials[1]={(0,0):1, #L=4 is exact
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./9.,
           (1,1):0,
           (2,2):0.0,
           (2,4):0.0,
           (4,4):0.0,
           (4,3):0}
doSparseTest(myDists,trials,'td','Std Uniform Sparse integ. simple monos')

testtype='hc'
trials={}
trials[4]={(0,0):1, #L=4 is exact
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./5.,
           (1,1):0,
           (2,2):1./9.,
           (2,4):1./27.,
           (4,4):1./81.,
           (4,3):0}
trials[3]={(0,0):1,
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./5.,
           (1,1):0,
           (2,2):1./9.,
           (2,4):1./27.,
           (4,4):1./81.,
           (4,3):0}
trials[2]={(0,0):1,
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./5.,
           (1,1):0,
           (2,2):0.0,
           (2,4):0.0,
           (4,4):0.0,
           (4,3):0}
trials[1]={(0,0):1,
           (0,1):0,
           (0,2):1./3.,
           (0,4):1./9.,
           (1,1):0,
           (2,2):0.0,
           (2,4):0.0,
           (4,4):0.0,
           (4,3):0}
doSparseTest(myDists,trials,'hc','Std Uniform Sparse integ. simple monos')

# TEST Mixed Legendre-Hermite
myDists={}
y1 = copy(distros['uniform'])
y1.setQuadrature(quads['Legendre'])
y1.setPolynomials(polys['Legendre'],L)

y2 = copy(distros['normal'])
y2.setQuadrature(quads['Hermite'])
y2.setPolynomials(polys['Hermite'],L)

myDists['y1']=y1
myDists['y2']=y2

tpSet.initialize(myDists)
tdSet.initialize(myDists)
hcSet.initialize(myDists)

testtype='tp'
trials={}
trials[4]={(0,0):1, #L=4, 3, 2 are exact
           (0,1):0,
           (1,0):0,
           (0,2):1,
           (2,0):1.0/3.0,
           (0,4):3,
           (4,0):1.0/5.0,
           (1,1):0,
           (2,2):1.0/3.0,
           (2,4):1,
           (4,2):1.0/5.0,
           (4,4):3.0/5.0,
           (4,3):0}
trials[3]=trials[4]
trials[2]=trials[4]
trials[1]={(0,0):1, #L=4, 3, 2 are exact
           (0,1):0,
           (0,2):1,
           (0,4):1,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):1.0/3.0,
           (2,4):1.0/3.0,
           (4,0):1.0/9.0,
           (4,2):1.0/9.0,
           (4,3):0,
           (4,4):1.0/9.0}
doSparseTest(myDists,trials,'tp','Std Uni-Norm Sparse integ. simple monos')

testtype='td'
trials={}
trials[4]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):3,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):1.0/3.0,
           (2,4):1,
           (4,0):1.0/5.0,
           (4,2):1.0/5.0,
           (4,3):0,
           (4,4):3.0/5.0}
trials[3]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):3,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):1.0/3.0,
           (2,4):1,
           (4,0):1.0/5.0,
           (4,2):1.0/5.0,
           (4,3):0,
           (4,4):19./45.}
trials[2]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):3,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):1.0/3.0,
           (2,4):1.0/3.0,
           (4,0):1.0/5.0,
           (4,2):1.0/9.0,
           (4,3):0,
           (4,4):1./9.}
trials[1]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):1,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):0.0,
           (2,4):0.0,
           (4,0):1.0/9.0,
           (4,2):0.0,
           (4,3):0,
           (4,4):0.0}
doSparseTest(myDists,trials,'td','Std Uni-Norm Sparse integ. simple monos')

testtype='hc'
trials={}
trials[4]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):3,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):1.0/3.0,
           (2,4):1.0/3.0,
           (4,0):1.0/5.0,
           (4,2):1.0/9.0,
           (4,3):0,
           (4,4):1.0/9.0}
trials[3]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):3,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):1.0/3.0,
           (2,4):1.0/3.0,
           (4,0):1.0/5.0,
           (4,2):1.0/9.0,
           (4,3):0,
           (4,4):1./9.}
trials[2]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):3,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):0.0,
           (2,4):0.0,
           (4,0):1.0/5.0,
           (4,2):0.0,
           (4,3):0,
           (4,4):0.0}
trials[1]={(0,0):1,
           (0,1):0,
           (0,2):1,
           (0,4):1,
           (1,0):0,
           (1,1):0,
           (2,0):1.0/3.0,
           (2,2):0.0,
           (2,4):0.0,
           (4,0):1.0/9.0,
           (4,2):0.0,
           (4,3):0,
           (4,4):0.0}
doSparseTest(myDists,trials,'hc','Std Uni-Norm Sparse integ. simple monos')

# TEST Mixed Legendre-Hermite on non-polynomial
def testfunc(vals):
  tot=1
  for v in vals:
    tot*=np.exp(-v)
  return tot

def doSparseNoPolyTest(myDists,trials,testype,label):
  norm = 1
  for dist in myDists.values():
    norm*=dist.probabilityNorm()

  for L in trials.keys():
    changePolyOrder(L,myDists)
    iset = makeAnIndexSet(myDists,testtype)
    sparseQuad = makeASparseGrid(iset,quadrule,myDists)
    #print('DEBUG',label,sparseQuad._pointKey())
    soln = trials[L]
    tot=0
    for i in range(len(sparseQuad)):
      pt,wt = sparseQuad[i]
      tot+=wt*testfunc(pt)*norm
    checkAnswer(label+' '+testtype+' exp(-x-y) using L='+str(L),tot,soln)

myDists={}
y1 = copy(distros['uniform'])
y1.setQuadrature(quads['Legendre'])
y1.setPolynomials(polys['Legendre'],L)

y2 = copy(distros['normal'])
y2.setQuadrature(quads['Hermite'])
y2.setPolynomials(polys['Hermite'],L)

myDists['y1']=y1
myDists['y2']=y2

tpSet.initialize(myDists)
tdSet.initialize(myDists)
hcSet.initialize(myDists)

testtype='tp'
trials={2:1.92515214108,
        4:1.93753003177,
        6:1.93757911914,
        8:1.93757920531271569949912796256405} #exact to place
doSparseNoPolyTest(myDists,trials,'tp','Std Uni-Norm Sparse integ.')

testtype='td'
trials={2:1.90641674888,
        4:1.93736469377,
        6:1.93757861975,
        8:1.93757920447}
doSparseNoPolyTest(myDists,trials,'td','Std Uni-Norm Sparse integ.')

testtype='hc'
trials={2:1.8133609444,
        4:1.91693642634,
        6:1.93535022047,
        8:1.93740653619}
doSparseNoPolyTest(myDists,trials,'hc','Std Uni-Norm Sparse integ.')
print(results)
sys.exit(results["fail"])
