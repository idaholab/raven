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

# Test index set generation, N=2, L=4
if debug: print('Testing Index Set generation...')
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

tpSet = IndexSets.TensorProduct()
tpSet.initialize(myDists)
correct = [(0,0),(0,1),(0,2),(0,3),(0,4),
           (1,0),(1,1),(1,2),(1,3),(1,4),
           (2,0),(2,1),(2,2),(2,3),(2,4),
           (3,0),(3,1),(3,2),(3,3),(3,4),
           (4,0),(4,1),(4,2),(4,3),(4,4)]
checkIndexSet('Tensor Product set points',tpSet.points,correct)

correct = [(0,0),(0,1),(0,2),(0,3),(0,4),
           (1,0),(1,1),(1,2),(1,3),
           (2,0),(2,1),(2,2),
           (3,0),(3,1),
           (4,0)]
tdSet = IndexSets.TotalDegree()
tdSet.initialize(myDists)
checkIndexSet('Total Degree set points',tdSet.points,correct)

correct = [(0,0),(0,1),(0,2),(0,3),(0,4),
           (1,0),(1,1),
           (2,0),
           (3,0),
           (4,0)]
hcSet = IndexSets.HyperbolicCross()
hcSet.initialize(myDists)
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
if debug: print('Testing Index Set generation...')


print(results)
sys.exit(results["fail"])
