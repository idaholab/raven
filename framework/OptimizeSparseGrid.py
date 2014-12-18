from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import numpy as np
import sys, os
from copy import copy as copy
import cProfile
import pstats

from utils import find_crow

find_crow(os.path.dirname(os.path.abspath(sys.argv[0])))

import Distributions
import Quadrature
import OrthoPolynomials
import IndexSets


debug = False
N = 3 #dimension
L = 4 #polynomial order

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

# index set generation
if debug: print('Testing Index Set generation...')
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

# Test sparse grids #
sparseQuad = Quadrature.SparseQuad()

def quadrule(i):
  return i

sparseQuad.initialize(tdSet.points,quadrule,myDists)
#cProfile.run('sparseQuad.initialize(tdSet.points,quadrule,myDists)','initstats')
#p=pstats.Stats('initstats')
#p.strip_dirs().sort_stats('time').print_stats(5)

print('len',len(sparseQuad))

cProfile.run('sparseQuad.smarterMakeCoeffs()','motstats')
c_mot = sparseQuad.c[:]
pm = pstats.Stats('motstats')
pm.strip_dirs().sort_stats('time').print_stats(3)

print('DEBUG',sparseQuad.indexSet)

#sparseQuad.initialize(tdSet.points,quadrule,myDists)
#cProfile.run('sparseQuad.serialMakeCoeffs()','serstats')
#c_brute = sparseQuad.c[:]
#ps = pstats.Stats('serstats')
#ps.strip_dirs().sort_stats('time').print_stats(3)

#print('Size:',len(c_mot))
#for i in range(len(c_mot)):
#  if not (c_mot[i]==c_brute[i]): print('NOT SAME',c_mot[i],c_brute[i])
#sys.exit()
#sparseQuad.initialize(tpSet.points,quadrule,myDists)
#sparseQuad.initialize(tdSet.points,quadrule,myDists)



print(results)
sys.exit(results["fail"])
