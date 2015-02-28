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


# ISOTROPIC INDEX SETS
print('Starting Index Sets...')
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

tdSet = IndexSets.TotalDegree()
tdSet.initialize(myDists)

hcSet = IndexSets.HyperbolicCross()
hcSet.initialize(myDists)

plt.figure()
ax1=plt.subplot(2,2,1)
x,y = tpSet._xy()
plt.plot(x,y,'o',markersize=15)
plt.title('Tensor Product')

ax2=plt.subplot(2,2,2)
x,y = tdSet._xy()
plt.plot(x,y,'o',markersize=15)
plt.title('Total Degree')

ax3=plt.subplot(2,2,3)
x,y = hcSet._xy()
plt.plot(x,y,'o',markersize=15)
plt.title('Hyperbolic Cross')


# SPARSE GRID QUADRATURE, LEGENDRE
print('Starting Legendre quadrature...')
def quadrule(i):
  return i
sparseQuad = Quadrature.SparseQuad()

plt.figure()
plt.subplot(2,2,1)
print('...tensor product...')
sparseQuad.initialize(tpSet.points,quadrule,myDists)
x,y = sparseQuad._xy()
plt.plot(x,y,'o')
plt.plot([0,0],[-1,1],'k-')
plt.plot([-1,1],[0,0],'k-')
plt.title('Tensor Product, G-L (%i pts)' %len(sparseQuad))

plt.subplot(2,2,2)
print('...total degree...')
sparseQuad.initialize(tdSet.points,quadrule,myDists)
x,y = sparseQuad._xy()
plt.plot(x,y,'o')
plt.plot([0,0],[-1,1],'k-')
plt.plot([-1,1],[0,0],'k-')
plt.title('Total Degree, G-L (%i pts)' %len(sparseQuad))

plt.subplot(2,2,3)
print('...hyperbolic cross...')
sparseQuad.initialize(hcSet.points,quadrule,myDists)
x,y = sparseQuad._xy()
plt.plot(x,y,'o')
plt.plot([0,0],[-1,1],'k-')
plt.plot([-1,1],[0,0],'k-')
plt.title('Hyperbolic Cross, G-L (%i pts)' %len(sparseQuad))

# C-C quadrature #
print('Starting C-C quadrature...')
def quadrule(i):
  try: #iterable case
    return np.array(list((0 if p==0 else 2**p) for p in i))
  except TypeError: #scalar case
    return 0 if i==0 else 2**i

print('...reset dists...')
myDists['y1'].setQuadrature(quads['ClenshawCurtis'])
myDists['y2'].setQuadrature(quads['ClenshawCurtis'])

plt.figure()
plt.subplot(2,2,1)
print('...tensor product...')
sparseQuad.initialize(tpSet.points,quadrule,myDists)
x,y = sparseQuad._xy()
plt.plot(x,y,'o')
plt.plot([0,0],[-1,1],'k-')
plt.plot([-1,1],[0,0],'k-')
plt.title('Tensor Product, C-C (%i pts)' %len(sparseQuad))

plt.subplot(2,2,2)
print('...total degree...')
sparseQuad.initialize(tdSet.points,quadrule,myDists)
x,y = sparseQuad._xy()
plt.plot(x,y,'o')
plt.plot([0,0],[-1,1],'k-')
plt.plot([-1,1],[0,0],'k-')
plt.title('Total Degree, C-C (%i pts)' %len(sparseQuad))

plt.subplot(2,2,3)
print('...hyperbolic cross...')
sparseQuad.initialize(hcSet.points,quadrule,myDists)
x,y = sparseQuad._xy()
plt.plot(x,y,'o')
plt.plot([0,0],[-1,1],'k-')
plt.plot([-1,1],[0,0],'k-')
plt.title('Hyperbolic Cross, C-C (%i pts)' %len(sparseQuad))


plt.show()
