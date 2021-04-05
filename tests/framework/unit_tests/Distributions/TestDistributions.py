# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  This Module performs Unit Tests for the Distribution class.
  It can not be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import sys, os
import pickle as pk
import numpy as np

# find location of crow, message handler
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4+['framework'])))
sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)

import MessageHandler
import Distributions

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug'})

print (Distributions)
def createElement(tag,attrib={},text={}):
  """
    Method to create a dummy xml element readable by the distribution classes
    @ In, tag, string, the node tag
    @ In, attrib, dict, optional, the attribute of the xml node
    @ In, text, dict, optional, the dict containing what should be in the xml text
  """
  element = ET.Element(tag,attrib)
  element.text = text
  return element

results = {"pass":0,"fail":0}

#def floatNotEqual(a,b):
#  return abs(a - b) > 1e-10

def checkAnswer(comment,value,expected,tol=1e-10, relative=False):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, relative, bool, optional, the tolerance needs be checked relative?
    @ Out, None
  """
  if relative:
    denominator = expected if expected != 0. else 1.0
  diff = abs(value - expected) if not relative else abs(value - expected)/denominator
  if diff > tol:
    print("checking answer",comment,value,"!=",expected)
    results["fail"] += 1
  else:
    results["pass"] += 1

def checkCrowDist(comment,dist,expectedCrowDist):
  """
    Check the consistency of the crow distributions
    @ In, comment, string, a comment
    @ In, dist, instance, the distribution to inquire
    @ In, expectedCrowDist, dict, the dictionary of the expected distribution (with all the parameters)
    @ Out, None
  """
  crowDist = dist.getCrowDistDict()
  if crowDist != expectedCrowDist:
    results["fail"] += 1
    print(comment,'\n',crowDist,'\n',expectedCrowDist)
  else:
    results["pass"] += 1

def checkIntegral(name,dist,low,high,numpts=1e4,tol=1e-3):
  """
    Check the consistency of the pdf integral (cdf)
    @ In, name, string, the name printed out if it fails
    @ In, dist, instance, the distribution to inquire
    @ In, low, float, the lower bound of the dist
    @ In, high, float, the uppper bound of the dist
    @ In, numpts, int, optional, the number of integration points
    @ In, tol, float, optional, the tolerance
    @ Out, None
  """
  xs=np.linspace(low,high,int(numpts))
  dx = (high-low)/float(numpts)
  tot = sum(dist.pdf(x)*dx for x in xs)
  checkAnswer(name+' unity integration',tot,1,tol)

def getDistribution(xmlElement):
  """
    Parses the xmlElement and returns the distribution
  """
  distributionInstance = Distributions.factory.returnInstance(xmlElement.tag)
  distributionInstance.setMessageHandler(mh)
  paramInput = distributionInstance.getInputSpecification()()
  paramInput.parseNode(xmlElement)
  distributionInstance._handleInput(paramInput)
  distributionInstance.initializeDistribution()
  return distributionInstance

#Test module methods
print(Distributions.factory.knownTypes())
#Test error
try:
  Distributions.factory.returnInstance("unknown",'dud')
except:
  print("error worked")

#Test Uniform

uniformElement = ET.Element("Uniform",{"name":"test"})
uniformElement.append(createElement("lowerBound",text="1.0"))
uniformElement.append(createElement("upperBound",text="3.0"))

#ET.dump(uniformElement)

uniform = getDistribution(uniformElement)

#check pickled version as well
pk.dump(uniform,open('testDistrDump.pk','wb'))
puniform=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("uniform",uniform,{'xMin': 1.0, 'type': 'UniformDistribution', 'xMax': 3.0})
checkCrowDist("puniform",puniform,{'xMin': 1.0, 'type': 'UniformDistribution', 'xMax': 3.0})

checkIntegral("uniform",uniform,1,3)

checkAnswer("uniform cdf(1.0)",uniform.cdf(1.0),0.0)
checkAnswer("uniform cdf(2.0)",uniform.cdf(2.0),0.5)
checkAnswer("uniform cdf(3.0)",uniform.cdf(3.0),1.0)
checkAnswer("uniform mean",uniform.untruncatedMean(),2.0)
checkAnswer("uniform stddev",uniform.untruncatedStdDev(),0.5773502691896257) #sqrt((1/12))*(3.0-1.0)
checkAnswer("puniform cdf(1.0)",puniform.cdf(1.0),0.0)
checkAnswer("puniform cdf(2.0)",puniform.cdf(2.0),0.5)
checkAnswer("puniform cdf(3.0)",puniform.cdf(3.0),1.0)

checkAnswer("uniform ppf(0.0)",uniform.ppf(0.0),1.0)
checkAnswer("uniform ppf(0.5)",uniform.ppf(0.5),2.0)
checkAnswer("uniform ppf(1.0)",uniform.ppf(1.0),3.0)
checkAnswer("puniform ppf(0.0)",puniform.ppf(0.0),1.0)
checkAnswer("puniform ppf(0.5)",puniform.ppf(0.5),2.0)
checkAnswer("puniform ppf(1.0)",puniform.ppf(1.0),3.0)

print(uniform.rvs(5),uniform.rvs())
print(puniform.rvs(5),puniform.rvs())

#check rvsWithinCDFbounds
uniform.rvsWithinbounds(1.5,2.5)
puniform.rvsWithinbounds(1.5,2.5)

## Should these be checked?
initParams = uniform.getInitParams()
## Should these be checked?
initParams = puniform.getInitParams()

#Test LogUniform

logUniformElement = ET.Element("LogUniform",{"name":"test"})
logUniformElement.append(createElement("lowerBound",text="1.0"))
logUniformElement.append(createElement("upperBound",text="3.0"))
logUniformElement.append(createElement("base",text="decimal"))

log10Uniform = getDistribution(logUniformElement)

checkAnswer("log10Uniform pdf(10.0)"  ,log10Uniform.pdf(10.0)  ,0.0217147241)
checkAnswer("log10Uniform pdf(100.0)" ,log10Uniform.pdf(100.0) ,0.0021714724)
checkAnswer("log10Uniform pdf(1000.0)",log10Uniform.pdf(1000.0),0.0002171472)

checkAnswer("log10Uniform cdf(10.0)"  ,log10Uniform.cdf(10.0)  ,0)
checkAnswer("log10Uniform cdf(100.0)" ,log10Uniform.cdf(100.0) ,0.5)
checkAnswer("log10Uniform cdf(1000.0)",log10Uniform.cdf(1000.0),1.0)

checkAnswer("log10Uniform ppf(0.0)",log10Uniform.ppf(0.0),10)
checkAnswer("log10Uniform ppf(0.5)",log10Uniform.ppf(0.5),100)
checkAnswer("log10Uniform ppf(1.0)",log10Uniform.ppf(1.0),1000)

#Test Normal
mean=1.0
sigma=2.0
normalElement = ET.Element("Normal",{"name":"test"})
normalElement.append(createElement("mean",text="%f" %mean))
normalElement.append(createElement("sigma",text="%f" %sigma))

normal = getDistribution(normalElement)

#check pickled version as well
pk.dump(normal,open('testDistrDump.pk','wb'))
pnormal=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("normal",normal,{'mu': 1.0, 'sigma': 2.0, 'type': 'NormalDistribution'})
checkCrowDist("pnormal",pnormal,{'mu': 1.0, 'sigma': 2.0, 'type': 'NormalDistribution'})

checkIntegral("normal",normal,mean-5.*sigma,mean+5.*sigma)

checkAnswer("normal cdf(0.0)",normal.cdf(0.0),0.308537538726)
checkAnswer("normal cdf(1.0)",normal.cdf(1.0),0.5)
checkAnswer("normal cdf(2.0)",normal.cdf(2.0),0.691462461274)
checkAnswer("normal mean",normal.untruncatedMean(),1.0)
checkAnswer("normal stddev",normal.untruncatedStdDev(),2.0)
checkAnswer("pnormal cdf(0.0)",pnormal.cdf(0.0),0.308537538726)
checkAnswer("pnormal cdf(1.0)",pnormal.cdf(1.0),0.5)
checkAnswer("pnormal cdf(2.0)",pnormal.cdf(2.0),0.691462461274)

checkAnswer("normal ppf(0.1)",normal.ppf(0.1),-1.56310313109)
checkAnswer("normal ppf(0.5)",normal.ppf(0.5),1.0)
checkAnswer("normal ppf(0.9)",normal.ppf(0.9),3.56310313109)
checkAnswer("pnormal ppf(0.1)",pnormal.ppf(0.1),-1.56310313109)
checkAnswer("pnormal ppf(0.5)",pnormal.ppf(0.5),1.0)
checkAnswer("pnormal ppf(0.9)",pnormal.ppf(0.9),3.56310313109)

checkAnswer("normal mean()",normal.untruncatedMean(),1.0)
checkAnswer("normal median()",normal.untruncatedMedian(),1.0)
checkAnswer("normal mode()",normal.untruncatedMode(),1.0)
checkAnswer("pnormal mean()",pnormal.untruncatedMean(),1.0)
checkAnswer("pnormal median()",pnormal.untruncatedMedian(),1.0)
checkAnswer("pnormal mode()",pnormal.untruncatedMode(),1.0)

print(normal.rvs(5),normal.rvs())
print(pnormal.rvs(5),pnormal.rvs())

#Test Truncated Normal

truncNormalElement = ET.Element("Normal",{"name":"test"})
truncNormalElement.append(createElement("mean",text="1.0"))
truncNormalElement.append(createElement("sigma",text="2.0"))
truncNormalElement.append(createElement("lowerBound",text="-1.0"))
truncNormalElement.append(createElement("upperBound",text="3.0"))

truncNormal = getDistribution(truncNormalElement)

#check pickled version as well
pk.dump(truncNormal,open('testDistrDump.pk','wb'))
ptruncNormal=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("truncNormal",truncNormal,{'xMin': -1.0, 'mu': 1.0, 'type': 'NormalDistribution', 'sigma': 2.0, 'xMax': 3.0})
checkCrowDist("ptruncNormal",ptruncNormal,{'xMin': -1.0, 'mu': 1.0, 'type': 'NormalDistribution', 'sigma': 2.0, 'xMax': 3.0})

checkIntegral("truncNormal",truncNormal,-1.,3.)

checkAnswer("truncNormal cdf(0.0)",truncNormal.cdf(0.0),0.219546787406)
checkAnswer("truncNormal cdf(1.0)",truncNormal.cdf(1.0),0.5)
checkAnswer("truncNormal cdf(2.0)",truncNormal.cdf(2.0),0.780453212594)
checkAnswer("ptruncNormal cdf(0.0)",ptruncNormal.cdf(0.0),0.219546787406)
checkAnswer("ptruncNormal cdf(1.0)",ptruncNormal.cdf(1.0),0.5)
checkAnswer("ptruncNormal cdf(2.0)",ptruncNormal.cdf(2.0),0.780453212594)

checkAnswer("truncNormal ppf(0.1)",truncNormal.ppf(0.1),-0.498029197939)
checkAnswer("truncNormal ppf(0.5)",truncNormal.ppf(0.5),1.0)
checkAnswer("truncNormal ppf(0.9)",truncNormal.ppf(0.9),2.49802919794)
checkAnswer("ptruncNormal ppf(0.1)",ptruncNormal.ppf(0.1),-0.498029197939)
checkAnswer("ptruncNormal ppf(0.5)",ptruncNormal.ppf(0.5),1.0)
checkAnswer("ptruncNormal ppf(0.9)",ptruncNormal.ppf(0.9),2.49802919794)

lowtruncNormalElement = ET.Element("Normal",{"name":"test"})
lowtruncNormalElement.append(createElement("mean",text="1.0"))
lowtruncNormalElement.append(createElement("sigma",text="2.0"))
lowtruncNormalElement.append(createElement("lowerBound",text="-1.0"))
lowtruncNormal = getDistribution(lowtruncNormalElement)

uptruncNormalElement = ET.Element("Normal",{"name":"test"})
uptruncNormalElement.append(createElement("mean",text="1.0"))
uptruncNormalElement.append(createElement("sigma",text="2.0"))
uptruncNormalElement.append(createElement("upperBound",text="3.0"))
uptruncNormal = getDistribution(uptruncNormalElement)


#Test Gamma

gammaElement = ET.Element("Gamma",{"name":"test"})
gammaElement.append(createElement("low",text="0.0"))
gammaElement.append(createElement("alpha",text="1.0"))
gammaElement.append(createElement("beta",text="0.5"))

gamma = getDistribution(gammaElement)

## Should these be checked?
initParams = gamma.getInitParams()

#check pickled version as well
pk.dump(gamma,open('testDistrDump.pk','wb'))
pgamma=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("gamma",gamma,{'xMin': 0.0, 'theta': 2.0, 'k': 1.0, 'type': 'GammaDistribution', 'low': 0.0})
checkCrowDist("pgamma",pgamma,{'xMin': 0.0, 'theta': 2.0, 'k': 1.0, 'type': 'GammaDistribution', 'low': 0.0})

checkIntegral("gamma",gamma,0.,50.,numpts=1e5)

checkAnswer("gamma cdf(0.0)",gamma.cdf(0.0),0.0)
checkAnswer("gamma cdf(1.0)",gamma.cdf(1.0),0.393469340287)
checkAnswer("gamma cdf(10.0)",gamma.cdf(10.0),0.993262053001)
checkAnswer("gamma mean",gamma.untruncatedMean(),1.0/0.5)
checkAnswer("gamma stddev",gamma.untruncatedStdDev(),2.0) #sqrt(1.0/0.5**2)
checkAnswer("pgamma cdf(0.0)",pgamma.cdf(0.0),0.0)
checkAnswer("pgamma cdf(1.0)",pgamma.cdf(1.0),0.393469340287)
checkAnswer("pgamma cdf(10.0)",pgamma.cdf(10.0),0.993262053001)

checkAnswer("gamma ppf(0.1)",gamma.ppf(0.1),0.210721031316)
checkAnswer("gamma ppf(0.5)",gamma.ppf(0.5),1.38629436112)
checkAnswer("gamma ppf(0.9)",gamma.ppf(0.9),4.60517018599)
checkAnswer("pgamma ppf(0.1)",pgamma.ppf(0.1),0.210721031316)
checkAnswer("pgamma ppf(0.5)",pgamma.ppf(0.5),1.38629436112)
checkAnswer("pgamma ppf(0.9)",pgamma.ppf(0.9),4.60517018599)

nobeta_gammaElement = ET.Element("Gamma",{"name":"test"})
nobeta_gammaElement.append(createElement("alpha",text="1.0"))
nobeta_gammaElement.append(createElement("low",text="0.0"))
nobeta_gammaElement.append(createElement("upperBound",text="10.0"))
nobeta_gamma = getDistribution(nobeta_gammaElement)

print(gamma.rvs(5),gamma.rvs())

# shifted gamma
gammaElement = ET.Element("Gamma",{"name":"test"})
gammaElement.append(createElement("low",text="10.0"))
gammaElement.append(createElement("alpha",text="1.0"))
gammaElement.append(createElement("beta",text="0.5"))

gamma = getDistribution(gammaElement)

## Should these be checked?
initParams = gamma.getInitParams()

checkCrowDist("shifted gamma",gamma,{'xMin': 10.0, 'theta': 2.0, 'k': 1.0, 'type': 'GammaDistribution', 'low': 10.0})

checkIntegral("shifted gamma",gamma,10.,60.,numpts=1e5)

checkAnswer("shifted gamma cdf(10.0)",gamma.cdf(10.0),0.0)
checkAnswer("shifted gamma cdf(11.0)",gamma.cdf(11.0),0.393469340287)
checkAnswer("shifted gamma cdf(20.0)",gamma.cdf(20.0),0.993262053001)

checkAnswer("shifted gamma ppf(0.1)",gamma.ppf(0.1),10.210721031316)
checkAnswer("shifted gamma ppf(0.5)",gamma.ppf(0.5),11.38629436112)
checkAnswer("shifted gamma ppf(0.9)",gamma.ppf(0.9),14.60517018599)


#Test Beta

betaElement = ET.Element("Beta",{"name":"test"})
betaElement.append(createElement("low",text="0.0"))
betaElement.append(createElement("high",text="1.0"))
betaElement.append(createElement("alpha",text="5.0"))
betaElement.append(createElement("beta",text="2.0"))

beta = getDistribution(betaElement)

#check pickled version as well
pk.dump(beta,open('testDistrDump.pk','wb'))
pbeta=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("beta",beta,{'scale': 1.0, 'beta': 2.0, 'low':0.0, 'xMax': 1.0, 'xMin': 0.0, 'alpha': 5.0, 'type': 'BetaDistribution'})
checkCrowDist("pbeta",pbeta,{'scale': 1.0, 'beta': 2.0, 'low':0.0, 'xMax': 1.0, 'xMin': 0.0, 'alpha': 5.0, 'type': 'BetaDistribution'})

checkIntegral("beta",beta,0.0,1.0)

checkAnswer("beta cdf(0.1)",beta.cdf(0.1),5.5e-05)
checkAnswer("beta cdf(0.5)",beta.cdf(0.5),0.109375)
checkAnswer("beta cdf(0.9)",beta.cdf(0.9),0.885735)
checkAnswer("beta mean",beta.untruncatedMean(),0.7142857142857143) # 5.0/(2.0+5.0)
checkAnswer("beta stddev",beta.untruncatedStdDev(),0.15971914124998499) # a=5.0;b=2.0;sqrt((a*b)/((a+b)**2*(a+b+1)))
checkAnswer("pbeta cdf(0.1)",pbeta.cdf(0.1),5.5e-05)
checkAnswer("pbeta cdf(0.5)",pbeta.cdf(0.5),0.109375)
checkAnswer("pbeta cdf(0.9)",pbeta.cdf(0.9),0.885735)

checkAnswer("beta ppf(0.1)",beta.ppf(0.1),0.489683693449)
checkAnswer("beta ppf(0.5)",beta.ppf(0.5),0.735550016704)
checkAnswer("beta ppf(0.9)",beta.ppf(0.9),0.907404741087)
checkAnswer("pbeta ppf(0.1)",pbeta.ppf(0.1),0.489683693449)
checkAnswer("pbeta ppf(0.5)",pbeta.ppf(0.5),0.735550016704)
checkAnswer("pbeta ppf(0.9)",pbeta.ppf(0.9),0.907404741087)

checkAnswer("beta mean()",beta.untruncatedMean(),5.0/(5.0+2.0))
checkAnswer("beta median()",beta.untruncatedMedian(),0.735550016704)
checkAnswer("beta mode()",beta.untruncatedMode(),(5.0-1)/(5.0+2.0-2))
checkAnswer("pbeta mean()",pbeta.untruncatedMean(),5.0/(5.0+2.0))
checkAnswer("pbeta median()",pbeta.untruncatedMedian(),0.735550016704)
checkAnswer("pbeta mode()",pbeta.untruncatedMode(),(5.0-1)/(5.0+2.0-2))

checkAnswer("beta pdf(0.25)",beta.pdf(0.25),0.087890625)
checkAnswer("beta cdfComplement(0.25)",beta.untruncatedCdfComplement(0.25),0.995361328125)
checkAnswer("beta hazard(0.25)",beta.untruncatedHazard(0.25),0.0883002207506)
checkAnswer("pbeta pdf(0.25)",pbeta.pdf(0.25),0.087890625)
checkAnswer("pbeta cdfComplement(0.25)",pbeta.untruncatedCdfComplement(0.25),0.995361328125)
checkAnswer("pbeta hazard(0.25)",pbeta.untruncatedHazard(0.25),0.0883002207506)

print(beta.rvs(5),beta.rvs())
print(pbeta.rvs(5),pbeta.rvs())

#Test Beta Scaled

betaElement = ET.Element("Beta",{"name":"test"})
betaElement.append(createElement("low",text="0.0"))
betaElement.append(createElement("high",text="4.0"))
betaElement.append(createElement("alpha",text="5.0"))
betaElement.append(createElement("beta",text="1.0"))

beta = getDistribution(betaElement)

checkCrowDist("scaled beta",beta,{'scale': 4.0, 'beta': 1.0, 'low':0.0, 'xMax': 4.0, 'xMin': 0.0, 'alpha': 5.0, 'type': 'BetaDistribution'})

checkIntegral("scaled beta",beta,0.0,4.0)

checkAnswer("scaled beta cdf(0.1)",beta.cdf(0.1),9.765625e-09)
checkAnswer("scaled beta cdf(0.5)",beta.cdf(0.5),3.0517578125e-05)
checkAnswer("scaled beta cdf(0.9)",beta.cdf(0.9),0.000576650390625)

checkAnswer("scaled beta ppf(0.1)",beta.ppf(0.1),2.52382937792)
checkAnswer("scaled beta ppf(0.5)",beta.ppf(0.5),3.48220225318)
checkAnswer("scaled beta ppf(0.9)",beta.ppf(0.9),3.91659344944)

print(beta.rvs(5),beta.rvs())

#Test Beta Shifted and Scaled

betaElement = ET.Element("Beta",{"name":"test"})
betaElement.append(createElement("low",text="-1.0"))
betaElement.append(createElement("high",text="5.0"))
betaElement.append(createElement("alpha",text="5.0"))
betaElement.append(createElement("beta",text="2.0"))

beta = getDistribution(betaElement)

checkCrowDist("shifted beta",beta,{'scale': 6.0, 'beta': 2.0, 'low':-1.0, 'xMax': 5.0, 'xMin': -1.0, 'alpha': 5.0, 'type': 'BetaDistribution'})

checkIntegral("shifted beta",beta,-1.0,5.0)

checkAnswer("shifted beta cdf(-0.5)",beta.cdf(-0.5),2.2438164437585733882e-5)
checkAnswer("shifted beta cdf( 0.5)",beta.cdf( 0.5),4.638671875e-3)
checkAnswer("shifted beta cdf( 3.5)",beta.cdf( 3.5),5.33935546875e-1)

checkAnswer("shifted beta ppf(0.1)",beta.ppf(0.1),1.93810216069)
checkAnswer("shifted beta ppf(0.5)",beta.ppf(0.5),3.41330010023)
checkAnswer("shifted beta ppf(0.9)",beta.ppf(0.9),4.44442844652)

print(beta.rvs(5),beta.rvs())

#Test Truncated-Normal-Like Beta
betanElement = ET.Element("Beta",{"name":"test"})
betanElement.append(createElement("low",text="1.0"))
betanElement.append(createElement("high",text="5.0"))
betanElement.append(createElement("peakFactor",text="0.5"))

betan = getDistribution(betanElement)

checkCrowDist("truncnormal beta",betan,{'scale': 4.0, 'beta': 7.520872400521023, 'low':1.0, 'xMax': 5.0, 'xMin': 1.0, 'alpha': 7.520872400521023, 'type': 'BetaDistribution'})

#do an integral
checkIntegral("truncnormal beta",betan,1.0,5.0)

checkAnswer("truncnormal beta cdf(1.0)",betan.cdf(1.0),0)
checkAnswer("truncnormal beta cdf(2.0)",betan.cdf(2.0),0.020339936921)
checkAnswer("truncnormal beta cdf(3.0)",betan.cdf(3.0),0.5)
checkAnswer("truncnormal beta cdf(4.0)",betan.cdf(4.0),0.979660063079)
checkAnswer("truncnormal beta cdf(5.0)",betan.cdf(5.0),1)

checkAnswer("truncnormal ppf(0.1)",betan.ppf(0.1),2.34668338772)
checkAnswer("truncnormal ppf(0.5)",betan.ppf(0.5),3.0)
checkAnswer("truncnormal ppf(0.9)",betan.ppf(0.9),3.65331661228)

print(betan.rvs(5),betan.rvs())

#Test Triangular

triangularElement = ET.Element("Triangular",{"name":"test"})
triangularElement.append(createElement("min",text="0.0"))
triangularElement.append(createElement("apex",text="3.0"))
triangularElement.append(createElement("max",text="4.0"))

triangular = getDistribution(triangularElement)

#check pickled version as well
pk.dump(triangular,open('testDistrDump.pk','wb'))
ptriangular=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("triangular",triangular,{'lowerBound': 0.0, 'type': 'TriangularDistribution', 'upperBound': 4.0, 'xMax': 4.0, 'xMin': 0.0, 'xPeak': 3.0})
checkCrowDist("ptriangular",ptriangular,{'lowerBound': 0.0, 'type': 'TriangularDistribution', 'upperBound': 4.0, 'xMax': 4.0, 'xMin': 0.0, 'xPeak': 3.0})

checkIntegral("triangular",triangular,0.0,4.0)

checkAnswer("triangular cdf(0.25)",triangular.cdf(0.25),0.00520833333333)
checkAnswer("triangular cdf(3.0)",triangular.cdf(3.0),0.75)
checkAnswer("triangular cdf(3.5)",triangular.cdf(3.5),0.9375)
checkAnswer("triangular mean",triangular.untruncatedMean(),2.33333333333)
checkAnswer("triangular stddev",triangular.untruncatedStdDev(),0.849836585599)
checkAnswer("ptriangular cdf(0.25)",ptriangular.cdf(0.25),0.00520833333333)
checkAnswer("ptriangular cdf(3.0)",ptriangular.cdf(3.0),0.75)
checkAnswer("ptriangular cdf(3.5)",ptriangular.cdf(3.5),0.9375)

checkAnswer("triangular ppf(0.1)",triangular.ppf(0.1),1.09544511501)
checkAnswer("triangular ppf(0.5)",triangular.ppf(0.5),2.44948974278)
checkAnswer("triangular ppf(0.9)",triangular.ppf(0.9),3.36754446797)
checkAnswer("ptriangular ppf(0.1)",ptriangular.ppf(0.1),1.09544511501)
checkAnswer("ptriangular ppf(0.5)",ptriangular.ppf(0.5),2.44948974278)
checkAnswer("ptriangular ppf(0.9)",ptriangular.ppf(0.9),3.36754446797)

print(triangular.rvs(5),triangular.rvs())
print(ptriangular.rvs(5),ptriangular.rvs())

#Shift Triangular

triangularElement = ET.Element("Triangular",{"name":"test"})
triangularElement.append(createElement("min",text="5.0"))
triangularElement.append(createElement("apex",text="8.0"))
triangularElement.append(createElement("max",text="9.0"))

triangular = getDistribution(triangularElement)

checkCrowDist("shift triangular",triangular,{'lowerBound': 5.0, 'type': 'TriangularDistribution', 'upperBound': 9.0, 'xMax': 9.0, 'xMin': 5.0, 'xPeak': 8.0})

checkIntegral("shift triangular",triangular,5.0,9.0)

checkAnswer("shift triangular cdf(0.25)",triangular.cdf(5.25),0.00520833333333)
checkAnswer("shift triangular cdf(3.0)",triangular.cdf(8.0),0.75)
checkAnswer("shift triangular cdf(3.5)",triangular.cdf(8.5),0.9375)

checkAnswer("shift triangular ppf(0.1)",triangular.ppf(0.1),6.09544511501)
checkAnswer("shift triangular ppf(0.5)",triangular.ppf(0.5),7.44948974278)
checkAnswer("shift triangular ppf(0.9)",triangular.ppf(0.9),8.36754446797)

#Test Poisson

poissonElement = ET.Element("Poisson",{"name":"test"})
poissonElement.append(createElement("mu",text="4.0"))

poisson = getDistribution(poissonElement)

## Should these be checked?
initParams = poisson.getInitParams()

#check pickled version as well
pk.dump(poisson,open('testDistrDump.pk','wb'))
ppoisson=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("poisson",poisson,{'mu': 4.0, 'type': 'PoissonDistribution'})
checkCrowDist("ppoisson",ppoisson,{'mu': 4.0, 'type': 'PoissonDistribution'})

checkIntegral("poisson",poisson,0.0,1000.0, numpts=1000)

checkAnswer("poisson cdf(1.0)",poisson.cdf(1.0),0.0915781944437)
checkAnswer("poisson cdf(5.0)",poisson.cdf(5.0),0.7851303870304052)
checkAnswer("poisson cdf(10.0)",poisson.cdf(10.0),0.997160233879)
checkAnswer("poisson mean",poisson.untruncatedMean(),4.0)
checkAnswer("poisson stddev",poisson.untruncatedStdDev(),2.0)
checkAnswer("ppoisson cdf(1.0)",ppoisson.cdf(1.0),0.0915781944437)
checkAnswer("ppoisson cdf(5.0)",ppoisson.cdf(5.0),0.7851303870304052)
checkAnswer("ppoisson cdf(10.0)",ppoisson.cdf(10.0),0.997160233879)

checkAnswer("poisson ppf(0.0915781944437)",poisson.ppf(0.0915781944437),1.0)
checkAnswer("poisson ppf(0.785130387030405)",poisson.ppf(0.785130387030405),5.0)
checkAnswer("poisson ppf(0.997160233879)",poisson.ppf(0.997160233879),10.0)
checkAnswer("ppoisson ppf(0.0915781944437)",ppoisson.ppf(0.0915781944437),1.0)
checkAnswer("ppoisson ppf(0.785130387030405)",ppoisson.ppf(0.785130387030405),5.0)
checkAnswer("ppoisson ppf(0.997160233879)",ppoisson.ppf(0.997160233879),10.0)

print(poisson.rvs(5),poisson.rvs())
print(ppoisson.rvs(5),ppoisson.rvs())

#Test Binomial

binomialElement = ET.Element("Binomial",{"name":"test"})
binomialElement.append(createElement("n",text="10"))
binomialElement.append(createElement("p",text="0.25"))

binomial = getDistribution(binomialElement)

## Should these be checked?
initParams = binomial.getInitParams()

#check picklling
pk.dump(binomial,open('testDistrDump.pk','wb'))
pbinomial=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("binomial",binomial,{'p': 0.25, 'type': 'BinomialDistribution', 'n': 10.0})
checkCrowDist("pbinomial",pbinomial,{'p': 0.25, 'type': 'BinomialDistribution', 'n': 10.0})

checkIntegral("binomial",binomial,0.0,10.0,numpts=100,tol=3e-2) #TODO why is this so hard to integrate?

checkAnswer("binomial cdf(1)",binomial.cdf(1),0.244025230408)
checkAnswer("binomial cdf(2)",binomial.cdf(2),0.525592803955)
checkAnswer("binomial cdf(5)",binomial.cdf(5),0.980272293091)
checkAnswer("binomial mean",binomial.untruncatedMean(),2.5)
checkAnswer("binomial stddev",binomial.untruncatedStdDev(),1.3693063937629153) #sqrt(0.25*10*(1-0.25))
checkAnswer("pbinomial cdf(1)",pbinomial.cdf(1),0.244025230408)
checkAnswer("pbinomial cdf(2)",pbinomial.cdf(2),0.525592803955)
checkAnswer("pbinomial cdf(5)",pbinomial.cdf(5),0.980272293091)

checkAnswer("binomial ppf(0.1)",binomial.ppf(0.1),0.0)
checkAnswer("binomial ppf(0.5)",binomial.ppf(0.5),2.0)
checkAnswer("binomial ppf(0.9)",binomial.ppf(0.9),4.0)
checkAnswer("pbinomial ppf(0.1)",pbinomial.ppf(0.1),0.0)
checkAnswer("pbinomial ppf(0.5)",pbinomial.ppf(0.5),2.0)
checkAnswer("pbinomial ppf(0.9)",pbinomial.ppf(0.9),4.0)


#Test Bernoulli

bernoulliElement = ET.Element("Bernoulli",{"name":"test"})
bernoulliElement.append(createElement("p",text="0.4"))

bernoulli = getDistribution(bernoulliElement)

## Should these be checked?
initParams = bernoulli.getInitParams()

#check picklling
pk.dump(bernoulli,open('testDistrDump.pk','wb'))
pbernoulli=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("bernoulli",bernoulli,{'p': 0.4, 'type': 'BernoulliDistribution'})
checkCrowDist("pbernoulli",pbernoulli,{'p': 0.4, 'type': 'BernoulliDistribution'})

#checkIntegral("bernoulli",bernoulli,0.0,1.0,numpts=2) #why does this integrate to 0.5?

checkAnswer("bernoulli cdf(0)",bernoulli.cdf(0),0.6)
checkAnswer("bernoulli cdf(1)",bernoulli.cdf(1),1.0)
checkAnswer("bernoulli mean",bernoulli.untruncatedMean(),0.4)
checkAnswer("bernoulli stddev",bernoulli.untruncatedStdDev(),0.4898979485566356) #sqrt(0.4*(1-0.4))
checkAnswer("pbernoulli cdf(0)",pbernoulli.cdf(0),0.6)
checkAnswer("pbernoulli cdf(1)",pbernoulli.cdf(1),1.0)

checkAnswer("bernoulli ppf(0.1)",bernoulli.ppf(0.1),0.0)
checkAnswer("bernoulli ppf(0.3)",bernoulli.ppf(0.3),0.0)
checkAnswer("bernoulli ppf(0.8)",bernoulli.ppf(0.8),1.0)
checkAnswer("bernoulli ppf(0.9)",bernoulli.ppf(0.9),1.0)
checkAnswer("pbernoulli ppf(0.1)",pbernoulli.ppf(0.1),0.0)
checkAnswer("pbernoulli ppf(0.3)",pbernoulli.ppf(0.3),0.0)
checkAnswer("pbernoulli ppf(0.8)",pbernoulli.ppf(0.8),1.0)
checkAnswer("pbernoulli ppf(0.9)",pbernoulli.ppf(0.9),1.0)

#Test Geometric

geometricElement = ET.Element("Geometric",{"name":"test"})
geometricElement.append(createElement("p",text="0.25"))

geometric = getDistribution(geometricElement)

checkCrowDist("geometric",geometric,{'p': 0.25, 'type': 'GeometricDistribution'})

checkAnswer("geometric cdf(0)",geometric.cdf(0),0.25)
checkAnswer("geometric cdf(1)",geometric.cdf(1),0.4375)
checkAnswer("geometric mean",geometric.untruncatedMean(),3.0)
checkAnswer("geometric stddev",geometric.untruncatedStdDev(),3.46410161514)

checkAnswer("geometric ppf(0.1)",geometric.ppf(0.1),0.0)
checkAnswer("geometric ppf(0.3)",geometric.ppf(0.3),0.239823326142)
checkAnswer("geometric ppf(0.8)",geometric.ppf(0.8),4.59450194)
checkAnswer("geometric ppf(0.9)",geometric.ppf(0.9),7.00392277965)


#Test Logistic

logisticElement = ET.Element("Logistic",{"name":"test"})
logisticElement.append(createElement("location",text="4.0"))
logisticElement.append(createElement("scale",text="1.0"))

logistic = getDistribution(logisticElement)

## Should these be checked?
initParams = logistic.getInitParams()

#check picklling
pk.dump(logistic,open('testDistrDump.pk','wb'))
plogistic=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("logistic",logistic,{'scale': 1.0, 'type': 'LogisticDistribution', 'location': 4.0})
checkCrowDist("plogistic",plogistic,{'scale': 1.0, 'type': 'LogisticDistribution', 'location': 4.0})

checkIntegral("logistic",logistic,-5.0,13.0)

checkAnswer("logistic cdf(0)",logistic.cdf(0.0),0.0179862099621)
checkAnswer("logistic cdf(4)",logistic.cdf(4.0),0.5)
checkAnswer("logistic cdf(8)",logistic.cdf(8.0),0.982013790038)
checkAnswer("logistic mean",logistic.untruncatedMean(),4.0)
checkAnswer("logistic stddev",logistic.untruncatedStdDev(), 1.81379936423)
checkAnswer("plogistic cdf(0)",plogistic.cdf(0.0),0.0179862099621)
checkAnswer("plogistic cdf(4)",plogistic.cdf(4.0),0.5)
checkAnswer("plogistic cdf(8)",plogistic.cdf(8.0),0.982013790038)

checkAnswer("logistic ppf(0.25)",logistic.ppf(0.25),2.90138771133)
checkAnswer("logistic ppf(0.50)",logistic.ppf(0.50),4.0)
checkAnswer("logistic ppf(0.75)",logistic.ppf(0.75),5.09861228867)
checkAnswer("plogistic ppf(0.25)",plogistic.ppf(0.25),2.90138771133)
checkAnswer("plogistic ppf(0.50)",plogistic.ppf(0.50),4.0)
checkAnswer("plogistic ppf(0.75)",plogistic.ppf(0.75),5.09861228867)

lowLogisticElement = ET.Element("Logistic",{"name":"test"})
lowLogisticElement.append(createElement("location",text="4.0"))
lowLogisticElement.append(createElement("scale",text="1.0"))
lowLogisticElement.append(createElement("lowerBound",text="3.0"))
lowLogistic = getDistribution(lowLogisticElement)

upLogisticElement = ET.Element("Logistic",{"name":"test"})
upLogisticElement.append(createElement("location",text="4.0"))
upLogisticElement.append(createElement("scale",text="1.0"))
upLogisticElement.append(createElement("upperBound",text="5.0"))
upLogistic = getDistribution(upLogisticElement)

#Test Laplace

laplaceElement = ET.Element("Laplace",{"name":"test"})
laplaceElement.append(createElement("location",text="0.0"))
laplaceElement.append(createElement("scale",text="2.0"))

laplace = getDistribution(laplaceElement)

## Should these be checked?
initParams = laplace.getInitParams()

checkCrowDist("laplace",laplace,{'scale': 2.0, 'type': 'LaplaceDistribution', 'location': 0.0})

checkIntegral("laplace",laplace,-20.0,20.0)

checkAnswer("laplace cdf(0)",laplace.cdf(0.0),0.5)
checkAnswer("laplace cdf(4)",laplace.cdf(1.0),0.696734670144)
checkAnswer("laplace cdf(8)",laplace.cdf(2.0),0.816060279414)
checkAnswer("laplace mean",laplace.untruncatedMean(),0.0)
checkAnswer("laplace stddev",laplace.untruncatedStdDev(), 2.82842712475)

checkAnswer("laplace ppf(0.25)",laplace.ppf(0.25),-1.38629436112)
checkAnswer("laplace ppf(0.50)",laplace.ppf(0.50),0.0)
checkAnswer("laplace ppf(0.75)",laplace.ppf(0.75),1.38629436112)


#Test Exponential

exponentialElement = ET.Element("Exponential",{"name":"test"})
exponentialElement.append(createElement("lambda",text="5.0"))

exponential = getDistribution(exponentialElement)

#check picklling
pk.dump(exponential,open('testDistrDump.pk','wb'))
pexponential=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("exponential",exponential,{'xMin': 0.0, 'type': 'ExponentialDistribution', 'lambda': 5.0, 'low':0.0})
checkCrowDist("pexponential",pexponential,{'xMin': 0.0, 'type': 'ExponentialDistribution', 'lambda': 5.0, 'low':0.0})

checkIntegral("exponential",exponential,0.0,1.5)

checkAnswer("exponential cdf(0.3)",exponential.cdf(0.3),0.7768698399)
checkAnswer("exponential cdf(1.0)",exponential.cdf(1.0),0.993262053001)
checkAnswer("exponential cdf(3.0)",exponential.cdf(3.0),0.999999694098)
checkAnswer("exponential mean",exponential.untruncatedMean(),1/5.0)
checkAnswer("exponential stddev",exponential.untruncatedStdDev(),1/5.0)
checkAnswer("pexponential cdf(0.3)",pexponential.cdf(0.3),0.7768698399)
checkAnswer("pexponential cdf(1.0)",pexponential.cdf(1.0),0.993262053001)
checkAnswer("pexponential cdf(3.0)",pexponential.cdf(3.0),0.999999694098)

checkAnswer("exponential ppf(0.7768698399)",exponential.ppf(0.7768698399),0.3)
checkAnswer("exponential ppf(0.2)",exponential.ppf(0.2),0.0446287102628)
checkAnswer("exponential ppf(0.5)",exponential.ppf(0.5),0.138629436112)
checkAnswer("pexponential ppf(0.7768698399)",pexponential.ppf(0.7768698399),0.3)
checkAnswer("pexponential ppf(0.2)",pexponential.ppf(0.2),0.0446287102628)
checkAnswer("pexponential ppf(0.5)",pexponential.ppf(0.5),0.138629436112)

lowExponentialElement = ET.Element("Exponential",{"name":"test"})
lowExponentialElement.append(createElement("lambda",text="5.0"))
lowExponentialElement.append(createElement("lowerBound",text="0.0"))
lowExponential = getDistribution(lowExponentialElement)

upExponentialElement = ET.Element("Exponential",{"name":"test"})
upExponentialElement.append(createElement("lambda",text="5.0"))
upExponentialElement.append(createElement("upperBound",text="10.0"))
upExponential = getDistribution(upExponentialElement)

#Test truncated exponential

truncExponentialElement = ET.Element("Exponential",{"name":"test"})
truncExponentialElement.append(createElement("lambda",text="5.0"))
truncExponentialElement.append(createElement("lowerBound",text="0.0"))
truncExponentialElement.append(createElement("upperBound",text="10.0"))

truncExponential = getDistribution(truncExponentialElement)

checkCrowDist("truncExponential",truncExponential,{'xMin': 0.0, 'type': 'ExponentialDistribution', 'xMax': 10.0, 'lambda': 5.0, 'low':0.0})

checkIntegral("truncExponential",truncExponential,0.0,1.5) #TODO this doesn't actually test anything new.  This truncation is way out past the effective pdf.

checkAnswer("truncExponential cdf(0.1)",truncExponential.cdf(0.1),0.393469340287)
checkAnswer("truncExponential cdf(5.0)",truncExponential.cdf(5.0),0.999999999986)
checkAnswer("truncExponential cdf(9.9)",truncExponential.cdf(9.9),1.0)


checkAnswer("truncExponential ppf(0.1)",truncExponential.ppf(0.1),0.0210721031316)
checkAnswer("truncExponential ppf(0.5)",truncExponential.ppf(0.5),0.138629436112)
checkAnswer("truncExponential ppf(0.9)",truncExponential.ppf(0.9),0.460517018599)

#Shift Exponential

exponentialElement = ET.Element("Exponential",{"name":"test"})
exponentialElement.append(createElement("lambda",text="5.0"))
exponentialElement.append(createElement("low",text="10.0"))

exponential = getDistribution(exponentialElement)

checkCrowDist("shifted exponential",exponential,{'xMin': 10.0, 'type': 'ExponentialDistribution', 'lambda': 5.0, 'low':10.0})

checkIntegral("shifted exponential",exponential,10.0,11.5)

checkAnswer("shifted exponential cdf(0.3)",exponential.cdf(10.3),0.7768698399)
checkAnswer("shifted exponential cdf(1.0)",exponential.cdf(11.0),0.993262053001)
checkAnswer("shifted exponential cdf(3.0)",exponential.cdf(13.0),0.999999694098)

checkAnswer("shifted exponential ppf(0.7768698399)",exponential.ppf(0.7768698399),10.3)
checkAnswer("shifted exponential ppf(0.2)",exponential.ppf(0.2),10.0446287102628)
checkAnswer("shifted exponential ppf(0.5)",exponential.ppf(0.5),10.138629436112)


#Test log normal

logNormalElement = ET.Element("LogNormal",{"name":"test"})
logNormalElement.append(createElement("mean",text="3.0"))
logNormalElement.append(createElement("sigma",text="2.0"))

logNormal = getDistribution(logNormalElement)

## Should these be checked?
initParams = logNormal.getInitParams()

#check picklling
pk.dump(logNormal,open('testDistrDump.pk','wb'))
plogNormal=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("logNormal",logNormal,{'mu': 3.0, 'sigma': 2.0, 'type': 'LogNormalDistribution', 'low': 0.0})
checkCrowDist("plogNormal",plogNormal,{'mu': 3.0, 'sigma': 2.0, 'type': 'LogNormalDistribution', 'low': 0.0})

checkIntegral("logNormal",logNormal,0.0,10000.0,numpts=1e5,tol=3e-3)

checkAnswer("logNormal cdf(2.0)",logNormal.cdf(2.0),0.124367703363)
checkAnswer("logNormal cdf(1.0)",logNormal.cdf(1.0),0.0668072012689)
checkAnswer("logNormal cdf(3.0)",logNormal.cdf(3.0),0.170879904093)
checkAnswer("logNormal mean",logNormal.untruncatedMean(),148.4131591025766) # m=3;s=2.0;exp(m+s**2/2.0)
checkAnswer("logNormal stddev",logNormal.untruncatedStdDev(),1086.5439790316682) #m=3;s=2.0;sqrt((exp(s**2)-1.0)*(exp(2.0*m+s**2)))
checkAnswer("plogNormal cdf(2.0)",plogNormal.cdf(2.0),0.124367703363)
checkAnswer("plogNormal cdf(1.0)",plogNormal.cdf(1.0),0.0668072012689)
checkAnswer("plogNormal cdf(3.0)",plogNormal.cdf(3.0),0.170879904093)

checkAnswer("logNormal ppf(0.1243677033)",logNormal.ppf(0.124367703363),2.0)
checkAnswer("logNormal ppf(0.1)",logNormal.ppf(0.1),1.54789643258)
checkAnswer("logNormal ppf(0.5)",logNormal.ppf(0.5),20.0855369232)
checkAnswer("plogNormal ppf(0.1243677033)",plogNormal.ppf(0.124367703363),2.0)
checkAnswer("plogNormal ppf(0.1)",plogNormal.ppf(0.1),1.54789643258)
checkAnswer("plogNormal ppf(0.5)",plogNormal.ppf(0.5),20.0855369232)

lowlogNormalElement = ET.Element("LogNormal",{"name":"test"})
lowlogNormalElement.append(createElement("mean",text="3.0"))
lowlogNormalElement.append(createElement("sigma",text="2.0"))
lowlogNormalElement.append(createElement("lowerBound",text="0.0"))
lowlogNormal = getDistribution(lowlogNormalElement)

uplogNormalElement = ET.Element("LogNormal",{"name":"test"})
uplogNormalElement.append(createElement("mean",text="3.0"))
uplogNormalElement.append(createElement("sigma",text="2.0"))
uplogNormalElement.append(createElement("upperBound",text="10.0"))
uplogNormal = getDistribution(uplogNormalElement)

#shift log normal

logNormalElement = ET.Element("LogNormal",{"name":"test"})
logNormalElement.append(createElement("mean",text="3.0"))
logNormalElement.append(createElement("sigma",text="2.0"))
logNormalElement.append(createElement("low",text="10.0"))

logNormal = getDistribution(logNormalElement)

## Should these be checked?
initParams = logNormal.getInitParams()

checkCrowDist("shift logNormal",logNormal,{'mu': 3.0, 'sigma': 2.0, 'type': 'LogNormalDistribution', 'low': 10.0})

checkIntegral("shift logNormal",logNormal,10.0,10010.0,numpts=1e5,tol=3e-3)

checkAnswer("shift logNormal cdf(2.0)",logNormal.cdf(12.0),0.124367703363)
checkAnswer("shift logNormal cdf(1.0)",logNormal.cdf(11.0),0.0668072012689)
checkAnswer("shift logNormal cdf(3.0)",logNormal.cdf(13.0),0.170879904093)

checkAnswer("shift logNormal ppf(0.1243677033)",logNormal.ppf(0.124367703363),12.0)
checkAnswer("shift logNormal ppf(0.1)",logNormal.ppf(0.1),11.54789643258)
checkAnswer("shift logNormal ppf(0.5)",logNormal.ppf(0.5),30.0855369232)

#Test log normal with low mean

logNormalLowMeanElement = ET.Element("LogNormal",{"name":"test"})
logNormalLowMeanElement.append(createElement("mean",text="-0.00002"))
logNormalLowMeanElement.append(createElement("sigma",text="0.2"))

logNormalLowMean = getDistribution(logNormalLowMeanElement)

checkCrowDist("logNormalLowMean",logNormalLowMean,{'mu': -2e-5, 'sigma': 0.2, 'type': 'LogNormalDistribution', 'low': 0.0})

checkIntegral("logNormalLowMean",logNormalLowMean,0.0,1000.0)

checkAnswer("logNormalLowMean cdf(2.0)",logNormalLowMean.cdf(2.0),0.999735707106)
checkAnswer("logNormalLowMean cdf(1.0)",logNormalLowMean.cdf(1.0),0.500039894228)
checkAnswer("logNormalLowMean cdf(3.0)",logNormalLowMean.cdf(3.0),0.999999980238)

checkAnswer("logNormalLowMean ppf(0.500039894228)",logNormalLowMean.ppf(0.500039894228),1.0)
checkAnswer("logNormalLowMean ppf(0.1)",logNormalLowMean.ppf(0.1),0.773886301779)
checkAnswer("logNormalLowMean ppf(0.5)",logNormalLowMean.ppf(0.5),0.9999800002)

#Test Weibull

weibullElement = ET.Element("Weibull",{"name":"test"})
weibullElement.append(createElement("k", text="1.5"))
weibullElement.append(createElement("lambda", text="1.0"))

weibull = getDistribution(weibullElement)

## Should these be checked?
initParams = weibull.getInitParams()

#check picklling
pk.dump(weibull,open('testDistrDump.pk','wb'))
pweibull=pk.load(open('testDistrDump.pk','rb'))

checkCrowDist("weibull",weibull,{'k': 1.5, 'type': 'WeibullDistribution', 'lambda': 1.0, 'low': 0.0})
checkCrowDist("pweibull",pweibull,{'k': 1.5, 'type': 'WeibullDistribution', 'lambda': 1.0, 'low': 0.0})

checkIntegral("weibull",weibull,0.0,100.0)

checkAnswer("weibull cdf(0.5)",weibull.cdf(0.5),0.29781149863)
checkAnswer("weibull cdf(0.2)",weibull.cdf(0.2),0.0855593563928)
checkAnswer("weibull cdf(2.0)",weibull.cdf(2.0),0.940894253438)
checkAnswer("weibull mean",weibull.untruncatedMean(),0.9027452929509336) #l=1.0;k=1.5;l*gamma(1+1/k)
checkAnswer("weibull stddev",weibull.untruncatedStdDev(),0.6129357917546762) #l=1.0;k=1.5;sqrt(l**2*(gamma(1+2/k)-(gamma(1+1/k))**2))
checkAnswer("pweibull cdf(0.5)",pweibull.cdf(0.5),0.29781149863)
checkAnswer("pweibull cdf(0.2)",pweibull.cdf(0.2),0.0855593563928)
checkAnswer("pweibull cdf(2.0)",pweibull.cdf(2.0),0.940894253438)

checkAnswer("weibull ppf(0.29781149863)",weibull.ppf(0.29781149863),0.5)
checkAnswer("weibull ppf(0.1)",weibull.ppf(0.1),0.223075525637)
checkAnswer("weibull ppf(0.9)",weibull.ppf(0.9),1.7437215136)
checkAnswer("pweibull ppf(0.29781149863)",pweibull.ppf(0.29781149863),0.5)
checkAnswer("pweibull ppf(0.1)",pweibull.ppf(0.1),0.223075525637)
checkAnswer("pweibull ppf(0.9)",pweibull.ppf(0.9),1.7437215136)

#shift Weibull

weibullElement = ET.Element("Weibull",{"name":"test"})
weibullElement.append(createElement("k", text="1.5"))
weibullElement.append(createElement("lambda", text="1.0"))
weibullElement.append(createElement("low", text="10.0"))

weibull = getDistribution(weibullElement)

## Should these be checked?
initParams = weibull.getInitParams()

checkCrowDist("shift weibull",weibull,{'k': 1.5, 'type': 'WeibullDistribution', 'lambda': 1.0, 'low':10.0})

checkIntegral("shift weibull",weibull,10.0,110.0)

checkAnswer("shift weibull cdf(0.5)",weibull.cdf(10.5),0.29781149863)
checkAnswer("shift weibull cdf(0.2)",weibull.cdf(10.2),0.0855593563928)
checkAnswer("shift weibull cdf(2.0)",weibull.cdf(12.0),0.940894253438)

checkAnswer("shift weibull ppf(0.29781149863)",weibull.ppf(0.29781149863),10.5)
checkAnswer("shift weibull ppf(0.1)",weibull.ppf(0.1),10.223075525637)
checkAnswer("shift weibull ppf(0.9)",weibull.ppf(0.9),11.7437215136)

lowWeibullElement = ET.Element("Weibull",{"name":"test"})
lowWeibullElement.append(createElement("k", text="1.5"))
lowWeibullElement.append(createElement("lambda", text="1.0"))
lowWeibullElement.append(createElement("lowerBound",text="0.001"))
lowWeibull = getDistribution(lowWeibullElement)

upWeibullElement = ET.Element("Weibull",{"name":"test"})
upWeibullElement.append(createElement("k", text="1.5"))
upWeibullElement.append(createElement("lambda", text="1.0"))
upWeibullElement.append(createElement("upperBound",text="10.0"))
upWeibull = getDistribution(upWeibullElement)

#Testing N-Dimensional Distributions

#InverseWeight
ndInverseWeightElement = ET.Element("NDInverseWeight",{"name":"test"})
ndInverseWeightElement.append(createElement("workingDir", text="ND_test_Grid_cdf/"))
ndInverseWeightElement.append(createElement("p", text="0.5"))
filenode = createElement("dataFilename", text="2DgaussianScatteredPDF.txt")
filenode.set("type","PDF")
ndInverseWeightElement.append(filenode)

ET.dump(ndInverseWeightElement)

ndInverseWeight_test = getDistribution(ndInverseWeightElement)

## Should these be checked?
initParams = ndInverseWeight_test.getInitParams()

checkCrowDist("NDInverseWeight",ndInverseWeight_test,{'type': 'NDInverseWeightDistribution'})

#Cartesian Spline

ndCartesianSplineElement = ET.Element("NDCartesianSpline",{"name":"test"})
filenode = createElement("dataFilename", text="2DgaussianCartesianPDF.txt")
filenode.set("type","PDF")
ndCartesianSplineElement.append(filenode)
ndCartesianSplineElement.append(createElement("workingDir", text="ND_test_Grid_cdf/"))

ndCartesianSpline = getDistribution(ndCartesianSplineElement)

## Should these be checked?
initParams = ndCartesianSpline.getInitParams()

checkCrowDist("NDCartesianSpline",ndCartesianSpline,{'type': 'NDCartesianSplineDistribution'})

#ND MultiVariate Normal

#ndMultiVariateNormal = ET.Element("MultivariateNormal",{"name":"test","method":"spline"})
#munode = createElement("mu", text="10 20")
#ndMultiVariateNormal.append(munode)
#covariancenode = createElement("covariance", text=" 4 0 \n 0 16")
#ndMultiVariateNormal.append(covariancenode)

#ndMultiVariate = getDistribution(ndMultiVariateNormal)

## Should these be checked?
#initParams = ndMultiVariate.getInitParams()

#marginalCDF1 = ndMultiVariate.marginalDistribution(10, 0)
#marginalCDF2 = ndMultiVariate.marginalDistribution(20, 1)

#inverse1 = ndMultiVariate.inverseMarginalDistribution(0.5, 0)
#inverse2 = ndMultiVariate.inverseMarginalDistribution(0.5, 1)

#checkAnswer("MultiVariate marginalDim1(3000)" , marginalCDF1, 0.501, tol=0.01, relative=True)
#checkAnswer("MultiVariate marginalDim2(2500)" , marginalCDF2, 0.501, tol=0.01, relative=True)
#checkAnswer("MultiVariate inverseMarginalDim1(0.5)" , inverse1, 10., tol=0.01, relative=True)
#checkAnswer("MultiVariate inverseMarginalDim2(0.5)" , inverse2, 20., tol=0.01, relative=True)

#Test Categorical

CategoricalElement = ET.Element("Categorical",{"name":"test"})
filenode1=createElement("state", text="0.1")
filenode1.set("outcome","10")
CategoricalElement.append(filenode1)

filenode2=createElement("state", text="0.2")
filenode2.set("outcome","20")
CategoricalElement.append(filenode2)

filenode3=createElement("state", text="0.15")
filenode3.set("outcome","30")
CategoricalElement.append(filenode3)

filenode4=createElement("state", text="0.4")
filenode4.set("outcome","50")
CategoricalElement.append(filenode4)

filenode5=createElement("state", text="0.15")
filenode5.set("outcome","60")
CategoricalElement.append(filenode5)


Categorical = getDistribution(CategoricalElement)

## Should these be checked?
initParams = Categorical.getInitParams()

checkAnswer("Categorical  pdf(10)" , Categorical.pdf(10),0.1)
checkAnswer("Categorical  pdf(30)" , Categorical.pdf(30),0.15)
checkAnswer("Categorical  pdf(60)" , Categorical.pdf(60),0.15)

checkAnswer("Categorical  cdf(10)" , Categorical.cdf(10),0.1)
checkAnswer("Categorical  cdf(30)" , Categorical.cdf(30),0.45)
checkAnswer("Categorical  cdf(60)" , Categorical.cdf(60),1.0)

checkAnswer("Categorical  ppf(0.1)" , Categorical.ppf(0.1),10)
checkAnswer("Categorical  ppf(0.5)" , Categorical.ppf(0.5),50)
checkAnswer("Categorical  ppf(0.9)" , Categorical.ppf(0.9),60)

# Test Custom1D
Custom1DElement = ET.Element("Custom1D",{"name":"test"})
Custom1DElement.append(createElement("dataFilename", text="PointSetFile2_dump.csv"))
Custom1DElement.append(createElement("functionID",   text="pdf"))
Custom1DElement.append(createElement("variableID",   text="x"))
Custom1DElement.append(createElement("functionType", text="pdf"))
Custom1DElement.append(createElement("workingDir",   text="custom1D/"))

Custom1D = getDistribution(Custom1DElement)

checkAnswer("Custom1D pdf(-2.2)",Custom1D.pdf(-2.2),0.0354745928462)
checkAnswer("Custom1D pdf(1.9)",Custom1D.pdf(1.9) ,0.0656158147747)

checkAnswer("Custom1D cdf(-2.2)",Custom1D.cdf(-2.2),0.0139031606668)
checkAnswer("Custom1D cdf(1.9)",Custom1D.cdf(1.9), 0.971283153684)

checkAnswer("Custom1D ppf(0.0139034475135)",Custom1D.ppf(0.0139034475135),-2.19999191499)
checkAnswer("Custom1D ppf(00.971283440184)",Custom1D.ppf(0.971283440184),1.90000436617)

#Test UniformDiscrete

UniformDiscreteElement = ET.Element("UniformDiscrete",{"name":"test"})
UniformDiscreteElement.append(createElement("lowerBound", text="3"))
UniformDiscreteElement.append(createElement("upperBound", text="6"))
UniformDiscreteElement.append(createElement("strategy", text="withoutReplacement"))

UniformDiscrete = getDistribution(UniformDiscreteElement)

## Should these be checked?
initParams = UniformDiscrete.getInitParams()

discardedElems = np.array([5,6])

checkAnswer("UniformDiscrete rvs1",UniformDiscrete.selectedRvs(discardedElems),3)
checkAnswer("UniformDiscrete rvs2",UniformDiscrete.selectedRvs(discardedElems),3)
checkAnswer("UniformDiscrete rvs3",UniformDiscrete.selectedRvs(discardedElems),3)
checkAnswer("UniformDiscrete rvs4",UniformDiscrete.selectedRvs(discardedElems),4)
checkAnswer("UniformDiscrete rvs5",UniformDiscrete.selectedRvs(discardedElems),3)
checkAnswer("UniformDiscrete rvs6",UniformDiscrete.selectedRvs(discardedElems),3)
checkAnswer("UniformDiscrete rvs7",UniformDiscrete.selectedRvs(discardedElems),4)
checkAnswer("UniformDiscrete rvs8",UniformDiscrete.selectedRvs(discardedElems),4)
checkAnswer("UniformDiscrete rvs9",UniformDiscrete.selectedRvs(discardedElems),4)
checkAnswer("UniformDiscrete rvs10",UniformDiscrete.selectedRvs(discardedElems),3)
checkAnswer("UniformDiscrete rvs11",UniformDiscrete.selectedRvs(discardedElems),4)
checkAnswer("UniformDiscrete rvs12",UniformDiscrete.selectedRvs(discardedElems),3)


print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.test_distributions</name>
    <author>cogljj</author>
    <created>2013-12-10</created>
    <classesTested> </classesTested>
    <description>
       This test is a Unit Test for the Distributions classes. It tests all the distributions and all the methods.
    </description>
    <revisions>
      <revision author="cogljj" date="2013-12-10">Adding test of all the rest of the distributions except for binomial.        r23360</revision>
      <revision author="senrs" date="2015-01-26">fixed Bug in Distribution.py    the attribute mean is obsolete use untrMean instead</revision>
      <revision author="senrs" date="2015-01-26">Fixed bugs in the if statements, etc...and included tests for Distributions.py</revision>
      <revision author="talbpaul" date="2015-02-05">added pickle methods and tests for distributionsw</revision>
      <revision author="alfoa" date="2015-02-10">finished caching of data</revision>
      <revision author="talbpaul" date="2015-03-11">added way to do beta through keywords, still need to test stochcoll, but testdistros is passing</revision>
      <revision author="cogljj" date="2015-04-29">Adding test of standard deviation.</revision>
      <revision author="cogljj" date="2015-05-05">Adding check of std deviation of a binomial.</revision>
      <revision author="cogljj" date="2015-05-06">Adding additional checks of the mean and standard deviation.</revision>
      <revision author="alfoa" date="2015-05-18">modified test distribution</revision>
      <revision author="mandd" date="2015-06-16">fixed testDistributions</revision>
      <revision author="alfoa" date="2016-03-31">Closes #478</revision>
      <revision author="maljdan" date="2016-04-12">Improving readability of our own code and removing extraneous functions.</revision>
      <revision author="cogljj" date="2016-04-12">Converting Distributions to use the new input system. All distributions have been converted.</revision>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
      <revision author="alfoa" date="2018-05-10">Added Log Uniform distribution unit test</revision>
    </revisions>
    <requirements>R-RE-1</requirements>
  </TestInfo>
"""
