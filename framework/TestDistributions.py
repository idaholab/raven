#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import sys, os
frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(frameworkDir,'utils'))

from utils import find_crow

find_crow(os.path.dirname(os.path.abspath(sys.argv[0])))


import Distributions

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

def checkCrowDist(comment,dist,expected_crow_dist):
  crow_dist = dist.getCrowDistDict()
  if crow_dist != expected_crow_dist:
    print(comment,crow_dist,expected_crow_dist)
    results["fail"] += 1
  else:
    results["pass"] += 1

#Test module methods
print(Distributions.knownTypes())
#Test error
try: Distributions.returnInstance("unknown")
except: print("error worked")

#Test Uniform

uniformElement = ET.Element("uniform")
uniformElement.append(createElement("low",text="1.0"))
uniformElement.append(createElement("hi",text="3.0"))

#ET.dump(uniformElement)

uniform = Distributions.Uniform()
uniform._readMoreXML(uniformElement)
uniform.initializeDistribution()

checkCrowDist("uniform",uniform,{'xMin': 1.0, 'type': 'UniformDistribution', 'xMax': 3.0})

checkAnswer("uniform cdf(1.0)",uniform.cdf(1.0),0.0)
checkAnswer("uniform cdf(2.0)",uniform.cdf(2.0),0.5)
checkAnswer("uniform cdf(3.0)",uniform.cdf(3.0),1.0)

checkAnswer("uniform ppf(0.0)",uniform.ppf(0.0),1.0)
checkAnswer("uniform ppf(0.5)",uniform.ppf(0.5),2.0)
checkAnswer("uniform ppf(1.0)",uniform.ppf(1.0),3.0)

print(uniform.rvs(5),uniform.rvs())

#check rvsWithinCDFbounds
uniform.rvsWithinbounds(1.5,2.5)
# fake quadrature
uniform.setQuad({},2)
uniform.quad()
uniform.polyOrder()

#uniform.norm(2)
#uniform.standardToActualPoint(-1)
#uniform.actualToStandardPoint(2.0)
#uniform.standardToActualWeight(0.5)
#uniform.probNorm(0.5)
uniform.addInitParams({})
for _ in range(10): Distributions.randomIntegers(0,1)

uniform.poly_norm(2)
uniform.actual_point(-1)
uniform.std_point(2.0)
uniform.actual_weight(0.5)
uniform.probability_norm(0.5)

uniform.addInitParams({})
for _ in range(10): Distributions.randomIntegers(0,1)

Distributions.randomIntegers(2,1)

#Test Normal

normalElement = ET.Element("normal")
normalElement.append(createElement("mean",text="1.0"))
normalElement.append(createElement("sigma",text="2.0"))

normal = Distributions.Normal()
normal._readMoreXML(normalElement)
normal.initializeDistribution()

checkCrowDist("normal",normal,{'mu': 1.0, 'sigma': 2.0, 'type': 'NormalDistribution'})

checkAnswer("normal cdf(0.0)",normal.cdf(0.0),0.308537538726)
checkAnswer("normal cdf(1.0)",normal.cdf(1.0),0.5)
checkAnswer("normal cdf(2.0)",normal.cdf(2.0),0.691462461274)

checkAnswer("normal ppf(0.1)",normal.ppf(0.1),-1.56310313109)
checkAnswer("normal ppf(0.5)",normal.ppf(0.5),1.0)
checkAnswer("normal ppf(0.9)",normal.ppf(0.9),3.56310313109)

checkAnswer("normal mean()",normal.untruncatedMean(),1.0)
checkAnswer("normal median()",normal.untruncatedMedian(),1.0)
checkAnswer("normal mode()",normal.untruncatedMode(),1.0)

normal.poly_norm(2)
normal.actual_point(-1)
normal.std_point(2.0)
normal.actual_weight(0.5)
normal.probability_norm(0.5)


print(normal.rvs(5),normal.rvs())

#Test Truncated Normal

truncNormalElement = ET.Element("truncnorm")
truncNormalElement.append(createElement("mean",text="1.0"))
truncNormalElement.append(createElement("sigma",text="2.0"))
truncNormalElement.append(createElement("lowerBound",text="-1.0"))
truncNormalElement.append(createElement("upperBound",text="3.0"))

truncNormal = Distributions.Normal()
truncNormal._readMoreXML(truncNormalElement)
truncNormal.initializeDistribution()

checkCrowDist("truncNormal",truncNormal,{'xMin': -1.0, 'mu': 1.0, 'type': 'NormalDistribution', 'sigma': 2.0, 'xMax': 3.0})

checkAnswer("truncNormal cdf(0.0)",truncNormal.cdf(0.0),0.219546787406)
checkAnswer("truncNormal cdf(1.0)",truncNormal.cdf(1.0),0.5)
checkAnswer("truncNormal cdf(2.0)",truncNormal.cdf(2.0),0.780453212594)

checkAnswer("truncNormal ppf(0.1)",truncNormal.ppf(0.1),-0.498029197939)
checkAnswer("truncNormal ppf(0.5)",truncNormal.ppf(0.5),1.0)
checkAnswer("truncNormal ppf(0.9)",truncNormal.ppf(0.9),2.49802919794)

lowtruncNormalElement = ET.Element("lowtruncnorm")
lowtruncNormalElement.append(createElement("mean",text="1.0"))
lowtruncNormalElement.append(createElement("sigma",text="2.0"))
lowtruncNormalElement.append(createElement("lowerBound",text="-1.0"))
lowtruncNormal = Distributions.Normal()
lowtruncNormal._readMoreXML(lowtruncNormalElement)
lowtruncNormal.initializeDistribution()

uptruncNormalElement = ET.Element("uptruncnorm")
uptruncNormalElement.append(createElement("mean",text="1.0"))
uptruncNormalElement.append(createElement("sigma",text="2.0"))
uptruncNormalElement.append(createElement("upperBound",text="3.0"))
uptruncNormal = Distributions.Normal()
uptruncNormal._readMoreXML(uptruncNormalElement)
uptruncNormal.initializeDistribution()
#Test Gamma

gammaElement = ET.Element("gamma")
gammaElement.append(createElement("low",text="0.0"))
gammaElement.append(createElement("alpha",text="1.0"))
gammaElement.append(createElement("beta",text="0.5"))

gamma = Distributions.Gamma()
gamma._readMoreXML(gammaElement)
gamma.initializeDistribution()

gamma.addInitParams({})

checkCrowDist("gamma",gamma,{'xMin': 0.0, 'theta': 2.0, 'k': 1.0, 'type': 'GammaDistribution', 'low': 0.0})

checkAnswer("gamma cdf(0.0)",gamma.cdf(0.0),0.0)
checkAnswer("gamma cdf(1.0)",gamma.cdf(1.0),0.393469340287)
checkAnswer("gamma cdf(10.0)",gamma.cdf(10.0),0.993262053001)

checkAnswer("gamma ppf(0.1)",gamma.ppf(0.1),0.210721031316)
checkAnswer("gamma ppf(0.5)",gamma.ppf(0.5),1.38629436112)
checkAnswer("gamma ppf(0.9)",gamma.ppf(0.9),4.60517018599)

gamma.poly_norm(2)
gamma.actual_point(-1)
gamma.std_point(2.0)
gamma.actual_weight(0.5)
gamma.probability_norm(0.5)

nobeta_gammaElement = ET.Element("nobeta_gamma")
nobeta_gammaElement.append(createElement("alpha",text="1.0"))
nobeta_gammaElement.append(createElement("low",text="0.0"))
nobeta_gammaElement.append(createElement("upperBound",text="10.0"))
nobeta_gamma = Distributions.Gamma()
nobeta_gamma._readMoreXML(nobeta_gammaElement)
nobeta_gamma.initializeDistribution()

#print(gamma.rvs(5),gamma.rvs())

#Test Beta

betaElement = ET.Element("beta")
betaElement.append(createElement("low",text="0.0"))
betaElement.append(createElement("hi",text="1.0"))
betaElement.append(createElement("alpha",text="5.0"))
betaElement.append(createElement("beta",text="2.0"))

beta = Distributions.Beta()
beta._readMoreXML(betaElement)
beta.initializeDistribution()

beta.addInitParams({})

checkCrowDist("beta",beta,{'scale': 1.0, 'beta': 2.0, 'xMax': 1.0, 'xMin': 0.0, 'alpha': 5.0, 'type': 'BetaDistribution'})

checkAnswer("beta cdf(0.1)",beta.cdf(0.1),5.5e-05)
checkAnswer("beta cdf(0.5)",beta.cdf(0.5),0.109375)
checkAnswer("beta cdf(0.9)",beta.cdf(0.9),0.885735)

checkAnswer("beta ppf(0.1)",beta.ppf(0.1),0.489683693449)
checkAnswer("beta ppf(0.5)",beta.ppf(0.5),0.735550016704)
checkAnswer("beta ppf(0.9)",beta.ppf(0.9),0.907404741087)

checkAnswer("beta mean()",beta.untruncatedMean(),5.0/(5.0+2.0))
checkAnswer("beta median()",beta.untruncatedMedian(),0.735550016704)
checkAnswer("beta mode()",beta.untruncatedMode(),(5.0-1)/(5.0+2.0-2))

checkAnswer("beta pdf(0.25)",beta.pdf(0.25),0.087890625)
checkAnswer("beta cdfComplement(0.25)",beta.untruncatedCdfComplement(0.25),0.995361328125)
checkAnswer("beta hazard(0.25)",beta.untruncatedHazard(0.25),0.0883002207506)

print(beta.rvs(5),beta.rvs())

#Test Beta Scaled

betaElement = ET.Element("beta")
betaElement.append(createElement("low",text="0.0"))
betaElement.append(createElement("high",text="4.0"))
betaElement.append(createElement("alpha",text="5.0"))
betaElement.append(createElement("beta",text="1.0"))

beta = Distributions.Beta()
beta._readMoreXML(betaElement)
beta.initializeDistribution()

checkCrowDist("scaled beta",beta,{'scale': 4.0, 'beta': 1.0, 'xMax': 4.0, 'xMin': 0.0, 'alpha': 5.0, 'type': 'BetaDistribution'})

checkAnswer("scaled beta cdf(0.1)",beta.cdf(0.1),9.765625e-09)
checkAnswer("scaled beta cdf(0.5)",beta.cdf(0.5),3.0517578125e-05)
checkAnswer("scaled beta cdf(0.9)",beta.cdf(0.9),0.000576650390625)

checkAnswer("scaled beta ppf(0.1)",beta.ppf(0.1),2.52382937792)
checkAnswer("scaled beta ppf(0.5)",beta.ppf(0.5),3.48220225318)
checkAnswer("scaled beta ppf(0.9)",beta.ppf(0.9),3.91659344944)

print(beta.rvs(5),beta.rvs())

#Test Triangular

triangularElement = ET.Element("triangular")
triangularElement.append(createElement("min",text="0.0"))
triangularElement.append(createElement("apex",text="3.0"))
triangularElement.append(createElement("max",text="4.0"))

triangular = Distributions.Triangular()
triangular._readMoreXML(triangularElement)
triangular.initializeDistribution()

checkCrowDist("triangular",triangular,{'lowerBound': 0.0, 'type': 'TriangularDistribution', 'upperBound': 4.0, 'xMax': 4.0, 'xMin': 0.0, 'xPeak': 3.0})

checkAnswer("triangular cdf(0.25)",triangular.cdf(0.25),0.00520833333333)
checkAnswer("triangular cdf(3.0)",triangular.cdf(3.0),0.75)
checkAnswer("triangular cdf(3.5)",triangular.cdf(3.5),0.9375)

checkAnswer("triangular ppf(0.1)",triangular.ppf(0.1),1.09544511501)
checkAnswer("triangular ppf(0.5)",triangular.ppf(0.5),2.44948974278)
checkAnswer("triangular ppf(0.9)",triangular.ppf(0.9),3.36754446797)

print(triangular.rvs(5),triangular.rvs())

#Test Poisson

poissonElement = ET.Element("poisson")
poissonElement.append(createElement("mu",text="4.0"))

poisson = Distributions.Poisson()
poisson._readMoreXML(poissonElement)
poisson.initializeDistribution()

poisson.addInitParams({})

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

binomial.addInitParams({})

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

bernoulli.addInitParams({})

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

logistic.addInitParams({})

checkCrowDist("logistic",logistic,{'scale': 1.0, 'type': 'LogisticDistribution', 'location': 4.0})

checkAnswer("logistic cdf(0)",logistic.cdf(0.0),0.0179862099621)
checkAnswer("logistic cdf(4)",logistic.cdf(4.0),0.5)
checkAnswer("logistic cdf(8)",logistic.cdf(8.0),0.982013790038)

checkAnswer("logistic ppf(0.25)",logistic.ppf(0.25),2.90138771133)
checkAnswer("logistic ppf(0.50)",logistic.ppf(0.50),4.0)
checkAnswer("logistic ppf(0.75)",logistic.ppf(0.75),5.09861228867)

lowLogisticElement = ET.Element("lowlogistic")
lowLogisticElement.append(createElement("location",text="4.0"))
lowLogisticElement.append(createElement("scale",text="1.0"))
lowLogisticElement.append(createElement("lowerBound",text="3.0"))
lowLogistic = Distributions.Logistic()
lowLogistic._readMoreXML(lowLogisticElement)
lowLogistic.initializeDistribution()

upLogisticElement = ET.Element("uplogistic")
upLogisticElement.append(createElement("location",text="4.0"))
upLogisticElement.append(createElement("scale",text="1.0"))
upLogisticElement.append(createElement("upperBound",text="5.0"))
upLogistic = Distributions.Logistic()
upLogistic._readMoreXML(upLogisticElement)
upLogistic.initializeDistribution()
#Test Exponential

exponentialElement = ET.Element("exponential")
exponentialElement.append(createElement("lambda",text="5.0"))

exponential = Distributions.Exponential()
exponential._readMoreXML(exponentialElement)
exponential.initializeDistribution()

exponential.addInitParams({})

checkCrowDist("exponential",exponential,{'type': 'ExponentialDistribution', 'lambda': 5.0})

checkAnswer("exponential cdf(0.3)",exponential.cdf(0.3),0.7768698399)
checkAnswer("exponential cdf(1.0)",exponential.cdf(1.0),0.993262053001)
checkAnswer("exponential cdf(3.0)",exponential.cdf(3.0),0.999999694098)

checkAnswer("exponential ppf(0.7768698399)",exponential.ppf(0.7768698399),0.3)
checkAnswer("exponential ppf(0.2)",exponential.ppf(0.2),0.0446287102628)
checkAnswer("exponential ppf(0.5)",exponential.ppf(0.5),0.138629436112)

lowExponentialElement = ET.Element("lowExponential")
lowExponentialElement.append(createElement("lambda",text="5.0"))
lowExponentialElement.append(createElement("lowerBound",text="0.0"))
lowExponential = Distributions.Exponential()
lowExponential._readMoreXML(lowExponentialElement)
lowExponential.initializeDistribution()
upExponentialElement = ET.Element("upExponential")
upExponentialElement.append(createElement("lambda",text="5.0"))
upExponentialElement.append(createElement("upperBound",text="10.0"))
upExponential = Distributions.Exponential()
upExponential._readMoreXML(upExponentialElement)
upExponential.initializeDistribution()
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

logNormal.addInitParams({})

checkCrowDist("logNormal",logNormal,{'mu': 3.0, 'sigma': 2.0, 'type': 'LogNormalDistribution'})

checkAnswer("logNormal cdf(2.0)",logNormal.cdf(2.0),0.124367703363)
checkAnswer("logNormal cdf(1.0)",logNormal.cdf(1.0),0.0668072012689)
checkAnswer("logNormal cdf(3.0)",logNormal.cdf(3.0),0.170879904093)

checkAnswer("logNormal ppf(0.1243677033)",logNormal.ppf(0.124367703363),2.0)
checkAnswer("logNormal ppf(0.1)",logNormal.ppf(0.1),1.54789643258)
checkAnswer("logNormal ppf(0.5)",logNormal.ppf(0.5),20.0855369232)

lowlogNormalElement = ET.Element("lowlogNormal")
lowlogNormalElement.append(createElement("mean",text="3.0"))
lowlogNormalElement.append(createElement("sigma",text="2.0"))
lowlogNormalElement.append(createElement("lowerBound",text="0.0"))
lowlogNormal = Distributions.LogNormal()
lowlogNormal._readMoreXML(lowlogNormalElement)
lowlogNormal.initializeDistribution()

uplogNormalElement = ET.Element("uplogNormal")
uplogNormalElement.append(createElement("mean",text="3.0"))
uplogNormalElement.append(createElement("sigma",text="2.0"))
uplogNormalElement.append(createElement("upperBound",text="10.0"))
uplogNormal = Distributions.LogNormal()
uplogNormal._readMoreXML(uplogNormalElement)
uplogNormal.initializeDistribution()

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

weibull.addInitParams({})

checkCrowDist("weibull",weibull,{'k': 1.5, 'type': 'WeibullDistribution', 'lambda': 1.0})

checkAnswer("weibull cdf(0.5)",weibull.cdf(0.5),0.29781149863)
checkAnswer("weibull cdf(0.2)",weibull.cdf(0.2),0.0855593563928)
checkAnswer("weibull cdf(2.0)",weibull.cdf(2.0),0.940894253438)

checkAnswer("weibull ppf(0.29781149863)",weibull.ppf(0.29781149863),0.5)
checkAnswer("weibull ppf(0.1)",weibull.ppf(0.1),0.223075525637)
checkAnswer("weibull ppf(0.9)",weibull.ppf(0.9),1.7437215136)

lowWeibullElement = ET.Element("lowweibull")
lowWeibullElement.append(createElement("k", text="1.5"))
lowWeibullElement.append(createElement("lambda", text="1.0"))
lowWeibullElement.append(createElement("lowerBound",text="0.001"))
lowWeibull = Distributions.Weibull()
lowWeibull._readMoreXML(lowWeibullElement)
lowWeibull.initializeDistribution()

upWeibullElement = ET.Element("upweibull")
upWeibullElement.append(createElement("k", text="1.5"))
upWeibullElement.append(createElement("lambda", text="1.0"))
upWeibullElement.append(createElement("upperBound",text="10.0"))
upWeibull = Distributions.Weibull()
upWeibull._readMoreXML(upWeibullElement)
upWeibull.initializeDistribution()

#Testing N-Dimensional Distributions

#InverseWeight

ndInverseWeightElement = ET.Element("NDInverseWeight")
ndInverseWeightElement.append(createElement("data_filename", text="ND_data.dat"))
ndInverseWeightElement.append(createElement("p", text="0.5"))

ndInverseWeight = Distributions.NDInverseWeight()
ndInverseWeight._readMoreXML(ndInverseWeightElement)
ndInverseWeight.initializeDistribution()

ndInverseWeight.addInitParams({})

checkCrowDist("NDInverseWeight",ndInverseWeight,{'type': 'NDInverseWeightDistribution'})

#Scattered MS

ndScatteredMSElement = ET.Element("NDScatteredMS")
ndScatteredMSElement.append(createElement("data_filename", text="ND_data.dat"))
ndScatteredMSElement.append(createElement("precision", text="1"))
ndScatteredMSElement.append(createElement("p", text="0.5"))

ndScatteredMS = Distributions.NDScatteredMS()
ndScatteredMS._readMoreXML(ndScatteredMSElement)
ndScatteredMS.initializeDistribution()

ndScatteredMS.addInitParams({})

checkCrowDist("NDScatteredMS",ndScatteredMS,{'type': 'NDScatteredMSDistribution'})

#Cartesian Spline

ndCartesianSplineElement = ET.Element("NDCartesianSpline")
ndCartesianSplineElement.append(createElement("data_filename", text="ND_data.dat"))

ndCartesianSpline = Distributions.NDCartesianSpline()
ndCartesianSpline._readMoreXML(ndCartesianSplineElement)
ndCartesianSpline.initializeDistribution()

ndCartesianSpline.addInitParams({})

checkCrowDist("NDCartesianSpline",ndCartesianSpline,{'type': 'NDCartesianSplineDistribution'})



print(results)

sys.exit(results["fail"])
