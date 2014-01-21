#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import sys, os

#Add the module directory to the search path.
ravenDir = os.path.dirname(os.path.dirname(sys.argv[0]))
moduleDir = os.path.join(ravenDir,"control_modules")
print("moduleDir",moduleDir,"ravenDir",ravenDir)
sys.path.append(moduleDir)


import Distributions

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

#Test Uniform

uniformElement = ET.Element("uniform")
uniformElement.append(createElement("low",text="1.0"))
uniformElement.append(createElement("hi",text="3.0"))

#ET.dump(uniformElement)

uniform = Distributions.Uniform()
uniform.readMoreXML(uniformElement)
uniform.initializeDistribution()
checkAnswer("uniform cdf(1.0)",uniform.cdf(1.0),0.0)
checkAnswer("uniform cdf(2.0)",uniform.cdf(2.0),0.5)
checkAnswer("uniform cdf(3.0)",uniform.cdf(3.0),1.0)

checkAnswer("uniform ppf(0.0)",uniform.ppf(0.0),1.0)
checkAnswer("uniform ppf(0.5)",uniform.ppf(0.5),2.0)
checkAnswer("uniform ppf(1.0)",uniform.ppf(1.0),3.0)

print(uniform.rvs(5),uniform.rvs())

#Test Normal

normalElement = ET.Element("normal")
normalElement.append(createElement("mean",text="1.0"))
normalElement.append(createElement("sigma",text="2.0"))

normal = Distributions.Normal()
normal.readMoreXML(normalElement)
normal.initializeDistribution()

checkAnswer("normal cdf(0.0)",normal.cdf(0.0),0.308537538726)
checkAnswer("normal cdf(1.0)",normal.cdf(1.0),0.5)
checkAnswer("normal cdf(2.0)",normal.cdf(2.0),0.691462461274)

checkAnswer("normal ppf(0.1)",normal.ppf(0.1),-1.56310313109)
checkAnswer("normal ppf(0.5)",normal.ppf(0.5),1.0)
checkAnswer("normal ppf(0.9)",normal.ppf(0.9),3.56310313109)

checkAnswer("normal mean()",normal.untruncatedMean(),1.0)
checkAnswer("normal median()",normal.untruncatedMedian(),1.0)
checkAnswer("normal mode()",normal.untruncatedMode(),1.0)



print(normal.rvs(5),normal.rvs())

#Test Truncated Normal 

truncNormalElement = ET.Element("truncnorm")
truncNormalElement.append(createElement("mean",text="1.0"))
truncNormalElement.append(createElement("sigma",text="2.0"))
truncNormalElement.append(createElement("lowerBound",text="-1.0"))
truncNormalElement.append(createElement("upperBound",text="3.0"))

truncNormal = Distributions.Normal()
truncNormal.readMoreXML(truncNormalElement)
truncNormal.initializeDistribution()

checkAnswer("truncNormal cdf(0.0)",truncNormal.cdf(0.0),0.219546787406)
checkAnswer("truncNormal cdf(1.0)",truncNormal.cdf(1.0),0.5)
checkAnswer("truncNormal cdf(2.0)",truncNormal.cdf(2.0),0.780453212594)

checkAnswer("truncNormal ppf(0.1)",truncNormal.ppf(0.1),-0.498029197939)
checkAnswer("truncNormal ppf(0.5)",truncNormal.ppf(0.5),1.0)
checkAnswer("truncNormal ppf(0.9)",truncNormal.ppf(0.9),2.49802919794)

#Test Gamma

gammaElement = ET.Element("gamma")
gammaElement.append(createElement("low",text="0.0"))
gammaElement.append(createElement("alpha",text="1.0"))
gammaElement.append(createElement("beta",text="0.5"))

gamma = Distributions.Gamma()
gamma.readMoreXML(gammaElement)
gamma.initializeDistribution()

checkAnswer("gamma cdf(0.0)",gamma.cdf(0.0),0.0)
checkAnswer("gamma cdf(1.0)",gamma.cdf(1.0),0.393469340287)
checkAnswer("gamma cdf(10.0)",gamma.cdf(10.0),0.993262053001)

checkAnswer("gamma ppf(0.1)",gamma.ppf(0.1),0.210721031316)
checkAnswer("gamma ppf(0.5)",gamma.ppf(0.5),1.38629436112)
checkAnswer("gamma ppf(0.9)",gamma.ppf(0.9),4.60517018599)

#print(gamma.rvs(5),gamma.rvs())

#Test Beta

betaElement = ET.Element("beta")
betaElement.append(createElement("low",text="0.0"))
betaElement.append(createElement("hi",text="1.0"))
betaElement.append(createElement("alpha",text="5.0"))
betaElement.append(createElement("beta",text="1.0"))

beta = Distributions.Beta()
beta.readMoreXML(betaElement)
beta.initializeDistribution()

checkAnswer("beta cdf(0.1)",beta.cdf(0.1),1e-05)
checkAnswer("beta cdf(0.5)",beta.cdf(0.5),0.03125)
checkAnswer("beta cdf(0.9)",beta.cdf(0.9),0.59049)

checkAnswer("beta ppf(0.1)",beta.ppf(0.1),0.63095734448)
checkAnswer("beta ppf(0.5)",beta.ppf(0.5),0.870550563296)
checkAnswer("beta ppf(0.9)",beta.ppf(0.9),0.979148362361)

print(beta.rvs(5),beta.rvs())

#Test Beta Scaled

betaElement = ET.Element("beta")
betaElement.append(createElement("low",text="0.0"))
betaElement.append(createElement("hi",text="4.0"))
betaElement.append(createElement("alpha",text="5.0"))
betaElement.append(createElement("beta",text="1.0"))

beta = Distributions.Beta()
beta.readMoreXML(betaElement)
beta.initializeDistribution()

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
triangular.readMoreXML(triangularElement)
triangular.initializeDistribution()

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
poisson.readMoreXML(poissonElement)
poisson.initializeDistribution()

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
binomial.readMoreXML(binomialElement)
binomial.initializeDistribution()

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
bernoulli.readMoreXML(bernoulliElement)
bernoulli.initializeDistribution()

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
logistic.readMoreXML(logisticElement)
logistic.initializeDistribution()

checkAnswer("logistic cdf(0)",logistic.cdf(0.0),0.0179862099621)
checkAnswer("logistic cdf(4)",logistic.cdf(4.0),0.5)
checkAnswer("logistic cdf(8)",logistic.cdf(8.0),0.982013790038)

checkAnswer("logistic ppf(0.25)",logistic.ppf(0.25),2.90138771133)
checkAnswer("logistic ppf(0.50)",logistic.ppf(0.50),4.0)
checkAnswer("logistic ppf(0.75)",logistic.ppf(0.75),5.09861228867)

print(results)

sys.exit(results["fail"])
