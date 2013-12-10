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

ET.dump(uniformElement)

uniform = Distributions.Uniform()
uniform.readMoreXML(uniformElement)
uniform.initializeDistribution()
checkAnswer("uniform cdf(1.0)",uniform.cdf(1.0),0.0)
checkAnswer("uniform cdf(2.0)",uniform.cdf(2.0),0.5)
checkAnswer("uniform cdf(3.0)",uniform.cdf(3.0),1.0)

checkAnswer("uniform ppf(0.0)",uniform.ppf(0.0),1.0)
checkAnswer("uniform ppf(0.5)",uniform.ppf(0.5),2.0)
checkAnswer("uniform ppf(1.0)",uniform.ppf(1.0),3.0)

print(uniform.rvs(5))


print(results)

sys.exit(results["fail"])
