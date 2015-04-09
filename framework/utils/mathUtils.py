'''
 This file contains the mathematical methods used in the framework.
 Some of the methods were in the PostProcessor.py
 created on 03/26/2015
 @author: senrs
'''

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import math
from utils import printCsv, printCsvPart
from scipy import interpolate

def normal(x,mu=0.0,sigma=1.0):
  return (1.0/(sigma*math.sqrt(2*math.pi)))*math.exp(-(x - mu)**2/(2.0*sigma**2))

def normalCdf(x,mu=0.0,sigma=1.0):
  return 0.5*(1.0+math.erf((x-mu)/(sigma*math.sqrt(2.0))))

def skewNormal(x,alpha,xi,omega):
  def phi(x):
    return (1.0/math.sqrt(2*math.pi))*math.exp(-(x**2)/2.0)

  def Phi(x):
    return 0.5*(1+math.erf(x/math.sqrt(2)))

  return (2.0/omega)*phi((x-xi)/omega)*Phi(alpha*(x-xi)/omega)

def createInterp(x, y, low_fill, high_fill, kind='linear'):
  interp = interpolate.interp1d(x, y, kind)
  low = x[0]
  high = x[-1]
  def myInterp(x):
    try:
      return interp(x)+0.0
    except ValueError:
      if x <= low:
        return low_fill
      else:
        return high_fill
  return myInterp

def simpson(f, a, b, n):
  h = (b - a) / float(n)
  sum = f(a) + f(b)
  for i in range(1,n, 2):
    sum += 4*f(a + i*h)
  for i in range(2, n-1, 2):
    sum += 2*f(a + i*h)

  return sum * h / 3.0

def printGraphs(csv, functions, f_z_stats = False):
  """prints graphs of the functions.
  The functions are a list of (data_stats_dict, cdf_function, pdf_function,name)
  """

  dataStats = [x[0] for x in functions]
  means = [x["mean"] for x in dataStats]
  stddevs = [x["stdev"] for x in dataStats]
  cdfs = [x[1] for x in functions]
  pdfs = [x[2] for x in functions]
  names = [x[3] for x in functions]
  low = min([m - 3.0*s for m,s in zip(means,stddevs)])
  high = max([m + 3.0*s for m,s in zip(means,stddevs)])
  lowLow = min([m - 5.0*s for m,s in zip(means,stddevs)])
  highHigh = max([m + 5.0*s for m,s in zip(means,stddevs)])
  minBinSize = min([x["min_bin_size"] for x in dataStats])
  print("Graph from ",low,"to",high)
  n = int(math.ceil((high-low)/minBinSize))
  interval = (high - low)/n

  printCsvPart(csv,'"x"')
  for name in names:
    printCsvPart(csv,'"'+name+'_cdf"','"'+name+'_pdf"')
  printCsv(csv)

  for i in range(n):
    x = low+interval*i
    printCsvPart(csv,x)
    for stats, cdf, pdf, name in functions:
      printCsvPart(csv,cdf(x),pdf(x))
    printCsv(csv)

  def fZ(z):
    return simpson(lambda x: pdfs[0](x)*pdfs[1](x-z), lowLow, highHigh, 1000)

  if len(means) < 2:
    return
  midZ = means[0]-means[1]
  lowZ = midZ - 3.0*max(stddevs[0],stddevs[1])
  highZ = midZ + 3.0*max(stddevs[0],stddevs[1])
  printCsv(csv,'"z"','"f_z(z)"')
  zN = 20
  intervalZ = (highZ - lowZ)/zN
  for i in range(zN):
    z = lowZ + intervalZ*i
    printCsv(csv,z,fZ(z))
  cdfAreaDifference = simpson(lambda x:abs(cdfs[1](x)-cdfs[0](x)),lowLow,highHigh,100000)

  def firstMomentSimpson(f, a, b, n):
    return simpson(lambda x:x*f(x), a, b, n)

  pdfCommonArea = simpson(lambda x:min(pdfs[0](x),pdfs[1](x)),
                            lowLow,highHigh,100000)
  for i in range(len(pdfs)):
    pdfArea = simpson(pdfs[i],lowLow,highHigh,100000)
    printCsv(csv,'"pdf_area_'+names[i]+'"',pdfArea)
    dataStats[i]["pdf_area"] = pdfArea
  printCsv(csv,'"cdf_area_difference"',cdfAreaDifference)
  printCsv(csv,'"pdf_common_area"',pdfCommonArea)
  dataStats[0]["cdf_area_difference"] = cdfAreaDifference
  dataStats[0]["pdf_common_area"] = pdfCommonArea
  if f_z_stats:
    sumFunctionDiff = simpson(fZ, lowZ, highZ, 1000)
    firstMomentFunctionDiff = firstMomentSimpson(fZ, lowZ,highZ, 1000)
    varianceFunctionDiff = simpson(lambda x:((x-firstMomentFunctionDiff)**2)*fZ(x),lowZ,highZ, 1000)
    printCsv(csv,'"sum_function_diff"',sumFunctionDiff)
    printCsv(csv,'"first_moment_function_diff"',firstMomentFunctionDiff)
    printCsv(csv,'"variance_function_diff"',varianceFunctionDiff)


def countBins(sortedData, binBoundaries):
  """counts the number of data items in the sorted_data
  Returns an array with the number.  ret[0] is the number of data
  points <= bin_boundaries[0], ret[len(bin_boundaries)] is the number
  of points > bin_boundaries[len(bin_boundaries)-1]
  """
  binIndex = 0
  sortedIndex = 0
  ret = [0]*(len(binBoundaries)+1)
  while sortedIndex < len(sortedData):
    while not binIndex >= len(binBoundaries) and \
          sortedData[sortedIndex] > binBoundaries[binIndex]:
      binIndex += 1
    ret[binIndex] += 1
    sortedIndex += 1
  return ret

def log2(x):
  return math.log(x)/math.log(2.0)



def calculateStats(data):
  """Calculate statistics on a numeric array data
  and return them in a dictionary"""

  sum1 = 0.0
  sum2 = 0.0
  n = len(data)
  for value in data:
    sum1 += value
    sum2 += value**2

  mean = sum1/n
  variance = (1.0/n)*sum2-mean**2
  sampleVariance = (n/(n-1.0))*variance
  stdev = math.sqrt(sampleVariance)

  m4 = 0.0
  m3 = 0.0
  for value in data:
    m3 += (value - mean)**3
    m4 += (value - mean)**4
  m3 = m3/n
  m4 = m4/n
  skewness = m3/(variance**(3.0/2.0))
  kurtosis = m4/variance**2 - 3.0

  ret = {}
  ret["mean"] = mean
  ret["variance"] = variance
  ret["sample_variance"] = sampleVariance
  ret["stdev"] = stdev
  ret["skewness"] = skewness
  ret["kurtosis"] = kurtosis
  return ret
