import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde,rv_histogram
from scipy.special import comb
from sklearn.cluster import KMeans


peakStates=[]

import scipy.linalg as LA
import math


def baseline(y, deg=None, max_it=None, tol=None):
    """
    Computes the baseline of a given data.

    Iteratively performs a polynomial fitting in the data to detect its
    baseline. At every iteration, the fitting weights on the regions with
    peaks are reduced to identify the baseline only.

    Parameters
    ----------
    y : ndarray
        Data to detect the baseline.
    deg : int (default: 3)
        Degree of the polynomial that will estimate the data baseline. A low
        degree may fail to detect all the baseline present, while a high
        degree may make the data too oscillatory, especially at the edges.
    max_it : int (default: 100)
        Maximum number of iterations to perform.
    tol : float (default: 1e-3)
        Tolerance to use when comparing the difference between the current
        fit coefficients and the ones from the last iteration. The iteration
        procedure will stop when the difference between them is lower than
        *tol*.

    Returns
    -------
    ndarray
        Array with the baseline amplitude for every original point in *y*
    """
    # for not repeating ourselves in `envelope`
    if deg is None: deg = 3
    if max_it is None: max_it = 100
    if tol is None: tol = 1e-3

    order = deg + 1
    coeffs = np.ones(order)

    # try to avoid numerical issues
    cond = math.pow(abs(y).max(), 1. / order)
    x = np.linspace(0., cond, y.size)
    base = y.copy()

    vander = np.vander(x, order)
    vander_pinv = LA.pinv2(vander)

    for _ in range(max_it):
        coeffs_new = np.dot(vander_pinv, y)

        if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
            break

        coeffs = coeffs_new
        base = np.dot(vander, coeffs)
        y = np.minimum(y, base)

    return base

def envelope(y, deg=None, max_it=None, tol=None):
    """
    Computes the upper envelope of a given data.
    It is implemented in terms of the `baseline` function.

    Parameters
    ----------
    y : ndarray
        Data to detect the baseline.
    deg : int
        Degree of the polynomial that will estimate the envelope.
    max_it : int
        Maximum number of iterations to perform.
    tol : float
        Tolerance to use when comparing the difference between the current
        fit coefficients and the ones from the last iteration.

    Returns
    -------
    ndarray
        Array with the envelope amplitude for every original point in *y*
    """
    return y.max() - baseline(y.max() - y, deg, max_it, tol)



def reject_outliers(data, m=2):
  return data[abs(data - np.mean(data)) < m * np.std(data)]

def _setExpAmpTypes(expPeakTypes):
  expAmpTypes=expPeakTypes+int(comb(expPeakTypes,2))
  return expAmpTypes
# gather the signal into residual and peaks
def _classifyPeaksSignal(signal,low):
  maskPeak=   signal>low
  maskRes =   signal<=low
  peakSig =   signal[maskPeak]
  resSig  =   signal[maskRes]
  return peakSig,resSig

#collect the index of residual and peaks
def _classifyPeaksInd(signal,low):
  peakIndex   = np.where(signal>low)[0]
  resIndex  = np.where(signal<=low)[0]
  return peakIndex,resIndex

## index of begining and Ending in peak
def _collectPeakRange(peakIndex):
  Blocallist = [peakIndex[0]]
  Elocallist = []
  Wlocallist=[]
  for i in range(len(peakIndex)-1):
    if peakIndex[i+1]-peakIndex[i]!=1:
      Blocallist.append(peakIndex[i+1])
      Elocallist.append(peakIndex[i])
  Elocallist.append(peakIndex[-1])
  #print(Blocallist)
  #print(Elocallist)
  return Blocallist, Elocallist

## peak width
def _collectPeakWidth(Blocallist,Elocallist):
  Wlocallist=np.array(Elocallist)-np.array(Blocallist)+np.ones(len(Elocallist))
  return Wlocallist

## peak picker
def _peakPicker(signal,low,distance=None):
  peaks_1, properties_1 = find_peaks(signal, height=low,distance=distance)
  heights = properties_1['peak_heights']
  #print(peaks_1)
  #print(heights)
  return peaks_1,heights




## divide the peaks by amp
def _generateAmpRange(signal,low,expPeakTypes,expAmpTypes,heights):
  ## define range of amplitude
  ampRange = [low,np.max(signal)]
  if expPeakTypes > 1:
    ## cluster the amplitude in  groups
    kmeans = KMeans(n_clusters=expAmpTypes).fit(np.reshape(heights,(-1, 1)))
    #print(kmeans.labels_)
    #print(kmeans.cluster_centers_)
    sCluster = np.sort(kmeans.cluster_centers_,axis=None)
    for i in range(expAmpTypes-1):
      sep = 0.5*(sCluster[i]+sCluster[i+1])
      ampRange.insert(-1,sep)
    #print(ampRange)
    ## group into 3 group
  return ampRange

## group each type of peaks
def _groupByAmplitude(expAmpTypes,peaks_1,heights,ampRange,Blocallist,Elocallist,Wlocallist):
  groupsAmp = []
  for i in range(expAmpTypes): #
    peakInfo   = {}
    indLocal   = []
    ampLocal   = []
    rangeLocal = [] #first index of range is aways one
    for j in range(len(heights)):
      if heights[j]>ampRange[i] and heights[j]<=ampRange[i+1]:
        indLocal.append(peaks_1[j])
        ampLocal.append(heights[j])
        if len(rangeLocal)>0:
          for k in range(rangeLocal[-1],min(len(Wlocallist),j+1)):
            if peaks_1[j]>=Blocallist[k] and peaks_1[j]<=Elocallist[k]:
              rangeLocal.append(k)
              break
        else:
          for k in range(min(len(Wlocallist),j+1)):
            if peaks_1[j]>=Blocallist[k] and peaks_1[j]<=Elocallist[k]:
              rangeLocal.append(k)
              break
    peakInfo['Ind'] = indLocal
    peakInfo['Amp'] = ampLocal
    peakInfo['rId'] = rangeLocal
    groupsAmp.append(peakInfo)
      #print(groupsAmp)
      #print(groupsAmp[0]['rId'])
      #print(len(groupsAmp))
      #print(set(groupsAmp[-1]['rId']).union(set(groupsAmp[1]['rId']).intersection(set(groupsAmp[2]['rId']))))
  return groupsAmp

## collect SAPs
def _collectWidth(expPeakTypes,groupsAmp,Wlocallist):
  widthPeak = []
  widthHistDist=[]
  if expPeakTypes > 1:
    ##FIXME this dupRID is not so dope, only works for 2 groups
    dupRID=set(groupsAmp[-1]['rId']).union(set(groupsAmp[1]['rId']).intersection(set(groupsAmp[expPeakTypes]['rId'])))
    for i in range(expPeakTypes):
      wid  = []
      rIds = groupsAmp[i]['rId']
      rIds = list(set(rIds).difference(dupRID))
      for j in range(len(rIds)):
        wid.append(Wlocallist[rIds[j]])
      wid  = reject_outliers(np.asarray(wid))
      widthPeak.append(wid.tolist())
  else:
    widthPeak=Wlocallist
  for i in range(len(widthPeak)):
    hist=np.histogram(widthPeak[i])
    widthHistDist.append(rv_histogram(hist))
  return widthHistDist

## collect periods
# FIXME too many assumptions in this step
def _collectPeriod(expPeakTypes,groupsAmp,peaks_1):
  periodPeak = []
  periodHistDist=[]
  if expPeakTypes>1:
    for i in range(expPeakTypes):
      prd  = []
      Inds = groupsAmp[i]['Ind']+groupsAmp[-1]['Ind'] ## FIX this step in future
      Inds = np.sort(Inds)
      prd  = np.diff(Inds)
      prd  = reject_outliers(prd)   ###reject some outliers
      periodPeak.append(prd.tolist())
      #print(periodPeak[0])
  else:
    periodPeak = np.diff(peaks_1)
  for i in range(len(periodPeak)):
    hist=np.histogram(periodPeak[i])
    periodHistDist.append(rv_histogram(hist))
  return periodHistDist

## collect amps
def _collectAmp(expPeakTypes,groupsAmp,heights):
  ampPeak =[]
  ampHistDist=[]
  if expPeakTypes>1:
    for i in range(expPeakTypes):
      amp=groupsAmp[i]['Amp']
      ampPeak.append(amp)
  else:
    ampPeak=heights
  for i in range(len(ampPeak)):
    hist=np.histogram(ampPeak[i])
    ampHistDist.append(rv_histogram(hist))
  return ampHistDist








"""


  for i in range(len(ampPeak)):
    hist = np.histogram(data)
    hist_dist = rv_histogram(hist)

for i in range(len(Blocallist)):
    BInd=Blocallist[i]
    Eind=Elocallist[i]
    if Eind==BInd:
        peaks.append(BInd)
        heights.append(signal[BInd]-low)
        width.append(1)
    else:
        peaks.append(BInd+np.argmax(signal[BInd:Eind]))
        heights.append(max(signal[BInd:Eind]))
        width.append(Eind-BInd+1)

range=[low,max(signal)]
for i in range(len(sCluster)-1):
    sep=0.5*(sCluster[i]+sCluster[i+1])


print(range)
peaks, properties= find_peaks(signal, height=low,prominence=0.2)
peaks_1, properties_1= find_peaks(signal, height=low)

print(np.diff(peaks))

#x=np.linspace(0, len(peaksig)-1, len(peaksig))
#fig, ax = plt.subplots(figsize=(12, 10))
#ax.plot(x,peaksig)
#plt.show()

plt.plot(signal)
plt.plot(peaks, signal[peaks], "o")
plt.plot(peaks_1, signal[peaks_1], "x")

plt.plot(np.ones_like(signal)*low, "--", color="gray")
plt.show()
print(properties)
#print(type(signal[peaks]))

x       =   signal[peaks]
x_d     =   np.linspace(min(x), max(x), 100)

fig, ax = plt.subplots()
ax.plot(x_d, kde_scipy(x, x_d, bandwidth=0.1))
plt.show()
print(peakIndex)
print(len(peaks))
print(len(peakIndex))
plt.hist(properties_1['peak_heights'], bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with auto bins")
plt.show()


histPeak = np.histogram(properties_1['peak_heights'],range=(low,max(signal)))
freqPeaks= histPeak[0]
ampsPeaks= histPeak[1]
peaks_hist, properties_hist= find_peaks(freqPeaks)
 # arguments are passed to np.histogram
print('fp',freqPeaks)

print('ap',ampsPeaks)

print(peaks_hist)

print('B',Blocallist)
print('E',Elocallist)
print('H',heights)
print('W',width)
print('P',peaks)
print('period',np.diff(peaks))
"""

"""
from scipy.stats import norm
x_d = np.linspace(min(x), max(x), 100)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density,alpha=0.05)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
plt.show()
"""
