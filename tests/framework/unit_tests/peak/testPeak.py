import sys, os
from scipy import stats
import scipy

import pickle as pk
import copy
import xml.etree.ElementTree as ET

# find location of crow, message handler
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4+['framework'])))
sys.path.append(frameworkDir)
from utils.utils import find_crow
find_crow(frameworkDir)
from utils import randomUtils
import MessageHandler

# message handler
mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})
# input specs come mostly from the Models.ROM
from Models import ROM

# find location of ARMA
# sys.path.append('/Users/zhouj2/projects/raven/framework/SupervisedLearning/')
from SupervisedLearning import ARMA
#import ARMA



import zjpeak
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde,rv_histogram

def smooth(x,window_len=11):
  s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
  w=np.ones(window_len,'d')
  y=np.convolve(w/w.sum(),s,mode='valid')
  return y[int(np.ceil(window_len/2)-1):int(np.ceil(-window_len/2))]


## user define
expPeakTypes= 2
bg_p=3000
ed_p=4000
data    =   pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv")
time    =   data.ind.values[bg_p:ed_p]
signal  =   data.signal.values[bg_p:ed_p]
low     =   0


maskRes=signal==signal
width=5
# peaks_1, properties_1 = scipy.signal.find_peaks(signal, height=0)
# heights = properties_1['peak_heights']
#### window filter
period=24
windows=[[-2,3],[18,21]]
windowType=len(windows)

########
rangeWindow=[]
for i in range(windowType):
  windowRange={}
  bg_P=(windows[i][0]-1)%period
  end_P=(windows[i][1]+2)%period
  bg_P_ind  = np.where(time%period==bg_P )[0].tolist()
  end_P_ind = np.where(time%period==end_P)[0].tolist()
  if bg_P_ind[0]>end_P_ind[0]:
    tail=end_P_ind[0]
    end_P_ind.pop(0)
    end_P_ind.append(tail)
  windowRange['bg']=bg_P_ind
  windowRange['end']=end_P_ind
  rangeWindow.append(windowRange)
#print(rangeWindow)

#########
groupWin=[]
for i in range(windowType):
  bg  = rangeWindow[i]['bg']
  end = rangeWindow[i]['end']
  peakInfo   = {}
  indLocal   = []
  ampLocal   = []
  for j in range(len(bg)):
    bg_local = bg[j]
    end_local = end[j]
    peak, height = zjpeak._peakPicker(signal[bg_local:end_local], low=low)
    if len(peak) ==1:
      indLocal.append(int(peak))
      ampLocal.append(float(height))
      mask_bg=int(peak)+bg_local-int(np.floor(width/2))
      mask_end=int(peak)+bg_local+int(np.ceil(width/2))
      maskRes[mask_bg:mask_end]=False
    elif len(peak) >1:
      indLocal.append(int(peak[np.argmax(height)]))
      ampLocal.append(float(height[np.argmax(height)]))
      mask_bg=int(peak[np.argmax(height)])+bg_local-int(np.floor(width/2))
      mask_end=int(peak[np.argmax(height)])+bg_local+int(np.ceil(width/2))
      maskRes[mask_bg:mask_end]=False
  peakInfo['Ind'] = indLocal
  peakInfo['Amp'] = ampLocal

  groupWin.append(peakInfo)


Res=signal[maskRes]

print(maskRes[0:48])
#### Fourier
###################
######################################
def createElement(tag,attrib=None,text=None):
  """
    Method to create a dummy xml element readable by the distribution classes
    @ In, tag, string, the node tag
    @ In, attrib, dict, optional, the attribute of the xml node
    @ In, text, str, optional, the dict containig what should be in the xml text
  """
  if attrib is None:
    attrib = {}
  if text is None:
    text = ''
  element = ET.Element(tag,attrib)
  element.text = text
  return element

def createARMAXml(targets, pivot, p, q, fourier=None):
  if fourier is None:
    fourier = []
  xml = createElement('ROM',attrib={'name':'test', 'subType':'ARMA'})
  xml.append(createElement('Target',text=','.join(targets+[pivot])))
  xml.append(createElement('Features',text='scaling'))
  xml.append(createElement('pivotParameter',text=pivot))
  xml.append(createElement('P',text=str(p)))
  xml.append(createElement('Q',text=str(q)))
  if len(fourier):
    xml.append(createElement('Fourier',text=','.join(str(f) for f in fourier)))
  return xml

def createFromXML(xml):
  inputSpec = ROM.getInputSpecification()
  rom = ROM({})
  rom.messageHandler = mh
  rom._readMoreXML(xml)
  arma = rom.supervisedEngine.supervisedContainer[0]
  return rom, arma

def createARMA(targets, pivot, p, q, fourier=None):
  xml = createARMAXml(targets, pivot, p, q, fourier)
  rom, arma = createFromXML(xml)
  return rom, arma


rom, arma = createARMA(['a'], 't', 0, 0, [])
arma.pivotParameterValues=time
fourierResults={}
target='sgn'
fourierResults[target]= arma._trainFourier(time,[720,48,24],signal,masks=[maskRes])
signal_1 =signal- fourierResults[target]['predict']

plt.plot(signal)
plt.plot(np.ones_like(signal)*low, "--", color="gray")
plt.plot(fourierResults[target]['predict'])
plt.plot(time[maskRes]-bg_p,signal_1[maskRes],'.-')
#plt.plot(np.ones(len(signal))*maskRes*60,'-')
plt.xlim(100,175)
plt.show()


#### Peak treatment
#### Input:  data, expPeakTypes,low
#### Output: ampPeak,periodPeak,widthPeak,peaks_1[0],
####         resSig for arma
###




# peakSig,resSig=zjpeak._classifyPeaksSignal(signal,low)

# expAmpTypes=zjpeak._setExpAmpTypes(expPeakTypes)

# peaks_1,heights=zjpeak._peakPicker(signal,low)


# peakIndex,resIndex=zjpeak._classifyPeaksInd(signal,low)

# Blocallist, Elocallist=zjpeak._collectPeakRange(peakIndex)

# Wlocallist=zjpeak._collectPeakWidth(Blocallist,Elocallist)


# ampRange=zjpeak._generateAmpRange(signal,low,expPeakTypes,expAmpTypes,heights)

# groupsAmp=zjpeak._groupByAmplitude(expAmpTypes,peaks_1,heights,ampRange,Blocallist,Elocallist,Wlocallist)

# widthPeak=zjpeak._collectWidth(expPeakTypes,groupsAmp,Wlocallist)

# periodPeak=zjpeak._collectPeriod(expPeakTypes,groupsAmp,peaks_1)

# ampPeak=zjpeak._collectAmp(expPeakTypes,groupsAmp,heights)

# print('h',ampPeak[0].rvs())
# print('p',periodPeak[0].rvs())
# print('w',widthPeak[0].rvs())
# print('h1',ampPeak[1].rvs())
# print('p1',periodPeak[1].rvs())
# print('w1',widthPeak[1].rvs())
# print('p',periodPeak[0].mean())

# plt.plot(signal)
# for i in range(expAmpTypes):
#   plt.plot(groupsAmp[i]['Ind'], groupsAmp[i]['Amp'], "x")
# plt.plot(np.ones_like(signal)*low, "--", color="gray")
# plt.show()





def smooth(x,window_len=11):
  s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
  w=np.ones(window_len,'d')
  y=np.convolve(w/w.sum(),s,mode='valid')
  return y[int(np.ceil(window_len/2)-1):int(np.ceil(-window_len/2))]


"""

# print(len(signal))
# print(len(smooth(signal)))
# ptrange=range(0,480)
# plt.plot(time[ptrange],pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv").signal.values[ptrange])
# # plt.plot(time[ptrange],signal[ptrange])
# plt.plot(time[ptrange],signal[ptrange]-smooth(signal,1000)[ptrange])
# #for i in range(expAmpTypes):
# #  plt.plot(groupsAmp[i]['Ind'], groupsAmp[i]['Amp'], "x")
# #plt.plot(np.ones_like(signal)*low, "--", color="gray")
# plt.show()


# plt.hist(pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv").signal.values)
# plt.show()

# peaks_1, properties_1= scipy.signal.find_peaks(signal,height=5)
# peakyushu=peaks_1%24.0
# print(peakyushu)
# plt.hist(peakyushu)
# plt.show()
# peaks=[]
# for i in range(len(bg_P_ind)-1):
#   peaks_1, properties_1 = scipy.signal.find_peaks(signal[bg_P_ind[i]-1:end_P_ind[i]+1], height=low)
#   print(peaks_1)



# print(bg_P_ind)
# print(end_P_ind)
# print(len(bg_P_ind))
# print(len(end_P_ind))

# print(len(time%period))
#print(np.mean(signal))
peakbase=smooth(signal,72)
#peakbase=0
# peaks_1, properties_1= scipy.signal.find_peaks(signal,height=0,prominence=5)
# peaks_2, properties_1= scipy.signal.find_peaks(signal,height=10)

# winter=[1,1994]
# spring=[1994,3193]
# summer=[3193,6049]
# autumn=[6049,8760]
# plt.xlim(4000,5000)

# plt.plot(time,pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv").signal.values)
# plt.plot(peaks_1, pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv").signal.values[peaks_1], "o")
# plt.plot(peaks_2, pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv").signal.values[peaks_2], "x")

# low=10
# plt.plot(np.ones_like(signal)*low, "--", color="gray")
# plt.show()

plt.plot(signal)
for i in range(expAmpTypes):
  plt.plot(groupsAmp[i]['Ind'], groupsAmp[i]['Amp'], "x")
plt.plot(np.ones_like(signal)*low, "--", color="gray")
plt.show()


# plt.xlim(0,1440)
# plt.plot(time,pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv").signal.values)
# plt.plot(peaks_1, pd.read_csv("BW.node.prices.INL.2026_May10LP_solution_files.csv").signal.values[peaks_1], "x")
# plt.plot(np.ones_like(signal)*low, "--", color="gray")

# # plt.plot(time[ptrange],signal[ptrange])
# #plt.plot(time[ptrange],signal[ptrange]-smooth(signal,1000)[ptrange])
# #for i in range(expAmpTypes):
# #  plt.plot(groupsAmp[i]['Ind'], groupsAmp[i]['Amp'], "x")
# #plt.plot(np.ones_like(signal)*low, "--", color="gray")
# plt.show()



# import pylab

# def thresholding_algo(y, lag, threshold, influence):
#     signals = np.zeros(len(y))
#     filteredY = np.array(y)
#     avgFilter = [0]*len(y)
#     stdFilter = [0]*len(y)
#     avgFilter[lag - 1] = np.mean(y[0:lag])
#     stdFilter[lag - 1] = np.std(y[0:lag])
#     for i in range(lag, len(y) - 1):
#         if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
#             if y[i] > avgFilter[i-1]:
#                 signals[i] = 1
#             else:
#                 signals[i] = -1

#             filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
#             avgFilter[i] = np.mean(filteredY[(i-lag):i])
#             stdFilter[i] = np.std(filteredY[(i-lag):i])
#         else:
#             signals[i] = 0
#             filteredY[i] = y[i]
#             avgFilter[i] = np.mean(filteredY[(i-lag):i])
#             stdFilter[i] = np.std(filteredY[(i-lag):i])

#     return dict(signals = np.asarray(signals),
#                 avgFilter = np.asarray(avgFilter),
#                 stdFilter = np.asarray(stdFilter))



# # Data
# y = np.array([1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,
#        1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,
#        2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1])
# y=signal
# # Settings: lag = 30, threshold = 5, influence = 0
# lag = 40
# threshold = 1.9
# influence = 0

# # Run algo with settings from above
# result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

# # Plot result
# pylab.subplot(211)
# pylab.plot(np.arange(1, len(y)+1), y)

# pylab.plot(np.arange(1, len(y)+1),
#            result["avgFilter"], color="cyan", lw=2)

# pylab.plot(np.arange(1, len(y)+1),
#            result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

# pylab.plot(np.arange(1, len(y)+1),
#            result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

# pylab.subplot(212)
# pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
# pylab.ylim(-1.5, 1.5)
# pylab.show()

"""