"""
Created on October 28, 2015

"""

import numpy as np


def run(self,Input,Param):
  """
   This function does things.
   @ In : inputDict
   @ Out: outputDict
  """

  for i in Input:
    for histID in i.getParametersValues('output'):
      if samplingType == 'uniform' or samplingType == 'firstDerivative' or samplingType == 'secondDerivative':
        tempData = mathUtils.varsTimeInterp(self.sampling['numSamples'], self.sampling['pivot'], i.getParametersValues('output')[histID], self.sampling['type'],self.sampling['interp'])
      elif samplingType == 'filteredFirstDerivative' or samplingType == 'filteredSecondDerivative':
        tempData = mathUtils.timeSeriesFilter(self.sampling['pivot'], i.getParametersValues('output')[histID], self.sampling['type'], self.sampling['tolerance'])
      else:
        self.raiseAnError(IOError, 'DataConversion Post-Processor: sampling type ' + str(self.sampling['type']) + ' is not valid for HS2HS')
      for key in tempData:
        i.updateOutputValue(key,tempData[key])


def timeSeriesFilter(timeID, vars, filterType, filterValue):
  """ This function sample a multi-variate temporal function
  timeID      : the ID of the temporal variable
  vars        : data set that contained the information of the multi-variate temporal function (this is supposed to be a dictionary: {'timeID':time_array, 'var1':var1_array, ..., 'varn':varn_array})
  samplingType: type of sampling used to determine the coordinate of the numSamples samples ('firstDerivative', 'secondDerivative')
  filterValue : value associated to the filter
  """
  derivative = np.zeros(time.size)

  if filterType=='firstDerivative':
    for t in range(1, time.size):
      t_contrib=0.0
      for keys in vars.keys():
        t_contrib += abs(var[keys][t] - var[keys][t-1])/(time[t]-time[t-1])
      derivative[t] = t_contrib

  elif filterType=='secondDerivative':
    for t in range(1, time.size-1):
      t_contrib=0.0
      for keys in vars.keys():
        t_contrib += abs(var[keys][t+1] - 2.0 * var[keys][t+1] + var[keys][t-1])/(time[t]-time[t-1])**2
      derivative[t] = t_contrib
    derivative[-1] = derivative[time.shape[0]-2]

  else:
    self.raiseAnError(RuntimeError,'filter Type ' + filterType + ' is not a valid type. Function: timeSeriesFilter (mathUtils)')

  newVars = {}
  for key in vars:
    newVars[key]=np.array(vars[key][0])

  for t in range(derivative.size):
    if secondDerivative[t] > filterValue:
      for key in vars:
        newVars[key]=np.append(newVars[key],vars[key][t])

  for key in vars:
    newVars[key]=np.append(newVars[key],vars[key][-1])

  return newVars

def varsTimeInterp(numSamples, timeID, vars, samplingType, interpType):
  """ This function sample a multi-variate temporal function
  numSamples  : number of samples
  timeID      : the ID of the temporal variable
  samplingType: type of sampling used to determine the coordinate of the numSamples samples ('uniform', 'firstDerivative', 'secondDerivative')
  vars        : data set that contained the information of the multi-variate temporal function (this is supposed to be a dictionary: {'timeID':time_array, 'var1':var1_array, ..., 'varn':varn_array})
  samplingType: specifies how the location of the new samples is chosen (uniform,firstDerivative,secondDerivative)
  interpType  : Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of first, second or third order) or as an integer specifying  the order of the spline interpolator to use. Default is 'linear'.
  """

  t_min = vars[timeID][0]
  t_max = vars[timeID][-1]

  dt=(t_max-t_min)/numSamples

  newVars={}

  if samplingType == 'uniform':
    newTime = np.arange(t_min,t_max,dt)
  elif samplingType == 'firstDerivative' or samplingType == 'secondDerivative':
    newTime = derivativeTimeValues(numSamples, vars[timeID], vars, samplingType)
  else:
    self.raiseAnError(RuntimeError,'type ' + samplingType + ' is not a valid type. Function: varsTimeInterp (mathUtils)')

  for key in vars.keys():
    if key == timeID:
      newVars[key] = newTime
    else:
      interp = interpolate.interp1d(vars[timeID], vars[key], interpType)
      print(vars[timeID])
      print(vars[key])
      print(newTime)
      newVars[key]=interp(newTime)

  return newVars


def derivativeTimeValues(numSamples, timeID, vars, orderDerivative):
  t_min=time[0]
  t_max=time[-1]

  newTime = np.zeros(numSamples)
  cumDerivative = np.zeros(time.size)

  # data normalization
  for keys in vars.keys():
    total = np.sum(var[keys])
    var[keys] = var[keys]/total

  if orderDerivative=='firstDerivative':
    for t in range(1, time.shape[0]):
      t_contrib=0.0
      for keys in vars.keys():
        t_contrib += abs(var[keys][t] - var[keys][t-1])/(time[t]-time[t-1])
      cumDerivative[t] = cumDerivative[t-1] + t_contrib

  elif orderDerivative=='secondDerivative':
    for t in range(1, time.shape[0]-1):
      t_contrib=0.0
      for keys in vars.keys():
        t_contrib += abs(var[keys][t+1] - 2.0 * var[keys][t+1] + var[keys][t-1])/(time[t]-time[t-1])**2
      cumDerivative[t] = cumDerivative[t-1] + t_contrib
    cumDerivative[-1] = cumDerivative[time.shape[0]-2]

  else:self.raiseAnError(RuntimeError,'type ' + orderDerivative + ' is not a valid type. Function: derivativeTimeValues (mathUtils)')


  cumDamageInstant = np.arange(0,cumDerivative[-1],numSamples)

  for i in range(numSamples):
    index = (np.abs(cumDerivative - cumDamageInstant[i])).argmin()
    newTime[i] = time[index]

  return newTime
