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
Created on September 11, 2018

@author: talbpaul
"""
#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from ...utils import InputData, InputTypes
from .PostProcessorInterface import PostProcessorInterface
#Internal Modules End-----------------------------------------------------------

DETRENDS = ['none', 'constant', 'linear']
WINDOWS = ['none', 'hann', 'hamming', 'blackman']
WINDOW_FUNCTIONS = {'hann': np.hanning, 'hamming': np.hamming, 'blackman': np.blackman}

def computeFFT(values, pivot, detrend='none', window='none', spacingTol=1e-6):
  """
    Single-sided amplitude spectrum of a signal sampled at uniformly-spaced pivot values
    @ In, values, np.array, signal values
    @ In, pivot, np.array, pivot (e.g. time) values, uniformly spaced
    @ In, detrend, str, optional, 'none', 'constant' (remove the mean) or 'linear' (remove a linear fit)
    @ In, window, str, optional, 'none', 'hann', 'hamming' or 'blackman'
    @ In, spacingTol, float, optional, relative tolerance on the uniformity of the pivot spacing
    @ Out, (freq, period, amplitude, phase), tuple(np.array), frequencies (1/pivot units), periods (pivot units,
           inf for the zero frequency), amplitudes (a sinusoid of amplitude A gives a peak A; the zero frequency
           gives the mean) and phases (rad)
  """
  values = np.asarray(values, dtype=float)
  pivot = np.asarray(pivot, dtype=float)
  if detrend not in DETRENDS:
    raise ValueError(f'Unknown detrend "{detrend}"! Available: {DETRENDS}')
  if window not in WINDOWS:
    raise ValueError(f'Unknown window "{window}"! Available: {WINDOWS}')
  n = len(values)
  if n < 2:
    raise ValueError(f'At least 2 values are needed for the fast Fourier transform, got {n}!')
  steps = np.diff(pivot)
  dt = steps.mean()
  if dt <= 0 or np.any(np.abs(steps - dt) > spacingTol * abs(dt)):
    raise ValueError('The pivot values must be uniformly spaced and increasing for the fast Fourier transform '
                     f'(steps between {steps.min()} and {steps.max()}). Resample the data first, '
                     'e.g. with the HistorySetSync PostProcessor.')
  if detrend == 'constant':
    values = values - values.mean()
  elif detrend == 'linear':
    values = values - np.polyval(np.polyfit(pivot, values, 1), pivot)
  weights = np.ones(n) if window == 'none' else WINDOW_FUNCTIONS[window](n)
  transform = np.fft.rfft(values * weights)
  freq = np.fft.rfftfreq(n, d=dt)
  # single-sided amplitude, corrected for the window coherent gain: the zero frequency (and the Nyquist
  # frequency for an even number of values) are not doubled
  amplitude = np.abs(transform) / (n * weights.mean())
  doubled = np.ones(len(freq), dtype=bool)
  doubled[0] = False
  if n % 2 == 0:
    doubled[-1] = False
  amplitude[doubled] *= 2.0
  period = np.full(len(freq), np.inf)
  period[1:] = 1.0 / freq[1:]
  return freq, period, amplitude, np.angle(transform)

class FastFourierTransform(PostProcessorInterface):
  """
    Constructs fast-fourier transform data for a history
    Outputs are "frequency" for each index and "period", "amplitude" and "phase" for each target
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inSpec = super(FastFourierTransform, cls).getInputSpecification()
    inSpec.addSub(InputData.parameterInputFactory('target',
                                                  contentType=InputTypes.StringListType,
                                                  strictMode=True,
                                                  descr=r"""specifies the names of the target(s) for which the fast
                                                  Fourier transform should be calculated."""))
    inSpec.addSub(InputData.parameterInputFactory('detrend',
                                                  contentType=InputTypes.makeEnumType('fftDetrend', 'fftDetrendType', DETRENDS),
                                                  descr=r"""removes a trend from each signal before the transform:
                                                  \xmlString{none} (no detrending), \xmlString{constant} (the mean is
                                                  removed) or \xmlString{linear} (a linear fit over the pivot is removed).
                                                  Detrending is recommended for drifting or transient signals, whose trend
                                                  otherwise dominates the spectrum.""", default='none'))
    inSpec.addSub(InputData.parameterInputFactory('window',
                                                  contentType=InputTypes.makeEnumType('fftWindow', 'fftWindowType', WINDOWS),
                                                  descr=r"""window applied to each (detrended) signal before the transform
                                                  to reduce the spectral leakage: \xmlString{none}, \xmlString{hann},
                                                  \xmlString{hamming} or \xmlString{blackman}. The amplitudes are corrected
                                                  for the window coherent gain, so that a sinusoid keeps its amplitude.""",
                                                  default='none'))
    return inSpec

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dynamic = True # from base class, indicates time-dependence is handled internally
    self.targets = None # list of strings, variables to apply postprocessor to
    self.indices = None # dict of {string:string}, key is target and value is index (independent monotonic var)
    self.detrend = 'none' # detrending applied before the transform
    self.window = 'none' # window applied before the transform

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      tag = child.getName()
      if tag == 'target':
        # keep the input order (and remove duplicates)
        self.targets = list(dict.fromkeys(child.value))
      elif tag == 'detrend':
        self.detrend = child.value
      elif tag == 'window':
        self.window = child.value

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      In this case, we only want data objects!
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) > 1:
      self.raiseAnError(IOError, 'Expected 1 input HistorySet or DataSet, but received {} inputs!'.format(len(currentInp)))
    currentInp = currentInp[0]
    if currentInp.type not in ['HistorySet','DataSet']:
      self.raiseAnError(IOError, 'FastFourierTransform postprocessor "{}" requires a HistorySet or DataSet input! Got "{}".'
                                 .format(self.name, currentInp.type))
    # check targets requested depend only on one index (no more or less)
    okay = True
    self.indices = {}
    for target in self.targets:
      indices = currentInp.getDimensions(var=target)[target]
      if len(indices) != 1:
        okay = False
        if len(indices) == 0:
          self.raiseAWarning('Target "{}" is a scalar! FFT cannot be performed.'.format(target))
        else:
          self.raiseAWarning('Target "{}" has {} index dimensions! FFT can only currently handle one.'.format(target,len(indices)))
      else:
        self.indices[target] = indices[0]
    if not okay:
      self.raiseAnError(IndexError,'Some targets were not dimensioned correctly. See warnings for more details. Exiting.')
    # return data object
    return currentInp

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, realizations, list, list of realizations obtained
    """
    # do checking and isolate input
    inData = self.inputToInternal(inputIn)

    # storage for each realization
    realizations = []
    # loop over realizations and perform fft
    for s in range(len(inData)):
      # obtain relevant sample
      sample = inData.realization(index=s)
      # solutions for each realization are stored in this dict
      rlz = {}
      # separate curve for each target
      for target in self.targets:
        values = np.asarray(sample[target].values, dtype=float)
        pivot = np.asarray(sample[target][self.indices[target]].values, dtype=float)
        # values padded with NaN (e.g. histories of different lengths in a DataSet) are removed
        valid = np.isfinite(values) & np.isfinite(pivot)
        try:
          freq, period, amplitude, phase = computeFFT(values[valid], pivot[valid], self.detrend, self.window)
        except ValueError as err:
          self.raiseAnError(ValueError, f'FastFourierTransform "{self.name}", target "{target}", realization {s}: {err}')
        rlz[target+'_fft_frequency'] = freq
        rlz[target+'_fft_period'] = period
        rlz[target+'_fft_amplitude'] = amplitude
        rlz[target+'_fft_phase'] = phase
      realizations.append(rlz)
    return realizations

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    realizations = evaluation[1]
    for rlz in realizations:
      output.addRealization(rlz)
