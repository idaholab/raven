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
Created on 2021 September 12

@author: Robert Flanagan
"""

import numpy as np
import math
import pandas as pd

from scipy import stats


def buildSegments(array, segLength: int):
  """
    This method groups elements of an array into segments of length segLength
    @In array, array-like containing data to be grouped into segments.
    @In seg_length, size of the segment desired
    @Out, array of segments of length seg_length
  """
  s = math.floor(len(array)/segLength)
  segments = []
  for i in range(s+1):
    segments.append(array[i*segLength:(i+1)*segLength])
  return np.array(segments)

def buildIntervalSegs(array, interval: int):
  """
    This method groups elements of an array into segments based on an interval.
    @In array, array-like containing data to be grouped into segments.
    @In interval, size of the interval desired
    @ Out, array of segments of length seg_length
  """
  interSegs = []
  for i in range(interval):
    interSegs.append(array[i::interval])
  return np.array(interSegs)

def sumSegments(array):
  """
    This method comes segments arrays built using the buildSegements method using a KS-test.
    @In array1, an array-like of the historical data.
    @In array2, an array-like of the synthetic data.
    @In interval, int setting the length of the segment length desired
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test.
  """
  sum_segs = []
  for seg in array:
    sum_segs.append(sum(seg))
  return np.array(sum_segs)

def compare_segments_ks(array1, array2, segmentLength: int):
  """
  This method comes segments arrays built using the buildSegements method using a KS-test.
    @In array1, an array-like of the historical data.
    @In array2, an array-like of the synthetic data.
    @In interval, int setting the length of the segment length desired
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test.
  """
  results = []
  cols = ['segment','statistic', 'p-value']
  segments1 = buildSegments(array1, segmentLength)
  segments2 = buildSegments(array2, segmentLength)
  if len(segments1) != len(segments2) or len(segments1[0]) != len(segments2[0]):
    print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
    return
  for i in range(len(segments1)):
    results.append(i, stats.ks_2samp(segments1[i], segments2[i]))
  return pd.DataFrame(results, columns=cols)

def compare_segments_f(array1, array2, segmentLength: int):
  """
    This method comes segments arrays built using the buildSegements method using a KS-test.
    @In array1, an array-like of the historical data.
    @In array2, an array-like of the synthetic data.
    @In interval, int setting the length of the segment length desired
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test.
  """
  results = []
  cols = ['segment','statistic', 'p-value']
  segments1 = buildSegments(array1, segmentLength)
  segments2 = buildSegments(array2, segmentLength)
  if len(segments1) != len(segments2) or len(segments1[0]) != len(segments2[0]):
    print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
    return
  for i in range(len(segments1)):
    results.append(i, stats.f_oneway(segments1[i], segments2[i]))
  return pd.DataFrame(results, columns=cols)

def compare_segments_chi(array1, array2, segmentLength: int):
  """
    This method comes segments arrays built using the buildSegements method using a KS-test.
    @In array1, an array-like of the historical data.
    @In array2, an array-like of the synthetic data.
    @In interval, int setting the length of the segment length desired
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test.
  """
  results = []
  cols = ['segment','statistic', 'p-value']
  segments1 = buildSegments(array1, segmentLength)
  segments2 = buildSegments(array2, segmentLength)
  if len(segments1) != len(segments2) or len(segments1[0]) != len(segments2[0]):
    print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
    return
  for i in range(len(segments1)):
    results.append(i, stats.chisquare(segments1[i], segments2[i]))
  return pd.DataFrame(results, columns=cols)

def compare_interval_ks(array1, array2, interval: int):
  """
    This method comes segments arrays built using the buildSegements method using a KS-test.
    @In array1, an array-like of the historical data.
    @In array2, an array-like of the synthetic data.
    @In interval, int setting the length of the interval desired
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test.
  """
  results = []
  cols = ['segment','statistic', 'p-value']
  interval1 = buildIntervalSegs(array1, interval)
  interval2 = buildIntervalSegs(array2, interval)
  if len(interval1) != len(interval2) or len(interval1[0]) != len(interval2[0]):
    print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
    return
  for i in range(len(interval1)):
    results.append(i, stats.ks_2samp(interval1[i], interval2[i]))
  return pd.DataFrame(results, columns=cols)

def compare_interval_f(array1, array2, interval):
  """
    This method comes segments arrays built using the buildSegements method using an oneway f-test.
    @In array1, an array-like of the historical data.
    @In array2, an array-like of the synthetic data.
    @In interval, int setting the length of the interval desired
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the one way f-test.
  """
  results = []
  cols = ['segment','statistic', 'p-value']
  interval1 = buildIntervalSegs(array1, interval)
  interval2 = buildIntervalSegs(array2, interval)
  if len(interval1) != len(interval2) or len(interval1[0]) != len(interval2[0]):
    print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
    return
  for i in range(len(interval1)):
    results.append(i, stats.f_oneway(interval1[i], interval2[i]))
  return pd.DataFrame(results, columns=cols)

def compare_intervals_chi(array1, array2, interval):
  """
    This method comes segments arrays built using the buildSegements method using a chisquare test.
    @In array1, an array-like of the historical data.
    @In array2, an array-like of the synthetic data.
    @In interval, int setting the length of the interval desired
    @ Out, a Pandas DataFrame containing the statistic and the p-value of a chisquare test.
  """
  results = []
  cols = ['segment','statistic', 'p-value']
  interval1 = buildIntervalSegs(array1, interval)
  interval2 = buildIntervalSegs(array2, interval)
  if len(interval1) != len(interval2) or len(interval1[0]) != len(interval2[0]):
    print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
    return
  for i in range(len(interval1)):
    results.append(i, stats.chisquare(interval1[i], interval2[i]))
  return pd.DataFrame(results, columns=cols)

def compare_sums_ks(array1, array2):
  """
    This method compares the summed segments from the sumSegements arrays using a KS-test.
    @In array1, an array-like from the sumSegements function.
    @In array2, an array-like from the sumSegements function.
    @ Out, the output of a scipy two samples KS test.
  """
  return stats.ks_2samp(array1, array2)

def compare_sums_f(array1, array2):
  """
    This method compares the summed segments from the sumSegements arrays using an one-way f-test.
    @In array1, an array-like from the sumSegements function.
    @In array2, an array-like from the sumSegements function.
    @ Out, the output of a scipy one-way f-test.
  """
  return stats.f_oneway(array1, array2)

def compare_sums_chi(array1, array2):
  """
    This method compares the summed segments from the sumSegements arrays using a chisquare test.
    @In array1, an array-like from the sumSegements function.
    @In array2, an array-like from the sumSegements function.
    @ Out, the output of a scipy chisquare test.
  """
  return stats.chisquare(array1, array2)
