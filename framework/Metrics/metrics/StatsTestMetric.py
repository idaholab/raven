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

from .MetricInterface import MetricInterface
from Metrics.metrics import MetricUtilities
from scipy import stats
class CDFAreaDifference(MetricInterface):
  """
    Metric to compare two datasets using the CDF Area Difference.
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # If True the metric needs to be able to handle a passed in Distribution
    self.acceptsDistribution = True

  def run(self, x, y, weights=None, axis=0, **kwargs):
    """
      This method computes difference between two points x and y based on given metric
      @ In, x,  instance of Distributions.Distribution, tuple or list, array containing data of x,
        or given distribution.
      @ In, y, instance of Distributions.Distribution, tuple or list, array containing data of y,
        or given distribution.
      @ In, weights, array_like (numpy.ndarray or list), optional, not used in this metric
      @ In, axis, integer, optional, default is 0, not used for this metric.
      @ In, kwargs,dict, dictionary of parameters characteristic of each metric
      @ Out, value, float, metric result, CDF area difference
    """
    value = MetricUtilities._getCDFAreaDifference(x,y)
    return float(value)

  def build_segments(array, seg_length: int):
  """
    This method groups elements of an array into segments of length seg_length
    @ In array, array-like containing data to be grouped into segments.
    @ Out, array of segments of length seg_length
  """
    s = math.floor(len(array)/seg_length)
    segments = []
    for i in range(s+1):
        segments.append(array[i*seg_length:(i+1)*seg_length])
    return np.array(segments)

  def build_interval_segs(array, interval: int):
  """
    This method groups elements of an array into segments based on an interval.
    
    @ In array, array-like containing data to be grouped into segments.
    @ Out, array of segments of length seg_length
  """
    inter_segs = []
    for i in range(interval):
        intervals.append(array[i::interval])    
    return np.array(inter_segs)

  def sum_segments(array):
  """
    This method sums the elements of an array of segments to perform other metric comparisons. 
    @ In array, an array-like containing segments from the build_segments function.
    @ Out, an array of floats representing the sum of each segment. 
  """
    sum_segs = []
    for seg in array:
        sum_segs.append(sum(seg))
    return np.array(sum_segs)
        
  def compare_segments_ks(array1, array2):
  """
    This method comes segments arrays built using the build_segments method using a KS-test.
    @In array1, an array-like containing segments from the build_segments function.
    @In array2, an array-like containing segments from the build_segments function.
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test. 
  """
      results = []
      cols = ['segment','statistic', 'p-value']
      if len(array1) != len(array2) or len(array1[0]) != len(array2[0]):
          print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
          return
      for i in range(len(array1)):
          results.append(i, stats.ks_2samp(array1[i], array2[i]))
      return pd.DataFrame(results, columns=cols)

  def compare_segments_f(array1, array2):
  """
    This method comes segments arrays built using the build_segments method using an oneway f-test.
    @In array1, an array-like containing segments from the build_segments function.
    @In array2, an array-like containing segments from the build_segments function.
    @ Out, a Pandas DataFrame containing the statistic and the p-value of the one way f-test. 
  """
      results = []
      cols = ['segment','statistic', 'p-value']
      if len(array1) != len(array2) or len(array1[0]) != len(array2[0]):
          print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
          return
      for i in range(len(array1)):
          results.append(i, stats.f_oneway(array1[i], array2[i]))
      return pd.DataFrame(results, columns=cols)

  def compare_segments_chi(array1, array2):
  """
    This method comes segments arrays built using the build_segments method using a chisquare test.
    @In array1, an array-like containing segments from the build_segments function.
    @In array2, an array-like containing segments from the build_segments function.
    @ Out, a Pandas DataFrame containing the statistic and the p-value of a chisquare test. 
  """
      results = []
      cols = ['segment','statistic', 'p-value']
      if len(array1) != len(array2) or len(array1[0]) != len(array2[0]):
          print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
          return
      for i in range(len(array1)):
          results.append(i, stats.chisquare(array1[i], array2[i]))
      return pd.DataFrame(results, columns=cols)

  def compare_sums_ks(array1, array2):
  """
    This method compares the summed segments from the sum_segments arrays using a KS-test.
    @In array1, an array-like from the sum_segments function.
    @In array2, an array-like from the sum_segments function.
    @ Out, the output of a scipy two samples KS test.  
  """
      return stats.ks_2samp(array1, array2)

  def compare_sums_f(array1, array2):
  """
    This method compares the summed segments from the sum_segments arrays using an one-way f-test.
    @In array1, an array-like from the sum_segments function.
    @In array2, an array-like from the sum_segments function.
    @ Out, the output of a scipy one-way f-test.  
  """
      return stats.f_oneway(array1, array2)

  def compare_sums_chi(array1, array2):
  """
    This method compares the summed segments from the sum_segments arrays using a chisquare test.
    @In array1, an array-like from the sum_segments function.
    @In array2, an array-like from the sum_segments function.
    @ Out, the output of a scipy chisquare test.  
  """
      return stats.chisquare(array1, array2)