import numpy as np

from scipy import stats


def build_segments(array, seg_length: int):
"""
This method groups elements of an array into segments of length seg_length
@In array, array-like containing data to be grouped into segments.
@In seg_length, size of the segment desired
@Out, array of segments of length seg_length
"""
s = math.floor(len(array)/seg_length)
segments = []
for i in range(s+1):
    segments.append(array[i*seg_length:(i+1)*seg_length])
return np.array(segments)

def build_interval_segs(array, interval: int):
"""
This method groups elements of an array into segments based on an interval.

@In array, array-like containing data to be grouped into segments.
@In interval, size of the interval desired
@ Out, array of segments of length seg_length
"""
inter_segs = []
for i in range(interval):
    intervals.append(array[i::interval])    
return np.array(inter_segs)

def sum_segments(array):
"""
This method comes segments arrays built using the build_segments method using a KS-test.
@In array1, an array-like of the historical data.
@In array2, an array-like of the synthetic data.
@In interval, int setting the length of the segment length desired
@ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test. 
"""
sum_segs = []
for seg in array:
    sum_segs.append(sum(seg))
return np.array(sum_segs)

def compare_segments_ks(array1, array2, segment_length: int):
"""
This method comes segments arrays built using the build_segments method using a KS-test.
@In array1, an array-like of the historical data.
@In array2, an array-like of the synthetic data.
@In interval, int setting the length of the segment length desired
@ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test. 
"""
    results = []
    cols = ['segment','statistic', 'p-value']
    segments1 = build_segments(array1, segment_length)
    segments2 = build_segments(array2, segment_length)
    if len(segments1) != len(segments2) or len(segments1[0]) != len(segments2[0]):
        print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
        return
    for i in range(len(segments1)):
        results.append(i, stats.ks_2samp(segments1[i], segments2[i]))
    return pd.DataFrame(results, columns=cols)

def compare_segments_f(array1, array2, segment_length: int):
"""
This method comes segments arrays built using the build_segments method using a KS-test.
@In array1, an array-like of the historical data.
@In array2, an array-like of the synthetic data.
@In interval, int setting the length of the segment length desired
@ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test. 
"""
    results = []
    cols = ['segment','statistic', 'p-value']
    segments1 = build_segments(array1, segment_length)
    segments2 = build_segments(array2, segment_length)
    if len(segments1) != len(segments2) or len(segments1[0]) != len(segments2[0]):
        print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
        return
    for i in range(len(segments1)):
        results.append(i, stats.f_oneway(segments1[i], segments2[i]))
    return pd.DataFrame(results, columns=cols)

def compare_segments_chi(array1, array2, segment_length: int):
"""
This method comes segments arrays built using the build_segments method using a KS-test.
@In array1, an array-like of the historical data.
@In array2, an array-like of the synthetic data.
@In interval, int setting the length of the segment length desired
@ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test. 
"""
    results = []
    cols = ['segment','statistic', 'p-value']
    segments1 = build_segments(array1, segment_length)
    segments2 = build_segments(array2, segment_length)
    if len(segments1) != len(segments2) or len(segments1[0]) != len(segments2[0]):
        print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
        return
    for i in range(len(segments1)):
        results.append(i, stats.chisquare(segments1[i], segments2[i]))
    return pd.DataFrame(results, columns=cols)

def compare_interval_ks(array1, array2, interval: int):
"""
This method comes segments arrays built using the build_segments method using a KS-test.
@In array1, an array-like of the historical data.
@In array2, an array-like of the synthetic data.
@In interval, int setting the length of the interval desired
@ Out, a Pandas DataFrame containing the statistic and the p-value of the KS-test. 
"""
    results = []
    cols = ['segment','statistic', 'p-value']
    interval1 = build_interval_segs(array1, interval)
    interval2 = build_interval_segs(array2, interval)
    if len(interval1) != len(interval2) or len(interval1[0]) != len(interval2[0]):
        print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
        return
    for i in range(len(interval1)):
        results.append(i, stats.ks_2samp(interval1[i], interval2[i]))
    return pd.DataFrame(results, columns=cols)

def compare_interval_f(array1, array2, interval):
"""
This method comes segments arrays built using the build_segments method using an oneway f-test.
@In array1, an array-like of the historical data.
@In array2, an array-like of the synthetic data.
@In interval, int setting the length of the interval desired
@ Out, a Pandas DataFrame containing the statistic and the p-value of the one way f-test. 
"""
    results = []
    cols = ['segment','statistic', 'p-value']
    interval1 = build_interval_segs(array1, interval)
    interval2 = build_interval_segs(array2, interval)
    if len(interval1) != len(interval2) or len(interval1[0]) != len(interval2[0]):
        print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
        return
    for i in range(len(interval1)):
        results.append(i, stats.f_oneway(interval1[i], interval2[i]))
    return pd.DataFrame(results, columns=cols)

def compare_intervals_chi(array1, array2, interval):
"""
This method comes segments arrays built using the build_segments method using a chisquare test.
@In array1, an array-like of the historical data.
@In array2, an array-like of the synthetic data.
@In interval, int setting the length of the interval desired
@ Out, a Pandas DataFrame containing the statistic and the p-value of a chisquare test. 
"""
    results = []
    cols = ['segment','statistic', 'p-value']
    interval1 = build_interval_segs(array1, interval)
    interval2 = build_interval_segs(array2, interval)
    if len(interval1) != len(interval2) or len(interval1[0]) != len(interval2[0]):
        print('Warning you are attempting to compare two arrays of uneven length or with uneven length components')
        return
    for i in range(len(interval1)):
        results.append(i, stats.chisquare(interval1[i], interval2[i]))
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