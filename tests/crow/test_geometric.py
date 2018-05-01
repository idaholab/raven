#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import crowTestUtils as utils

distribution1D = utils.findCrowModule('distribution1D')

geometric_distribution = distribution1D.BasicGeometricDistribution(0.25)

results = {"pass":0,"fail":0}

utils.checkAnswer("geometric cdf(0)",geometric_distribution.cdf(0),0.25,results)
utils.checkAnswer("geometric cdf(1)",geometric_distribution.cdf(1),0.4375,results)
utils.checkAnswer("geometric mean",geometric_distribution.untrMean(),3.0,results)
utils.checkAnswer("geometric stddev",geometric_distribution.untrStdDev(),3.46410161514,results)

utils.checkAnswer("geometric ppf(0.1)",geometric_distribution.inverseCdf(0.1),0.0,results)
utils.checkAnswer("geometric ppf(0.3)",geometric_distribution.inverseCdf(0.3),0.239823326142,results)
utils.checkAnswer("geometric ppf(0.8)",geometric_distribution.inverseCdf(0.8),4.59450194,results)
utils.checkAnswer("geometric ppf(0.9)",geometric_distribution.inverseCdf(0.9),7.00392277965,results)


print(results)

sys.exit(results["fail"])
