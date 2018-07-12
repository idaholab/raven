import math
from scipy.integrate import quad
import time

def timeDepLambda(t,a,b):
  exponent = a+t*b
  return 10**exponent

def pdfFailure(t,a,b):
  first  = timeDepLambda(t,a,b)
  second = math.exp(-quad(timeDepLambda, 0, t, args=(a,b))[0]) 
  return first*second

t = 3700.
a_V2 = -4.
b_V2 = 1.E-4

start_time = time.clock()
probability1 = quad(pdfFailure, 0, t, args=(a_V2,b_V2))
print(probability1, time.clock() - start_time)

start_time = time.clock()
probability2 = 1. - math.exp(-quad(timeDepLambda, 0, t, args=(a_V2,b_V2))[0]) 
print(probability2, time.clock() - start_time)