import sys
import math

def pressureRupture(a,type):
   "calculate the rupture pressure for states C and L"
   
   h     = 0.038    # in m
   H     = 0.3048   # in m
   sigmaF = 333      # in MPa
   
   if type==1:   # circum type
     b=a/0.1
   else:
     b=a
  
   if a<h:
     M = 1-math.exp(-0.157*b/(math.sqrt(H*(h-a)/2)))
   else:
     M=0
     
   pressureF = 4*sigmaF*h/H*(1-a/h*M)
   
   return pressureF

