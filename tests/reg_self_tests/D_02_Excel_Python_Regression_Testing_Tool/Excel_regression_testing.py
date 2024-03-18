# The Interface between Python and Excel
## Warning: Please close the tool before running this code
import os
import json
import numpy as np
import xlwings as xw
import time
import sys
import argparse
from pathlib import Path # Core Python MOdule
import pandas as pd

# Read the file name of the excel from the tests file
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="pass the excel file name to python script")
  # Inputs from the user
  parser.add_argument("xlsbPython", help="xlsbPython")
  args = parser.parse_args()
  fileName=args.xlsbPython
  wr=xw.Book(r'./'+ fileName)
  wr2=xw.Book(r'./Gold_'+ fileName)

# Read the inputs and outputs from the gold file
shtInp2=wr2.sheets['Inputs']
inputOld=pd.DataFrame(shtInp2.range('Inputs_table').value,columns=['Parameter', 'Unit', 'value'])
#print (np.array(input_old.iloc[:,2]))
shtOut2=wr2.sheets['Outputs']
outputOld=pd.DataFrame(shtOut2.range('Outputs_table').value,columns=['Parameter', 'Unit', 'value'])

# Write the inputs to the existing file
shtInp=wr.sheets['Inputs']
# store the inputs to be resoted
input_restore=shtInp.range('Inputs_table').value
input_new=pd.DataFrame(shtInp.range('Inputs_table').value, columns=['Parameter', 'Unit', 'value'])
shtInp.range('Inputs_table').value=shtInp2.range('Inputs_table').value
shtOut=wr.sheets['Outputs']
outputNew=pd.DataFrame(shtOut.range('Outputs_table').value,columns=['Parameter', 'Unit', 'value'])
# Pass for T while Failure for F
T=0
F=0
for i in range (len(outputNew.iloc[:,2])):
  #(check if error is less than 0.1% for each output)
  if abs(outputNew.iloc[i,2]-outputOld.iloc[i,2])/outputOld.iloc[i,2]<0.001:
    T+=1
    print ('Test passed for calculating'+str(outputNew.iloc[i,0]))
  else:
    F+=1
    print ('Test failed for calculating'+str(outputNew.iloc[i,0]))
shtInp.range('Inputs_table').value=input_restore
wr.save()
wr.close()
wr2.close()
xw.App().quit()
if F==0:
  sys.exit(0)
else:
  sys.exit(1)
