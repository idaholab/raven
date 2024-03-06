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
  parser.add_argument("xlsb_python", help="xlsb_python")
  args = parser.parse_args()
  file_name=args.xlsb_python
  wr=xw.Book(r'./'+ file_name)
  wr2=xw.Book(r'./Gold_'+ file_name)

# Read the inputs and outputs from the gold file
sht_inp2=wr2.sheets['Inputs']
input_old=pd.DataFrame(sht_inp2.range('Inputs_table').value,columns=['Parameter', 'Unit', 'value'])
#print (np.array(input_old.iloc[:,2]))
sht_out2=wr2.sheets['Outputs']
output_old=pd.DataFrame(sht_out2.range('Outputs_table').value,columns=['Parameter', 'Unit', 'value'])

# Write the inputs to the existing file
sht_inp=wr.sheets['Inputs']
# store the inputs to be resoted
input_restore=sht_inp.range('Inputs_table').value
input_new=pd.DataFrame(sht_inp.range('Inputs_table').value, columns=['Parameter', 'Unit', 'value'])
sht_inp.range('Inputs_table').value=sht_inp2.range('Inputs_table').value
sht_out=wr.sheets['Outputs']
output_new=pd.DataFrame(sht_out.range('Outputs_table').value,columns=['Parameter', 'Unit', 'value'])
# Pass for T while Failure for F
T=0
F=0
for i in range (len(output_new.iloc[:,2])):
    #(check if error is less than 0.1% for each output)
    if abs(output_new.iloc[i,2]-output_old.iloc[i,2])/output_old.iloc[i,2]<0.001:
        T=T+1
        print ('Test passed for calculating'+str(output_new.iloc[i,0]))
    else:
        F=F+1
        print ('Test failed for calculating'+str(output_new.iloc[i,0]))

sht_inp.range('Inputs_table').value=input_restore

if F==0:
  wr.save()
  wr.close()
  wr2.close()
  xw.App().quit()
  sys.exit(0)
else:
  wr.save()
  wr.close()
  wr2.close()
  xw.App().quit()
  sys.exit(1)
