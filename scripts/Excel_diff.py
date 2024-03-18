import os
import json
import sys
import xlwings as xw
import time
from pathlib import Path # Core Python Module

def compareExcelfiles(file1, file2):
  """
    Compares two excel files sheet-by-sheet and row-by-row.
    @ In, file1, str, filename of the initial excel file
    @ In, file2, str, filename of the updated excel file
    @ Out, None
  """
  ## Compare the excel sheets cell by cell
  # the time checks can be enable by the developers
  #start=time.time()
  intialVersion= Path.cwd()/file1
  updatedVersion=Path.cwd()/file2
  fileDir= os.path.dirname(file1)
  fileName=os.path.basename(file1)
  with xw.App(visible=False) as app:
    initialWb2=app.books.open(intialVersion)
    initialWb2.save(fileDir+"/new_"+fileName)
    initialWb2.close()
    intialVersion2=Path.cwd()/(fileDir+"/new_"+fileName)
    numSheet=len(app.books.open(updatedVersion).sheet_names)
    f = open(fileDir+"/DiffResults.txt", mode="wt",encoding="utf-8")
    #read=time.time()
    for i in range (numSheet):
      initialWb=app.books.open(intialVersion2)
      initialWs=initialWb.sheets(i+1)
      updatedWb=app.books.open(updatedVersion)
      updatedWs=updatedWb.sheets(i+1)
      f.write("["+str(initialWb.sheet_names[i])+"]")
      f.write("\n")
      # print (updated_ws.used_range)
      for cell in updatedWs.used_range:
        OV= initialWs.range((cell.row,cell.column)).value
        OF= initialWs.range((cell.row,cell.column)).formula
        if cell.formula!= OF or cell.value!= OV:
          # Print the differences in a format you prefer
          f.write("Diff_values(row, column, new value, old value, new formula, old formula):")
          f.write(str((cell.row, cell.column,cell.value, OV, cell.formula,OF)))
          f.write("\n")
        #startCheck=time.time()
    #end=time.time()
    # f.write("Read time: ")
    # f.write(str(read-start))
    # f.write("\n")
    # f.write("Loop Checking Time:")
    # f.write(str(startCheck-read))
    # f.write("\n")
    # f.write("Save time: ")
    # f.write(str(end-startCheck))
    # f.write("\n")
    f.close()
    # print ("Read",read-start)
    # print ("Loop",startCheck-read)
    # print ("Save",end-startCheck)
  os.remove (fileDir+"/new_"+fileName)
if __name__ == "__main__":
  #print (sys.argv[1], sys.argv[2])
  compareExcelfiles(sys.argv[1], sys.argv[2])
