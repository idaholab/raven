import os
import json
import sys
import xlwings as xw
import time
from pathlib import Path # Core Python Module

def compare_json(file1, file2):
    # Your comparison logic here
    ## Compare the excel sheets cell by cell
    start=time.time()
    intial_version= Path.cwd()/file1
    updated_version=Path.cwd()/file2
    with xw.App(visible=False) as app:
        initial_wb_2=app.books.open(intial_version)
        initial_wb_2.save("new_"+file1)
        initial_wb_2.close()
        intial_version_2=Path.cwd()/("new_"+file1)
        num_sheet=len(app.books.open(updated_version).sheet_names)
        f = open("Diff_Results.txt", mode="wt",encoding="utf-8")
        read=time.time()
        for i in range (num_sheet):
            initial_wb=app.books.open(intial_version_2)
            initial_ws=initial_wb.sheets(i+1)
            updated_wb=app.books.open(updated_version)
            updated_ws=updated_wb.sheets(i+1)
            f.write("["+str(initial_wb.sheet_names[i])+"]")
            f.write("\n")
            # print (updated_ws.used_range)
            for cell in updated_ws.used_range:
                OV= initial_ws.range((cell.row,cell.column)).value
                OF= initial_ws.range((cell.row,cell.column)).formula
                if cell.formula!= OF or cell.value!= OV:       
                    # Print the differences in a format you prefer
                    f.write("Diff_values(row, column, new value, old value, new formula, old formula):")
                    f.write(str((cell.row, cell.column,cell.value, OV, cell.formula,OF)))
                    f.write("\n")
                    
            start_check=time.time()
    end=time.time()
    os.remove ("new_"+file1)
    f.write("Read time: ")
    f.write(str(read-start))
    f.write("\n")
    f.write("Loop Checking Time:")
    f.write(str(start_check-read))
    f.write("\n")
    f.write("Save time: ")
    f.write(str(end-start_check))
    f.write("\n")
    f.close()
    print ("Read",read-start)
    print ("Loop",start_check-read)
    print ("Save",end-start_check)
if __name__ == "__main__":
    #print (sys.argv[1], sys.argv[2])
    compare_json(sys.argv[1], sys.argv[2])