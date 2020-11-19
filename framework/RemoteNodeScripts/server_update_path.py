#!/usr/bin/env python3
import sys, os, subprocess
# get working dir
workingDir=sys.argv[1]
# change working dir
os.chdir(workingDir)
# output
output=sys.argv[2]
# python path
pythonpath=sys.argv[3]
#create an output file
out = open(output, "a")
print("working dir:", workingDir, file=out)
if "PYTHONPATH" in  os.environ:
  olderPath = os.environ["PYTHONPATH"].split(os.pathsep)
  newPath = pythonpath.split(os.pathsep)
  os.environ["PYTHONPATH"] = os.pathsep.join(list(set(olderPath+newPath)))
else:
  os.environ["PYTHONPATH"] = pythonpath
print("new pythonpath :", os.environ["PYTHONPATH"], file=out)
#Close standard in, out, and error
# Note if not done, ssh will wait until these close
for f in [0,1,2]:
  os.close(f)

