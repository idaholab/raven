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
# command
command=sys.argv[4:]
#create an output file
out = open(output, "a")
print("working dir:", workingDir, file=out)
print("command    :", command, file=out)
print("pythonpath :", pythonpath, file=out)
os.environ["PYTHONPATH"] = pythonpath

#Close standard in, out, and error
# Note if not done, ssh will wait until these close
for f in [0,1,2]:
  os.close(f)

subprocess.Popen(command)
