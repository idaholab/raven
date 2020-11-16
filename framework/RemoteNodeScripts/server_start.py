#!/usr/bin/env python3
import sys, os, subprocess

output=sys.argv[1]
command=sys.argv[2:]
#create an output file
out = open(output, "a")
print("command", command, file=out)

#Close standard in, out, and error
# Note if not done, ssh will wait until these close
for f in [0,1,2]:
  os.close(f)

subprocess.Popen(command)
