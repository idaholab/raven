#!/usr/bin/env python3
# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on Nov 15, 2020

@author: alfoa
"""
import sys, os, subprocess, re
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
if re.search("lib.python", pythonpath):
  #strip out python libraries from path
  #XXX ideally, this would have a way to tell if
  # the paths we are stripping are the real builtin python paths
  # instead of just using a regular expression
  splitted=pythonpath.split(os.pathsep)
  newpath = []
  for part in splitted:
    if not re.search("lib.python", part):
      newpath.append(part)
  pythonpath = os.pathsep.join(newpath)
  print("adjusted pythonpath:",pythonpath)
os.environ["PYTHONPATH"] = pythonpath

#Close standard in, out, and error
# Note if not done, ssh will wait until these close
for f in [0,1,2]:
  os.close(f)

out.close()
subprocess.Popen(command)
