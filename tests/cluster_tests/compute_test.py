#!/usr/bin/env python
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
Simple program that spends about a minute computing.
"""

import time
import sys
import os

inputFilename = sys.argv[2]
root = os.path.splitext(inputFilename)[0]
head,tail = os.path.split(root)
outFile = open(os.path.join(head,"out~"+tail+".csv"),"w")

print(sys.argv)
start = time.time()
print(start)

iterations = 0
#The below number of iterations takes about 20 seconds on sawtooth
while iterations < 100000:
    a = 2**2**2**2**2
    iterations += 1

end = time.time()

print(time.time())

outFile.write("start,end,delta,iterations\n")
outFile.write(str(start)+","+str(end)+","+str(end-start)+","+str(iterations)+"\n")
outFile.close()
