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
Created on 8/1/16

@author: maljdan

<TestInfo>
  <name>framework.colorbar_plot</name>
  <author>maljdan</author>
  <created>2016-08-01</created>
  <classesTested> </classesTested>
  <description>
     This test the online generation of plots (colorbar plot).
     It can not be considered part of the active code but of the regression test
     system. This file is a near identical copy of testImageGeneration.py and at
     some point, we should see if the interface can be generalized to handle
     image file diffs more generically. Right now, the number of image file
     tests is small, but in order to be scalable, we need a better solution.
  </description>
  <revisions>
    <revision author="maljdan" date="2016-05-04">Fixing the test for the compare executable to test the gold image against itself, if this returns a non-zero code, then the version of imageMagick cannot be used to get a valid difference. Also, I am removing the difference image and instead doing null: to remove the output file when using compare.</revision>
    <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
  </revisions>
</TestInfo>

(possibly promote to a diff class at some point, but relies on an external
 application, namely ImageMagick) This test was created in reference to issue
#639
"""
import subprocess
import sys
import os


## Some tweakable parameters
differenceMetric = 'ae' ## Careful you may need to parse the output of different
                        ## metrics since they don't all output a single value

fuzzAmount = '5%'       ## This dictates how close pixel values need to be to be
                        ## considered the same

## Make this resuable for an arbitrary pair of images and an arbitary test file
## by passing them all as variables
inputFile = 'test_colorbar.xml'
testImage = os.path.join('plot','colorbarTest.png')
goldImage = os.path.join('gold',testImage)

retCode = subprocess.call(['python','../../../framework/Driver.py',inputFile])

if retCode == 0:
  proc = subprocess.Popen(['compare', '-metric', differenceMetric, '-fuzz',fuzzAmount, testImage,goldImage,'null:'],stderr=subprocess.PIPE)
  retCode = int(proc.stderr.read())

sys.exit(retCode)
