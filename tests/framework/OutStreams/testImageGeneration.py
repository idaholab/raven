"""
Created on 4/27/16

@author: maljdan

<TestInfo>
  <name>framework.image_generation_raw</name>
  <author>maljdan</author>
  <created>2016-04-27</created>
  <classesTested> </classesTested>
  <description>
     This test the online generation of plots (colorbar plot).
     It can not be considered part of the active code but of the regression test
     system.

     This test will use ImageMagick's "compare" utility in order to determine if
     two image files are within some amount of tolerance. This script operates
     by executing raven on the input file and then using compare to determine
     if the gold file and the generated file are near identical. If they are
     near enough to identical this test will report a "pass," otherwise it will
     return a "fail."
  </description>
  <revisions>
    <revision author="maljdan" date="2016-05-04">Fixing the test for the compare executable to test the gold image against itself, if this returns a non-zero code, then the version of imageMagick cannot be used to get a valid difference. Also, I am removing the difference image and instead doing null: to remove the output file when using compare.</revision>
    <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
  </revisions>
</TestInfo>

(possibly promote to a diff class at some point, but relies on an external
 application, namely ImageMagick)
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
inputFile = 'imageGeneration_png.xml'
testImage = os.path.join('plot','1-test_scatter.png')
goldImage = os.path.join('gold',testImage)

retCode = subprocess.call(['python','../../../framework/Driver.py',inputFile])

if retCode == 0:
  proc = subprocess.Popen(['compare', '-metric', differenceMetric, '-fuzz',fuzzAmount, testImage,goldImage,'null:'],stderr=subprocess.PIPE)
  retCode = int(proc.stderr.read())

sys.exit(retCode)
