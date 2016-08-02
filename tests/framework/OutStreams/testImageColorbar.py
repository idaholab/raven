'''
Created on 8/1/16

@author: maljdan
(possibly promote to a diff class at some point, but relies on an external
 application, namely ImageMagick) This test was created in reference to issue
#639
'''
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
