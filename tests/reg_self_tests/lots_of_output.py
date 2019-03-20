from __future__ import division, print_function, absolute_import

import sys

s = "*"*1024
for i in range(256):
  for j in range(1024):
    print(i,j,s)

sys.exit(0)
