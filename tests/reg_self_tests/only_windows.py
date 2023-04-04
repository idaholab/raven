"""
  this only runs successfully in windows
"""

import platform
import sys

if platform.system().lower() == 'windows':
  sys.exit(0)

#Otherwise fail
sys.exit(-1)
