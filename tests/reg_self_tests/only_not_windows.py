"""
  this only runs successfully in when not in windows
"""

import platform
import sys

if platform.system().lower() == 'windows':
  sys.exit(-1)

#Otherwise succeed
sys.exit(0)
