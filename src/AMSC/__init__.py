"""
The AMSC module includes the approximate Morse-Smale complex code and all of its
associated visualization views.

Created on January 11, 2016
@author: maljdp
"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from contrib.AMSC import AMSC_Object' outside of this submodule
from .AMSC_Object import AMSC_Object
# from .MainWindow import MainWindow

# We should not really need this as we do not use wildcard imports
__all__ = ['AMSC_Object']
