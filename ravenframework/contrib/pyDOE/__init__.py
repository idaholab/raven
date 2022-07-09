"""
================================================================================
pyDOE: Design of Experiments for Python
================================================================================

This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    
    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by 
Abraham Lee.
"""

# from __future__ import absolute_import

__author__ = 'Abraham Lee'
__version__ = '0.3.6'

from .doe_box_behnken import *
from .doe_composite import *
from .doe_factorial import *
from .doe_lhs import *
from .doe_fold import *
from .doe_plackett_burman import *
from .var_regression_matrix import *

