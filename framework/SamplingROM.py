'''
Module for ROM models that require specific sampling sets, e.g. Stochastic Collocation
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import shutil
import numpy as np
from utils import metaclass_insert, returnPrintTag, returnPrintPostTag
import abc
import importlib
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType, Assembler
import PostProcessors #import returnFilterInterface
import Samplers
import Models
from CustomCommandExecuter import execCommand
#Internal Modules End--------------------------------------------------------------------------------

class SamplingROM(ROM,Sampler):
  def __init__(self):
    Sampler.__init__(self)
    ROM.__init__(self)


class StochasticPolynomials(SamplingROM):
  pass
