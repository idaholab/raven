#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
#  Combines both tensor_poly and attenuate for simple evaluation
#
import numpy as np
import tensor_poly
import attenuate

def evaluate(inp):
  return tensor_poly.evaluate(inp)

def evaluate2(inp):
  return attenuate.evaluate(inp)

def run(self,Input):
  self.ans  = evaluate (Input.values())
  self.ans2 = evaluate2(Input.values())

#
#  These tests have analytic mean and variance, documented in raven/doc/tests
#
