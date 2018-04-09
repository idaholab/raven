#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Given a set of lengths, will compute the longest diagonal in the hyperrectangle created by using each
#   of these lengths as measurements of orthogonal axes.
#
import numpy as np

#static seed
np.random.seed(42)

def base(values):
  center = values.get('center',0.0)
  return -sum( (l-center)*(l-center) for l in values.values() if str(l) != 'center')+1.

def random(scale=0.5,loc=-1.0):
  return scale*(2.*np.random.rand()+loc)

def evaluate(values):
  ran = random()/10.0
  return base(values) + ran

def run(self,Input):
  """
    Method require by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  #self.ans = evaluate(Input.values())
  self.reg = base(Input)
