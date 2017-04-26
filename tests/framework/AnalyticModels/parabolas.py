#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Given a set of lengths, will compute the longest diagonal in the hyperrectangle created by using each
#   of these lengths as measurements of orthogonal axes.
#

def run(self,Input):
  """
    Method require by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  self.ans = -sum(l*l for l in Input.values())
