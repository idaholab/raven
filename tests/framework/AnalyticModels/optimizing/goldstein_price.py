# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# documented in analytic functions

def evaluate(x,y):
  first = 1. + (x + y + 1.)**2 * (19. - 14.*x + 3.*x*x - 14.*y + 6.*x*y + 3.*y*y)
  second = 30. + (2.*x - 3.*y)**2 * (18. - 32.*x + 12.*x*x + 48.*y - 36.*x*y + 27.*y*y)
  return first*second

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

