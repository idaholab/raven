# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(3,0.5) = 0
# parameter range is -4.5 <= x,y <= 4.5

def evaluate(x,y):
  return (1.5 - x + x*y)**2 + (2.25 - x + x*y*y)**2 + (2.625 - x + x*y*y*y)**2

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

