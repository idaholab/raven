# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(3,0.5) = 0
# parameter range is -4.5 <= x,y <= 4.5

def evaluate(x,y):
  return (1.5 - x + x*y)**2 + (2.25 - x + x*y*y)**2 + (2.625 - x + x*y*y*y)**2

def run(self,Inputs):
  if 0.95 < self.x < 1.0 and 1.5 < self.y < 1.61:
    print("Expected failure for testing ... x:"+str(self.x)+" | y:"+str(self.y))
    raise Exception("expected failure for testing")
  self.ans = evaluate(self.x,self.y)
