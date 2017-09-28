# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(3,0.5) = 0
# parameter range is -4.5 <= x,y <= 4.5

def evaluate(x,y):
  return (1.5 - x + x*y)**2 + (2.25 - x + x*y*y)**2 + (2.625 - x + x*y*y*y)**2

def run(self,Inputs):
  #if abs(self.y - 0.35213920079781236) <= 0.00000001 and abs(self.x - 2.506071482338081) <= 0.00000001 :
  if abs(self.y - -0.875380424612) <= 0.00000001 and abs(self.x - -1.7239478044) <= 0.00000001 :
    print("Expected failure for testing ... x:"+str(self.x)+" | y:"+str(self.y))
    raise Exception("expected failure for testing")
  #if abs(self.y - 0.2236548034966761) <= 0.00000001 and abs(self.x - 2.119304954170401) <= 0.00000001 :
  if abs(self.y - 1.1329985399132454) <= 0.00000001 and abs(self.x - -2.9226157061835902) <= 0.00000001 :
    print("Expected failure for testing (grad point 3_6_1) ... x:"+str(self.x)+" | y:"+str(self.y))
    raise Exception("expected failure for testing(grad point 3_6_1)")
  self.ans = evaluate(self.x,self.y)
