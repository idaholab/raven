def __residuumSign(self):
  print('SYSTEM FAILURE IS ' + str(self.FAIL))
  if self.FAIL == 1: return -1.0
  else                     : return 1.0
