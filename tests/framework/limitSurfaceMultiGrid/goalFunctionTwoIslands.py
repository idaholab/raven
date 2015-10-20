def __residuumSign(self):
  returnValue = 1.0
  if self.x2 >= 0:
    if self.y4 <= 1 and self.y4 >0.5: returnValue = 1
    else                            : returnValue = -1
  return returnValue
