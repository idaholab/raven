def __residuumSign(self):
  returnValue = 1.0
  if self.A  <= 0.4:
    returnValue = -1.0
  return returnValue
