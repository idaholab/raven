


def constrain(self):
  if (self.x-1.)**2 + self.y**2 < 1.0:
    return False
  else:
    return True
