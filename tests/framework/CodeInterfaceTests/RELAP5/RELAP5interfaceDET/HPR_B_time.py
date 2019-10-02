def evaluate(self):
    self.N_HP = min(self.N_HPR,self.N_HPI)
    if self.N_HP==0:
        return 86400.0
    elif self.N_HP==1:
        return 86400.0
    elif self.N_HP==2:
        return self.HPR_B_time
    elif self.N_HP==3:
        return self.HPR_B_time
