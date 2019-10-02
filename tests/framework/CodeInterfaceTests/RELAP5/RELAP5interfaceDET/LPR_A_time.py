def evaluate(self):
    self.N_LP = min(self.N_LPR,self.N_LPI)
    if self.N_LP==0:
        return 86400.0
    elif self.N_LP==1:
        return self.LPR_A_time
    elif self.N_LP==2:
        return self.LPR_A_time
