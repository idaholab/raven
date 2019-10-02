def evaluate(self):
    if self.N_LPI==0:
        return 86400.0
    elif self.N_LPI==1:
        return 86400.0
    elif self.N_LPI==2:
        return self.LPI_B_time
