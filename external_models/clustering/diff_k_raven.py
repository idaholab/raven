# In this script we will try to classify subcritical and super critical.

import numpy as np
import copy

def initialize(self, runInfoDict, inputFiles):

    print('Example classify k \n\n\n');
    return

def createNewInput(self, myInput, samplerType, **Kwargs): return Kwargs['SampledVars']

# Define the run function - this will calculate k based on length input

def run(self, Input):

    sig_tr = 3.62e-2;       # Transport XS
    sig_a = 0.1532;         # Absorption XS
    nu_sig_f = 0.1570;      # Nu*Fission XS
    rel_abs = 1.0;          # Relative Absorption
    D = 1/(3*sig_tr);       # Diffusion Coefficient
    k_inf = nu_sig_f/sig_a; # Multiplication factor in Infinite Media
    Bm_sq = (nu_sig_f - sig_a)/D;   # Material Buckling
    ext_dis = 19.6;         # Extrapolation Distance
    L_sq = D/sig_a;
    self.length = copy.deepcopy(154.5875 + Input['l0']);        # ~ Critical Slab lenght
    Bg_sq = (np.pi/self.length)**2;
    self.k = (nu_sig_f/sig_a)/(1 + (L_sq)*(Bg_sq))

    print(self.k);
