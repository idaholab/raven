# In this script we will setup an external function to calculate k-calc. We will
# perturb x and see what kind of clustering we get. We will not do anything
# "so-called" complicated. We will just take analytical results from Duderstadt
# and Martin Ch-5 to setup problem. We want two clear distinctions -
# super critical and subcritical.

import numpy as np;

sig_tr = 3.62e-2;       # Transport XS
sig_a = 0.1532;         # Absorption XS
nu_sig_f = 0.1570;      # Nu*Fission XS
rel_abs = 1.0;          # Relative Absorption
D = 1/(3*sig_tr);       # Diffusion Coefficient
k_inf = nu_sig_f/sig_a; # Multiplication factor in Infinite Media
Bm_sq = (nu_sig_f - sig_a)/D;   # Material Buckling
ext_dis = 19.6;         # Extrapolation Distance
L_sq = D/sig_a;
length = 154.587;        # ~ Critical Slab lenght

Bg_sq = (np.pi/length)**2;
k = (nu_sig_f/sig_a)/(1 + (L_sq)*(Bg_sq))

print(k);
