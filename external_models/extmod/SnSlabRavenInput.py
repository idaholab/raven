# In this script we will define the SN Slab function and make it talk to raven
# for post processing - Basic Statistics.

#===============================================================================
# In this script we will use NUMPY to do an SN code and use NDA to accelerate
#===============================================================================

import numpy as np
#import pylab as mp
import copy

# First we define problem parameters - the ones that are not perturbed

def initialize(self, runInfoDict, inputFiles):

    print('Example Slab Transport \n\n\n\n');
    # length = 1.0;                           # Length
    # order = 2;                              # SN order
    # cells = 3;                             # Number of cells
    # BC = np.array([1, 1]);                  # Boundary conditions - 0 = Ref; 1 = Vac
    # src = 2*np.ones(cells);                 # Internal source
    # tol = 1e-6;
    # sigma_t = 1*np.ones(cells);             # Total cross section
    # itct = 0;                                # Iteration count
    # eps = 1;                                 # Initialalize error
    # ang_flux = np.zeros((cells, order));     # Initialize angular flux array
    # ang_flux_e = np.zeros((cells+1, order)); # Edge value of ang_flux - will need this when we do holo
    # self.flux = np.zeros(cells);	                 # Scalar flux
    # flux_old = np.zeros(cells); 	         # Old flux
    # flux_oo = np.zeros(cells);
    # ang_flux_rb = np.zeros(order);	         # Boundary values on right - used for doing reflecting bc
    # ang_flux_lb = np.zeros(order); 	         # Boundary values on left
    # delta_x = (length/cells)*np.ones(cells); # Delta_X vector can be varies but here we choose same dx

    return

def createNewInput(self,myInput,samplerType,**Kwargs): return Kwargs['SampledVars']

# Define the run function - this does source iteration

def run(self, Input):

    length = 1.0;                           # Length
    order = 2;                              # SN order
    cells = 3;                             # Number of cells
    BC = np.array([1, 1]);                  # Boundary conditions - 0 = Ref; 1 = Vac
    src = 2*np.ones(cells);                 # Internal source
    tol = 1e-6;
    sigma_t = 1*np.ones(cells);             # Total cross section
    itct = 0;                                # Iteration count
    eps = 1;                                 # Initialalize error
    ang_flux = np.zeros((cells, order));     # Initialize angular flux array
    ang_flux_e = np.zeros((cells+1, order)); # Edge value of ang_flux - will need this when we do holo
    self.flux = np.zeros(cells);	                 # Scalar flux
    flux_old = np.zeros(cells); 	         # Old flux
    flux_oo = np.zeros(cells);
    ang_flux_rb = np.zeros(order);	         # Boundary values on right - used for doing reflecting bc
    ang_flux_lb = np.zeros(order); 	         # Boundary values on left
    delta_x = (length/cells)*np.ones(cells); # Delta_X vector can be varies but here we choose same dx


    # First define the sampled parameters

    self.c = 0.1*np.ones(cells);                 # Scattering ratio
    self.c0 = 0.2;                                      # Initializing c0
    # Now extract a value from a normal distribution mean 0.5 std 0.3 and add it
    # to the scattering ratio
    for step in range(cells):
        self.c[step] = copy.deepcopy(self.c[step]+Input['c0']);
    self.sigma_s = self.c*sigma_t;                 # Scattering cross section
    print(self.sigma_s);
    # Now we generate Gauss-Legendre weights and angles

    mu, wt = np.polynomial.legendre.leggauss(order);    # Numpy has an inbuilt function for generating gauss quadrature

    # Now that all inputs have been taken, start a while loop for source iteration.

    while tol < eps:

        itct = itct + 1;
        q = (src + self.sigma_s*self.flux)/2;     # Generate source

        for o in range(0, order):

            # We will begin with negative angles - sweep from right to left since the
            # quadrature function outputs negative angles first.

            if mu[o]<0:

                # First we do boundary conditions

                if BC[1] == 0: # This means reflecting boundary

                    ang_flux_right = 0; #ang_flux_rb[order - o + 1]; # Because ang_flux_rb(order-o+1) = outgoing angular flux for mu(-o)

                else: # This means vacuum boundary

                    ang_flux_right = 0;


                ang_flux_e[cells, o] = ang_flux_right;

                for k in range(cells-1, -1, -1):

                    numerator_neg = ang_flux_right + ((delta_x[k]*q[k])/(2*abs(mu[o])));
                    denominator_neg = 1 + ((delta_x[k]*sigma_t[k])/(2*abs(mu[o])));
                    ang_flux[k, o] = numerator_neg/denominator_neg;
                    ang_flux_right = 2*ang_flux[k, o] - ang_flux_right;
                    ang_flux_e[k, o] = ang_flux_right;

                ang_flux_lb[o] = ang_flux_right; # No need to check if it is at the
                                                 # boundary since the last term we
                                                 # calculate is at the left boundary


            else: # Now we do positive mu - sweep from left to right

                 # First we do boundary conditions

                if BC[0] == 0: # This means reflecting boundary

                    ang_flux_left = 0; # ang_flux_lb[order - o + 1]; # Because ang_flux_rb(order-o+1) = outgoing angular flux for mu(-o)

                else: # This means vacuum boundary

                    ang_flux_left = 0;


                ang_flux_e[0, o] = ang_flux_left;

                for k in range(0, cells):

                    numerator_pos = ang_flux_left + ((delta_x[k]*q[k])/(2*abs(mu[o])));
                    denominator_pos = 1 + ((delta_x[k]*sigma_t[k])/(2*abs(mu[o])));
                    ang_flux[k, o] = numerator_pos/denominator_pos;
                    ang_flux_left = 2*ang_flux[k, o] - ang_flux_left;
                    ang_flux_e[k+1, o] = ang_flux_left;

                ang_flux_rb[o] = ang_flux_left; # No need to check if it is at the
                                                 # boundary since the last term we
                                                 # calculate is at the right boundary

        # Now we calculate flux and check for convergence

        self.flux = np.zeros(cells).transpose();

        for o in range(0, order):

            self.flux = self.flux + wt[o]*ang_flux[:, o];

        if itct > 1:

            sig_a = np.linalg.norm(self.flux - flux_old);
            sig_b = np.linalg.norm(flux_old - flux_oo);
            sig = sig_a/sig_b;
            eps = abs(sig_b/(1-sig_a));
            print eps;

            # sp_rad = sig_a/sig_b;
            # print sp_rad;


        flux_oo = flux_old;
        flux_old = self.flux;
    print(self.flux);
    # print(itct);
    # mp.plot(self.flux);
    # mp.show();
