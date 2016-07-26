Overview

RAVEN is a flexible and multi-purpose statistical analysis
framework. It has been developed at Idaho National Laboratory (INL)
since 2012 within the Nuclear Energy Advanced Modeling and Simulation
(NEAMS) and the Light Water Reactor Sustainability (LWRS) Risk
Informed Safety Margin Characterization (RISMC) programs.

Statistical analysis includes the following major capabilities:
- statistical sampling of codes (e.g., Monte Carlo, Latin hypercube
  and grid sampling, and Dynamic Event Trees) for uncertainty
  quantification and reliability analyses
- generation and use of reduced-order models (also known as surrogate
  models or emulators)
- data post-processing
- statistical estimation and sensitivity analysis (e.g., mean,
  variance, sensitivity coefficients and covariance matrix).


Applications

The RAVEN statistical analysis framework can be employed for several
types of applications:
- Uncertainty Quantification
- Sensitivity Analysis
- Probabilistic Risk and Reliability Analysis (PRA)
- Regression Analysis
- Data Mining Analysis
- Model Optimization (currently under development)

Capabilities

RAVEN provides a set of basic and advanced capabilities that ranges
from data generation, data processing and data visualization.

A full set of RAVEN computational capabilities are listed below:

Computing capabilities:
- Parallel computation capabilities (multi-thread and multi-core)
- Supported operating systems: MAC, Linux and Windows
- Workstation and high performance computing (HPC) systems

Multi-steps analyses: RAVEN analyses are performed through a series of
simulation steps. Each simulation step allows the user to perform a
set of basic actions:

- Multi-Run
- Training of a ROM
- Post-Process
- IOStep

More complex analyses are performed by simply assembling and linking a
series of steps listed above.


Creation and use of reduced order models (scikit-learn and CROW library):
- SVM
- Gaussian process models
- Linear models
- Multi-class classifiers
- Decision trees
- Naive Bayes
- Neighbors classifiers and regressors
- Multi-dimensional interpolators
- High dimension model reduction (HDMR)
- Morse-Smale complex

Forward propagation of uncertainties:
- MonteCarlo sampling
- Grid sampling
- Stratified Sampling
- Factorial design
- Response surface design
- Multi-variate analysis
- Generic Polynomial Chaos expansion

Advance sampling methods:
- Sobol index sampling
- Adaptive sampling
- Sparse grid collocation sampling
- Dynamic event trees (DETs)

Model capabilities:
- Generic interface with codes
- Custom code interfaces
- Custom ad-hoc external models

Data Post-Processing capabilities:
- Data clustering
- Data regression
- Data dimensionality Reduction
- Custom generic post-processors
- Time-dependent data analysis
- Data plotting

Model parameter optimization:
- Simultaneous perturbation stochastic approximation method

Data management:
- Data importing and exporting
- Databases creation

Team
Project Manager: Cristian Rabiti @crisr cristian.rabiti@inl.gov

Technical Lead: Andrea Alfonsi @alfoa andrea.alfonsi@inl.gov

Testing, Quality Assurance and Installation: Joshua Cogliati @cogljj joshua.cogliati@inl.gov

Developers
    Andrea Alfonsi @alfoa
    Congjian Wang @wangc
    Diego Mandelli @mandd
    Daniel Maljovec @maljdan
    Joshua Cogliati @cogljj
    Paul Talbot @talbpaul
    Robert Kinoshita @bobk
    Jong Suk Kim @kimj
    Jun Chen @chenj
    Alain Giorla @gioralai

Other Contacts
    RAVEN user list raven-users@inl.gov
    RAVEN service id raven@inl.gov
    RAVEN development list raven-devel@inl.gov


A list of other contributors can be obtained with the following command:
```git shortlog -sn```

See the file LICENSE for copyright and export information.



