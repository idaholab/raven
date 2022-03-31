# Raven
Risk Analysis Virtual Environment

RAVEN is designed to perform parametric and probabilistic analysis based on the response of complex system codes. RAVEN is capable of investigating the system response as well as the input space using Monte Carlo, Grid, or Latin Hyper Cube sampling schemes, but its strength is focused toward system feature discovery, such as limit surfaces, separating regions of the input space leading to system failure, using dynamic supervised learning techniques. RAVEN includes the following major capabilities:

- Sampling of codes for uncertainty quantification and reliability analyses
- Generation and use of reduced-order models (also known as surrogate)
- Data post-processing (time dependent and steady state)
- Time dependent and steady state, statistical estimation and sensitivity analysis (mean, variance, sensitivity coefficients, etc.).

The RAVEN statistical analysis framework can be employed for several types of applications:

- Uncertainty Quantification
- Sensitivity Analysis / Regression Analysis
- Probabilistic Risk and Reliability Analysis (PRA)
- Data Mining Analysis
- Model Optimization

RAVEN provides a set of basic and advanced capabilities that ranges from data generation, data processing and data visualization.

## Computing environment

- Parallel computation capabilities (multi-thread and multi-core)
- Supported operating systems: MAC, Linux and Windows
- Workstation and high performance computing (HPC) systems

## Forward propagation of uncertainties

- MonteCarlo sampling
- Grid sampling
- Stratified Sampling
- Factorial design
- Response surface design
- Generalized Polynomial Chaos (gPC) with sparse grid collocation (SGC)
- Generalized Polynomial Chaos (gPC) with sparse grid collocation (SGC) using the High Dimensional Model Representation expansion (HDMR)

- General combination of the above sampling strategies

## Advance sampling methods

- Moment driven adaptive gPC using SGC
- Sobol index driven HDMR integrated using SGC over gPC basis
- Adaptive sampling for limit surface finding (surrogate and multi grid based accelerations)
- Dynamic event tree-based sampling (Dynamic Event Trees, Hybrid Dynamic Event Trees, Adaptive Dynamic Event Trees, Adaptive Hybrid Dynamic Event Trees)

## Creation and use of reduced order models

- Support Vector Machine-based surrogates
- Gaussian process models
- Linear models
- Multi-class classifiers
- Decision trees
- Naive Bayes
- Neighbors classifiers and regressors
- Multi-dimensional interpolators
- High dimension model reduction (HDMR)
- Morse-Smale complex

## Model capabilities

- Generic interface with external codes
- Custom code interfaces (third-party software(s) currently available:
    - [RELAP5-3D](https://relap53d.inl.gov/SitePages/Home.aspx)
    - [MELCOR](https://melcor.sandia.gov/about.html)
    - [MAAP5](https://www.fauske.com/nuclear/maap-modular-accident-analysis-program)
    - [MOOSE-BASED Apps](https://mooseframework.inl.gov/)
    - [SCALE](https://www.ornl.gov/onramp/scale-code-system)
    - [SERPENT](http://montecarlo.vtt.fi/)
    - [CTF - COBRA TF](https://www.ne.ncsu.edu/rdfmg/cobra-tf/)
    - [SAPHIRE](https://saphire.inl.gov/)
    - [MODELICA](https://www.modelica.org/modelicalanguage)
    - [DYMOLA](https://www.3ds.com/products-services/catia/products/dymola/)
    - [BISON](https://bison.inl.gov/SitePages/Home.aspx)
    - [RATTLESNAKE](https://rattlesnake.inl.gov/SitePages/Home.aspx)
    - [MAMMOTH](https://moose.inl.gov/mammoth/SitePages/Home.aspx)
    - [GOTHIC](http://www.numerical.com/products/gothic/gothic_all.php)
    - [PHISICS](https://modsimcode.inl.gov/SitePages/Home.aspx)
    - [NEUTRINO](http://www.neutrinodynamics.com/)
    - [RAVEN running itself](https://raven.inl.gov/SitePages/Overview.aspx)

- Custom ad-hoc external models (build in python internally to RAVEN)

## Data Post-Processing capabilities

- Data clustering
- Data regression
- Data dimensionality Reduction
- Custom generic post-processors
- Time-dependent data analysis (statistics, clustering and time warping metrics)
- Data plotting

## Model parameter optimization

- Simultaneous perturbation stochastic approximation method

## Data management

- Data importing and exporting
- Databases creation

More information on this project is available at the [RAVEN website](https://raven.inl.gov/SitePages/Overview.aspx).

This project is supported by [Idaho National Laboratory](https://www.inl.gov/).

### Other Software
[Idaho National Laboratory](https://www.inl.gov/) is a cutting edge research facility which is a constantly producing high quality research and software. Feel free to take a look at our other software and scientific offerings at:

[Primary Technology Offerings Page](https://www.inl.gov/inl-initiatives/technology-deployment)

[Supported Open Source Software](https://github.com/idaholab)

[Raw Experiment Open Source Software](https://github.com/IdahoLabResearch)

[Unsupported Open Source Software](https://github.com/IdahoLabCuttingBoard)


### License

Files in crow/contrib, src/contrib and framework/contrib are third party libraries that are not part of Raven and are provided here for covenience. These are under their own, seperate licensing which is described in those directories.

Raven itself is licensed as follows:

Copyright 2016 Battelle Energy Alliance, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
