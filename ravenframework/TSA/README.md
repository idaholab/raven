# TSA Module
The TSA module houses algorithms which are helpful for characterizing and generating time series signals.
It is designed to be modular to allow users to mix and match algorithms to best suit their data and goals.
Each algorithm in this module inherits from one or more base classes which define the interface and capabilities of the algorithm.
Broadly, an algorithm can be any combination of a characterizer, generator, and transformer, with each of these categorizations corresponding to a base class.
- Characterizers map a time series signal to a vector of characteristic features.
- Generators produce a signal, either deterministically or stochastically.
- Transformers map one signal space to another through some transformation function.

## Base Classes
Three base classes define the interface and capabilities of each TSA algorithm: `TimeSeriesCharacterizer`, `TimeSeriesGenerator`, and `TimeSeriesTransformer`.
Each of these in turn extends the `TimeSeriesAnalyzer` class.
The following subsections briefly describe the functionality contributed by each base class.
The purpose of this document is to explain the structure of this module and the various base classes used to construct time series algorithms to give guidance to future users and developers on which base classes are appropriate for new TSA algorithms.
Algorithm-specific documentation and examples are provided in the RAVEN User Manual.

### TimeSeriesAnalyzer
Provides an interface common to all TSA module algorithms.
This class serves exclusively as a base class for the `TimeSeriesCharacterizer`, `TimeSeriesGenerator`, and `TimeSeriesTransformer` classes.
This class should never be directly instatiated or used as a base class for a new algorithm!

### TimeSeriesCharacterizer
Algorithms which inherit from `TimeSeriesCharacterizer` seek to characterize a time series signal by mapping the signal to some vector of characteristic features.
This often takes the form of fitting some number of parameters to the data and using those parameters to form the feature vector.
The characteristic parameters for a characterization algorithm are listed in the `_features` class attribute.
Note that while a model may need to be fit to the data, the resulting model parameters may not be "useful" for characterization!
Only inherit from this class if the model parameters are useful for characterization tasks.

### TimeSeriesGenerator
Algorithms which generate a signal are `TimeSeriesGenerator` instances.
This generation may be either deterministic or stochastic, distinguished by the value of the `_isStochastic` class attribute.

### TimeSeriesTransformer
Transformers, as their name implies, apply a transformation to the input data.
Specifically, the purpose of these transformers is not to generate a new time series signal or to characterize the signal, but to simply map the input signal to some other signal through a transformation function.
The `TimeSeriesTransformer` class is also responsible for computing model residuals during the fitting process and combining one or more signals into a single signal during the generation process.

## TSA Algorithm Inheritance
The following table shows which base classes each currently implemented TSA algorithm inherits from.

| Algorithm              | Transformer | Generator | Characterizer |
|------------------------|:-----------:|:---------:|:-------------:|
| `ARMA`                 |   &check;   |  &check;  |    &check;    |
| `Fourier`              |   &check;   |  &check;  |    &check;    |
| `PolynomialRegression` |   &check;   |  &check;  |    &check;    |
| `RWD`                  |             |           |    &check;    |
| `Wavelet`              |   &check;   |  &check;  |               |
| `OutTruncation`        |   &check;   |           |               |
| `ZeroFilter`           |   &check;   |           |               |
| `MaxAbsScaler`         |   &check;   |           |    &check;    |
| `MinMaxScaler`         |   &check;   |           |    &check;    |
| `StandardScaler`       |   &check;   |           |    &check;    |
| `RobustScaler`         |   &check;   |           |    &check;    |
| `LogTransformer`       |   &check;   |           |               |
| `ArcsinhTransformer`   |   &check;   |           |               |
| `TanhTransformer`      |   &check;   |           |               |
| `SigmoidTransformer`   |   &check;   |           |               |
| `QuantileTransformer`  |   &check;   |           |               |
