# TSA Module
The TSA module houses algorithms which are helpful for characterizing and generating time series signals.
It is designed to be modular to allow users to mix and match algorithms to best suit their data and goals.

The module is diagrammed below:
<!-- TODO: Add module diagram here -->

Three base classes define the interface and capabilities of each TSA algorithm: TimeSeriesCharacterizer, TimeSeriesGenerator, and TimeSeriesTransformer.
Each of these in turn extends the TimeSeriesAnalyzer class.
The following subsections describe the functionality contributed by each base class.

The purpose of this document is to explain the structure of this module.
Algorithm-specific documentation and examples are provided in the RAVEN User Manual.

## TimeSeriesAnalyzer
Provides an interface common to all TSA module algorithms.
This includes tasks like checking the capability of the subclass instance through the `canCharacterize()` and `canGenerate()` class methods.
The `getResidual()` and `getComposite()` methods are also housed here but may warrant migration to another base class. <!-- TODO: update this line if necessary -->
Note that these both call the `generate()` method which is only implemented for `TimeSeriesGenerator` instances.

Utility methods like `getClusteringValues()` and `setClusteringValues()` are also contained here.
I need to check where and how these are used! <!-- TODO -->

## TimeSeriesCharacterizer
Algorithms which inherit from `TimeSeriesCharacterizer` seek to characterize a time series signal in some way.
This is evident by the algorithm being stateful and needing to be fit to the data.
Characterizing algorithms generally seek to fit one or more parameters of a model to the data.
`TimeSeriesCharacterizer` implements a `characterize()` method where this happens.

This class is also used to define the limitations of the characterization algorithm.
For example, this class has a `_acceptsMissingValues` class attribute which specifies if the characterization algorithm is capable of handling missing values in the time series, represented as NaNs.

## TimeSeriesGenerator
Generative algorithms are those which can produce a time series signal from a class state without requiring an input signal.
This capability is accessed through the `generate()` method.
Note that generation typically does not exist in isolation; a model must characterize a signal (`TimeSeriesCharacterizer`) before having a model suitable

## TimeSeriesTransformer
Transformers, as their name implies, applies a transformation to the input data.
Specifically, the purpose of these transformers is not to generate or to characterize, but to simply apply a transformation.
They may, however, contain useful parameters for characterization.
Key to the `TimeSeriesTransformer` base class is the `transform()`/`inverseTransform()` interface for
