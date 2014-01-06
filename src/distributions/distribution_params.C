/*
 * distribution_params.C
 *
 * Created on Nov 27, 2013
 *  Author cogljj (Joshua J Cogliati)
 */

#include "distribution_params.h"

template<>
InputParameters validParams<distribution>(){

  InputParameters params = validParams<RavenObject>();

  params.addParam<double>("xMin", -std::numeric_limits<double>::max( ),"Lower bound");
  params.addParam<double>("xMax", std::numeric_limits<double>::max( ),"Upper bound");

   params.addParam< std::vector<double> >("PBwindow", "Probability window");
   params.addParam< std::vector<double> >("Vwindow" , "Value window");

   params.addParam<double>("ProbabilityThreshold", 1.0, "Probability Threshold");

   params.addParam<unsigned int>("seed", _defaultSeed ,"RNG seed");
   params.addRequiredParam<std::string>("type","distribution type");
   params.addParam<unsigned int>("truncation", 1 , "Type of truncation"); // Truncation types: 1) pdf_prime(x) = pdf(x)*c   2) [to do] pdf_prime(x) = pdf(x)+c
   params.addPrivateParam<std::string>("built_by_action", "add_distribution");
   params.addParam<unsigned int>("force_distribution", 0 ,"force distribution to be evaluated at: if (0) Don't force distribution, (1) xMin, (2) Mean, (3) xMax");

   return params;
}

class distribution;

distribution::distribution(const std::string & name, InputParameters parameters):
      RavenObject(name,parameters)
{
   _type=getParam<std::string>("type");
   if(_type != "CustomDistribution"){
      _dis_parameters["xMin"] = getParam<double>("xMin");
      _dis_parameters["xMax"] = getParam<double>("xMax");
   }
   else
   {
     std::vector<double> x_coordinates = getParam<std::vector<double> >("x_coordinates");
     _dis_parameters["xMin"] = x_coordinates[0];
     _dis_parameters["xMax"] = x_coordinates[x_coordinates.size()-1];
     std::vector<double> y_cordinates = getParam<std::vector<double> >("y_coordinates");
     //custom_dist_fit_type fitting_type = static_cast<custom_dist_fit_type>((int)getParam<MooseEnum>("fitting_type"));

     //_interpolation=Interpolation_Functions(x_coordinates,
     //                                       y_cordinates,
     //                                       fitting_type);
   }
      _seed = getParam<unsigned int>("seed");
      _force_dist = getParam<unsigned int>("force_distribution");
      _dis_parameters["truncation"] = double(getParam<unsigned int>("truncation"));

      _dis_vectorParameters["PBwindow"] = getParam<std::vector<double> >("PBwindow");
      _dis_vectorParameters["Vwindow"] = getParam<std::vector<double> >("Vwindow");

      _dis_parameters["ProbabilityThreshold"] = getParam<double>("ProbabilityThreshold");

      _checkStatus = false;
}

distribution::~distribution(){
}

/*
 * CLASS UNIFORM DISTRIBUTION
 */


template<>
InputParameters validParams<UniformDistribution>(){

   InputParameters params = validParams<distribution>();
    
   params.addRequiredParam<double>("xMin", "Distribution lower bound");
   params.addRequiredParam<double>("xMax", "Distribution upper bound");

   return params;
}


UniformDistribution::UniformDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), BasicUniformDistribution(getParam<double>("xMin"), getParam<double>("xMax"))
{
}

UniformDistribution::~UniformDistribution()
{
}


/*
 * CLASS NORMAL DISTRIBUTION
 */

template<>
InputParameters validParams<NormalDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("mu", "Mean");
   params.addRequiredParam<double>("sigma", "Standard deviation");
   return params;
}

NormalDistribution::NormalDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), 
  BasicNormalDistribution(getParam<double>("mu"),getParam<double>("sigma")) {
}

NormalDistribution::~NormalDistribution(){
}


/*
 * CLASS LOG NORMAL DISTRIBUTION
 */

template<>
InputParameters validParams<LogNormalDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("mu", "Mean");
   params.addRequiredParam<double>("sigma", "Standard deviation");
    
   return params;
}

LogNormalDistribution::LogNormalDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), BasicLogNormalDistribution(getParam<double>("mu"), getParam<double>("sigma"))
{
}

LogNormalDistribution::~LogNormalDistribution()
{
}

/*
 * CLASS TRIANGULAR DISTRIBUTION
 */

template<>
InputParameters validParams<TriangularDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("xPeak", "Maximum coordinate");
   params.addRequiredParam<double>("lowerBound", "Lower bound");
   params.addRequiredParam<double>("upperBound", "Upper bound");
   return params;
}

TriangularDistribution::TriangularDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), BasicTriangularDistribution(getParam<double>("xPeak"),getParam<double>("lowerBound"),getParam<double>("upperBound"))
{
}
TriangularDistribution::~TriangularDistribution()
{
}


/*
 * CLASS EXPONENTIAL DISTRIBUTION
 */

template<>
InputParameters validParams<ExponentialDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("lambda", "lambda");
   return params;
}

ExponentialDistribution::ExponentialDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), BasicExponentialDistribution(getParam<double>("lambda"))
{
}
ExponentialDistribution::~ExponentialDistribution()
{
}

/*
 * CLASS WEIBULL DISTRIBUTION
 */

template<>
InputParameters validParams<WeibullDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("k", "shape parameter");
   params.addRequiredParam<double>("lambda", "scale parameter");
   return params;
}

WeibullDistribution::WeibullDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), 
  BasicWeibullDistribution(getParam<double>("k"),getParam<double>("lambda"))
                                                         
{
}

WeibullDistribution::~WeibullDistribution()
{
}

/*
 * CLASS GAMMA DISTRIBUTION
 */

template<>
InputParameters validParams<GammaDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("k", "shape parameter");
   params.addRequiredParam<double>("theta", "scale parameter");
   params.addParam<double>("low",0.0,"low value for distribution");
   return params;
}

GammaDistribution::GammaDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), 
  BasicGammaDistribution(getParam<double>("k"),getParam<double>("theta"),getParam<double>("low"))
                                                         
{
}

GammaDistribution::~GammaDistribution()
{
}

/*
 * CLASS BETA DISTRIBUTION
 */

template<>
InputParameters validParams<BetaDistribution>(){

   InputParameters params = validParams<distribution>();

   params.addRequiredParam<double>("alpha", "alpha parameter");
   params.addRequiredParam<double>("beta", "beta parameter");
   params.addParam<double>("scale",1.0,"scale value for distribution");
   return params;
}

BetaDistribution::BetaDistribution(const std::string & name, InputParameters parameters):
  distribution(name,parameters), 
  BasicBetaDistribution(getParam<double>("alpha"),getParam<double>("beta"),
                        getParam<double>("scale"))
                                                         
{
}

BetaDistribution::~BetaDistribution()
{
}

/*
 * CLASS CUSTOM DISTRIBUTION
 */

// template<>
// InputParameters validParams<CustomDistribution>(){

//    InputParameters params = validParams<distribution>();

//    params.addRequiredParam< vector<double> >("x_coordinates", "coordinates along x");
//    params.addRequiredParam< vector<double> >("y_coordinates", "coordinates along y");
//    MooseEnum fitting_enum("step_left=0,step_right=1,linear=2,cubic_spline=3");
//    params.addRequiredParam<MooseEnum>("fitting_type",fitting_enum, "type of fitting");
//    params.addParam<int>("n_points",3,"Number of fitting point (for spline only)");
//    return params;
// }

// class CustomDistribution;

// CustomDistribution::CustomDistribution(const std::string & name, InputParameters parameters):
//   distribution(name,parameters), 
//   BasicCustomDistribution(getParam<double>("x_coordinates"),
//                           getParam<double>("y_coordinates"),
//                           getParam<MooseEnum>("fitting_type"),
//                           getParam<double>("n_points"))
// {
// }

// CustomDistribution::~CustomDistribution()
// {
// }
