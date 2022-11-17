/******************************************************************************
 * Software License Agreement (BSD License)                                   *
 *                                                                            *
 * Copyright 2014 University of Utah                                          *
 * Scientific Computing and Imaging Institute                                 *
 * 72 S Central Campus Drive, Room 3750                                       *
 * Salt Lake City, UT 84112                                                   *
 *                                                                            *
 * THE BSD LICENSE                                                            *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * 1. Redistributions of source code must retain the above copyright          *
 *    notice, this list of conditions and the following disclaimer.           *
 * 2. Redistributions in binary form must reproduce the above copyright       *
 *    notice, this list of conditions and the following disclaimer in the     *
 *    documentation and/or other materials provided with the distribution.    *
 * 3. Neither the name of the copyright holder nor the names of its           *
 *    contributors may be used to endorse or promote products derived         *
 *    from this software without specific prior written permission.           *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    *
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   *
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  *
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   *
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          *
 ******************************************************************************/

#ifndef AMSC_H
#define AMSC_H

#include "ngl/ngl.h"

#include <map>
#include <vector>
#include <set>
#include <sstream>
#include <list>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

/**
 * Merge data structure
 * A integer index to describe the parent and saddle point samples, and a
 * floating point value to describe the persistence of a particular point sample
 * labeled as either a local minimum or a local maximum
 */
template<typename T>
struct Merge
{
  Merge() : persistence(-1),saddle(-1),parent(-1) { }

  Merge(T pers, int saddleIdx, int parentIdx)
    : persistence(pers),saddle(saddleIdx),parent(parentIdx) { }

  T persistence;
  int saddle;
  int parent;
};

/**
 * Discrete gradient flow estimation data structure
 * A pair of integer indices describing upward and downward gradient flow from a
 * point sample.
 */
struct FlowPair
{
  FlowPair(): down(-1), up(-1) { }
  FlowPair(int _down, int _up): down(_down), up(_up) { }

  int down;
  int up;
};

/**
 * Approximate Morse-Smale Complex.
 * Stores the hierarchical decomposition of an arbitrary point cloud according
 * to its estimated gradient flow.
 */
template<typename T>
class AMSC
{
 public:
  /* Here are a list of typedefs to make things more compact and readable */
  typedef std::pair<int, int> int_pair;

  typedef typename std::map< int_pair, std::pair<T,int> > map_pi_pfi;
  typedef typename map_pi_pfi::iterator map_pi_pfi_it;

  typedef typename std::map< std::pair<T,int>, int_pair> map_pfi_pi;
  typedef typename map_pfi_pi::iterator map_pfi_pi_it;

  typedef typename std::map< int, Merge<T> > persistence_map;
  typedef typename std::map< int, Merge<T> >::iterator persistence_map_it;

  typedef void (*graphFunction)(ngl::NGLPointSet<T> &points,
                                ngl::IndexType **indices, int &numEdges,
                                ngl::NGLParams<T> params);

  /**
   * Constructor that will decompose a passed in dataset, note the user passes
   * in a list of candidate edges from which it will prune accordingly using ngl
   * @param Xin flattened vector of input data in row-major order
   * @param yin vector of response values in a one-to-one correspondence with
   *        Xin
   * @param names a vector of string names for each dimension in the input data
   *        and the name of the response value which should be at the end of the
   *        vector
   * @param gradientMethod string identifier for what type of gradient
   *        estimation method is used
   * @param maxN integer specifying the maximum number of neighbors to use in
   *        computing/pruning a neighborhood graph
   * @param beta floating point value in the range (0,2] determining the beta
   *        value used if the neighborhood type is a beta-skeleton, otherwise
   *        ignored
   * @param persistenceType string identifier for what type of persistence
   *        computation should be used
   * @param win vector of probability values in a one-to-one correspondence with
   *        Xin
   * @param edgeIndices an optional list of edges specified as a flattened
   *        n-by-2 array to use as the underlying graph structure (will be
   *        pruned by ngl)
   */
  AMSC(std::vector<T> &Xin, std::vector<T> &yin,
       std::vector<std::string> &_names, std::string graph,
       std::string gradientMethod, int maxN, T beta,
       std::string persistenceType, std::vector<T> &win,
       std::vector<int> &edgeIndices, bool verbosity=false);

  /**
   * Returns the number of input dimensions in the associated dataset
   */
  int Dimension();

  /**
   * Returns the number of sample points in the associated dataset
   */
  int Size();

  /**
   * Returns the global maximum value attained by the output of the associated
   * dataset
   */
  T MaxY();

  /**
   * Returns the global minimum value attained by the output of the associated
   * dataset
   */
  T MinY();

  /**
   * Returns MaxY()-MinY()
   */
  T RangeY();

  /**
   * Returns the maximum value attained by a specified dimension of the input
   * space of the associated dataset
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T MaxX(int dim);

  /**
   * Returns the minimum value attained by a specified dimension of the input
   * space of the associated dataset
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T MinX(int dim);

  /**
   * Returns MaxX(dim)-MinX(dim)
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T RangeX(int dim);

  /**
   * Extracts the input values for a specified sample of the associated data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   * @param xi a pointer that will be updated to point at the specified data
   *        sample
   */
  void GetX(int i, T *xi);

  /**
   * Extracts the input value for a specified sample and dimension of the
   * associated data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   * @param j integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T GetX(int i, int j);

  /**
   * Extracts the scalar output value for a specified sample of the associated
   * data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   */
  T GetY(int i);

  /**
   * Returns the index of the minimum sample to which sample "i" flows to at a
   * specified level of persistence simplification
   * @param i integer specifying the unique sample queried
   * @param pers floating point value specifying an optional amount of
   *        simplification to consider when retrieving the minimum index
   */
  int MinLabel(int i, T pers);

  /**
   * Returns the index of the maximum sample to which sample "i" flows to at a
   * specified level of persistence simplification
   * @param i integer specifying the unique sample queried
   * @param pers floating point value specifying an optional amount of
   *        simplification to consider when retrieving the maximum index
   */
  int MaxLabel(int i, T pers);

  /**
   * Returns the string name associated to the specified dimension index
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  std::string Name(int dim);

  /**
   * Returns a list of indices marked as neighbors to the specified sample given
   * given by "index"
   * @param index integer specifying the unique sample queried
   */
  std::set<int> Neighbors(int index);

  /**
   * Returns a formatted string that can be used to determine the merge
   * hierarchy of the topological decomposition of the associated dataset
   */
  std::string PrintHierarchy();

  /**
   * Returns a sorted list of persistence values for this complex
   **/
   std::vector<T> SortedPersistences();

  /**
   * Returns an xml-formatted string that can be used to determine the merge
   * hierarchy of the topological decomposition of the associated dataset
   */
  std::string XMLFormattedHierarchy();

  /**
   * Returns a map where the key represent a minimum/maximum pair
   * ('minIdx,maxIdx') and the value is a list of associated indices from the
   * input data
   * @param persistence floating point value that optionally simplifies the
   *        topological decomposition before fetching the indices.
   */
  std::map< std::string, std::vector<int> > GetPartitions(T persistence);

  /**
   * Returns a map where the key represent a maximum and the value is a list of
   * associated indices from the input data
   * @param persistence floating point value that optionally simplifies the
   *        topological decomposition before fetching the indices.
   */
  std::map< int, std::vector<int> > GetStableManifolds(T persistence);

  /**
   * Returns a map where the key represent a minimum and the value is a list of
   * associated indices from the input data
   * @param persistence floating point value that optionally simplifies the
   *        topological decomposition before fetching the indices.
   */
  std::map< int, std::vector<int> > GetUnstableManifolds(T persistence);

//  std::string ComputeLinearRegressions(T persistence);

 private:
  std::string persistenceType;          /** A string identifier specifying    *
                                         *  how we should compute persistence */
  bool verbose;                     /** A flag used for toggling debug output */

  boost::numeric::ublas::matrix<T> X;                   /** Input data matrix */
  boost::numeric::ublas::vector<T> y;                  /** Output data vector */
  boost::numeric::ublas::vector<T> w;             /** Probability data vector */

  std::vector<std::string> names;    /** Names of the input/output dimensions */

  std::map< int, std::set<int> > neighbors;         /** Maps a list of points
                                                     *  that are neighbors of
                                                     *  the index             */

  std::vector<FlowPair> neighborFlow;         /** Estimated neighbor gradient
                                               * flow first=desc,second = asc */

  std::vector<FlowPair> flow;               /** Local minimum/maximum index to
                                             *  which each point flows from/to
                                             *  first = min, second = max     */

  int globalMinIdx;         /** The index of the overall global minimum point */
  int globalMaxIdx;         /** The index of the overall global maximum point */

  //////////////////////////////////////////////////////////////////////////////
  // Key is my index and the value is the persistence value, extrema index that
  // I merge to, and which saddle I go through
  persistence_map maxHierarchy;       /** The simplification hierarchy for all
                                        * of the maxima                       */

  persistence_map minHierarchy;       /** The simplification hierarchy for all
                                        * of the minima                       */
  //////////////////////////////////////////////////////////////////////////////

  // Private Methods

  /**
   * Helper method that optionally prints a message depending on the verbosity
   * of this object
   * @param text string of the message to display
   */
  void DebugPrint(std::string text);

  /**
   * Helper method that optionally starts a timer and prints a message depending
   * on the verbosity of this object
   * @param t0 reference time_t struct that will be written to
   * @param text string of message to display onscreen
   */
  void DebugTimerStart(time_t &t0, std::string text);

  /**
   * Helper method that optionally stops a timer and prints a message depending
   * on the verbosity of this object
   * @param t0 reference time_t struct that will be used as the start time
   */
  void DebugTimerStop(time_t &t0, std::string text="");

  /**
   * Returns the ascending neighbor of the sample specified by index
   * @param index integer specifying the unique sample to query
   */
  int ascending(int index);

  /**
   * Returns the descending neighbor of the sample specified by index
   * @param index integer specifying the unique sample to query
   */
  int descending(int index);

  /**
   * Computes and internally stores the neighborhood graph used for
   * approximating the gradient flow behavior
   * @param data a matrix of input points upon which we will construct a
   *        neighborhood graph.
   * @param edgeIndices a vector of nonegative integer indices representing
   *        a flattened array of pre-computed edge indices to use for pruning.
   * @param nn a matrix of integers storing the edge indices for each sample.
   * @param dists a matrix of floating point values storing the edge distances
   *        for each sample.
   * @param type a string specifying the type of neighborhood graph to build.
   * @param beta floating point value used for the beta skeleton computation.
   * @param kmax an integer representing the maximum number of k-nearest
   *        neighbors to consider.
   * @param connect a boolean specifying whether we should enforce the graph
   *        to be a single connected component (will do a brute force search of
   *        the point samples and connect the closest points between separate
   *        components until everything is one single component)
   */
  void computeNeighborhood(std::vector<int> &edgeIndices,
                           boost::numeric::ublas::matrix<int> &nn,
                           boost::numeric::ublas::matrix<T> &dists,
                           std::string type, T beta, int &kmax,
                           bool connect=false);

  /**
   * Helper function to be called after a neighborhood has been constructed in
   * order to connect the entire domain into a single connected component.
   * This was only necessary in Sam's visualization, in theory it is fine if the
   * data is disconnected.
   */
  void ConnectComponents(std::set<int_pair> &ngraph, int &maxCount);

  // Gradient estimation Methods

  /**
   * Function that will delegate the gradient estimation to the appropriate
   * method
   * @param method
   * @param edges
   * @param distances
   */
  void EstimateIntegralLines(std::string method,
                             boost::numeric::ublas::matrix<int> &edges,
                             boost::numeric::ublas::matrix<T> &distances);

  /**
   * Implements the Max Flow algorithm (TODO)
   * @param edges
   * @param distances
   */
  void MaxFlow(boost::numeric::ublas::matrix<int> &edges,
               boost::numeric::ublas::matrix<T> &distances);

  /**
   * Implements the Steepest Edge algorithm
   * @param edges
   * @param distances
   */
  void SteepestEdge(boost::numeric::ublas::matrix<int> &edges,
                    boost::numeric::ublas::matrix<T> &distances);

  //Persistence Simplification

  /**
   * Implements the Steepest Edge algorithm
   * @param NN a matrix of integers representing a neighborhood structure, where
            each row specifies the neighbors of a corresponding sample point.
   */
  void ComputeMaximaPersistence(boost::numeric::ublas::matrix<int> &NN);

  /**
   * Implements the Steepest Edge algorithm
   * @param NN a matrix of integers representing a neighborhood structure, where
            each row specifies the neighbors of a corresponding sample point.
   */
  void ComputeMinimaPersistence(boost::numeric::ublas::matrix<int> &NN);
};

#endif //AMSC_H
