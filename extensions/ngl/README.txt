NGL
Neighborhood Graph Library v0.1

1. What is NGL?

NGL is an open source C++ library to compute a variety of geometric neighborhood graphs in arbitrary dimensions. Neighborhood graphs include the relative neighbor graph, the Gabriel graph and the parameterized beta-skeleton.

2. Required Libraries
CMake (www.cmake.org)
ANN  -- Approximate Nearest Neighbors (www.cs.umd.edu/~mount/ANN/  bundled with NGL)

Optionally, for interactive and drawing tools:
Cairo (www.cairographics.org)
OpenGL (www.opengl.org)

3. How do I compile NGL?

mkdir build
cd build
cmake ../
make 
make install

4. Are there any example programs?
./bin/getNeighborGraph computes a neighborhood graph from a set of points.

A typical program to compute the Gabriel graph using a naive ON(N^3) implementation is something like this:

	#include "ngl.h"

	int dims = 3; 
	Geometry<float>::init(dims);   // Initialize NGL for 3-dimensional points

	float *pts;
	int n;
	//
	// Allocate memory and populate pts with n 3-dimensional points
	//

	NGLPointSet<float> P(pts, n);  // Initialize Point set

	NGLParams<float> params;
	params.iparam0 = -1;	 	// Initialize parameters
					// Gabriel graph does not require any special parameters
	IndexType *indices;
	int numEdges;
	ngl::getGabrielGraph(P, &indices, numEdges, params);
					// Get graph
	
	// Iterate over indices to output neighbor pairs
	for(int i=0;i<numEdges;i++) 
        {
                fprintf(stdout, "%d %d\n", indices[2*i], indices[2*i+1]);
        }


To speed up the computation, we also present an implementation that
pre-computes a KD-tree of the data points using ANN and assumes that the
desired graph is a subset of the K-nearest neighbor graph, with K<<N a sufficiently large value
For this, use ANNPointSet instead:


	#include "ngl.h"

	int dims = 3; 
	Geometry<float>::init(dims);   // Initialize NGL for 3-dimensional points

	float *pts;
	int n;
	//
	// Allocate memory and populate pts with n 3-dimensional points
	//

	ANNPointSet<float> P(pts, n);  // Initialize Point set using ANN (computes a kd-tree)

	NGLParams<float> params;
	params.iparam0 = KMAX;	 	// Initialize parameters 
					// Only computes Gabriel graph from KMAX candidates
					// which are the KMAX nearest neighbors as computed
					// using ANN
	IndexType *indices;
	int numEdges;
	ngl::getGabrielGraph(P, &indices, numEdges, params);
					// Get graph
	
	// Iterate over indices to output neighbor pairs
	for(int i=0;i<numEdges;i++) 
        {
                fprintf(stdout, "%d %d\n", indices[2*i], indices[2*i+1]);
        }




5. License
NGL is licensed using the Simplified BSD License.
http://opensource.org/licenses/BSD-2-Clause
