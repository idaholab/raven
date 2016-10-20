/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2012  Carlos D. Correa (info@ngraph.org) All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/
#ifndef NGLHPP
#define NGLHPP

#include <assert.h>
#include <vector>

#include "nglGeometry.hpp"

namespace ngl
{
	template<typename T>
	struct NGLParams
	{
		int iparam0;
		T param1, param2;
	};

	template<typename T>
	class NGLPointSet
	{
	protected:
		NGLPoint<T> *pts;
	public:
		unsigned int numPoints;
		NGLPointSet() { }
		NGLPointSet(T*datain, unsigned int numPoints)
		{
			this->numPoints = numPoints;
			this->pts = Geometry<T>::allocate(numPoints);
			for(unsigned int i=0;i<numPoints;i++)
			{
				Geometry<T>::set(this->pts[i], &(datain[i*Geometry<T>::D]));
			}
		}

		virtual ~NGLPointSet() { }

		inline NGLPoint<T>& operator[](int i)
		{
			return pts[i];
		}
		virtual void initialize(NGLParams<T>& params) { }
		virtual void destroy() { }

		virtual void getNeighbors(NGLPoint<T> &p, IndexType **ptrIndices,
                              int &numNeighbors)
		{
			*ptrIndices = new IndexType[numPoints];
			IndexType *indices = *ptrIndices;
			for(IndexType i = 0; i < numPoints; i++) {
				indices[i] = i;
			}
			numNeighbors = numPoints;
		}

    virtual void getNeighbors(int queryIndex, IndexType **ptrIndices,
                              int &numNeighbors)
    {
      getNeighbors(pts[queryIndex], ptrIndices, numNeighbors);
    }

	};

	template<typename T>
	class NGMethod
	{
		bool *valid;
	public:
		NGMethod() {
			valid = 0;
		}

		~NGMethod() { }

		virtual void initialize() { }
		virtual void destroy() { }
		virtual void createValid(int numPts)
		{
			valid = new bool[numPts];
			for(int k = 0; k< numPts; k++)
			{
				valid[k] = true;
			}
		}
		virtual void destroyValid()
		{
			if(valid) delete valid;
		}
		virtual void invalidate(IndexType i)
		{
			valid[i] = false;
		}
		virtual void validate(IndexType i)
		{
			valid[i] = true;
		}
		virtual bool isValid(IndexType i)
		{
			return valid[i];
		}
		virtual void getNeighbors(NGLPoint<T> &p, NGLPointSet<T> &points,
                              IndexType **indices, int &numNeighbors) = 0;

    // Added overload function to prevent from having to search for a point,
    // when we know its index
		virtual void getNeighbors(IndexType queryIndex, NGLPointSet<T> &points,
                              IndexType **indices, int &numNeighbors) = 0;

		virtual void getNeighborGraph(NGLPointSet<T> &points,
                                  IndexType **ptrIndices, int &numEdges)
		{
			createValid(points.numPoints);
			std::vector<int> edges;
		  for(unsigned int i=0; i<points.numPoints;i++)
		  {
			  IndexType *indices_i = 0;
			  int numNeighbors = 0;

			  this->invalidate(i);
        // Don't use this one anymore
//			  this->getNeighbors(points[i], points, &indices_i, numNeighbors);
			  this->getNeighbors(i, points, &indices_i, numNeighbors);
			  this->validate(i);

			  for(int k=0;k<numNeighbors;k++)
			  {
				  if(i==indices_i[k]) continue;
				  edges.push_back(i);
				  edges.push_back(indices_i[k]);
			  }
			  if(indices_i)
          delete indices_i;
		  }

		  *ptrIndices = new IndexType[(int) edges.size()];
		  IndexType *indices = *ptrIndices;
		  for(unsigned int k = 0; k < edges.size(); k++)
		  {
			  indices[k] = edges[k];
		  }
		  numEdges = (int) edges.size() / 2;
		  destroyValid();
		}
	};

	template<typename T>
	class KNNMethod: public NGMethod<T>
	{
		unsigned int K;
	public:
		KNNMethod(unsigned int K):NGMethod<T>()
		{
			this->K = K;
		}

		~KNNMethod() { }

		virtual void initialize() { }
		virtual void destroy() { }

		virtual void getNeighbors(NGLPoint<T> &p, NGLPointSet<T> &points,
                              IndexType **ptrIndices, int &numNeighbors)
		{
			int candidateSize;
			IndexType *candidateNeighbors;
			points.getNeighbors(p, &candidateNeighbors, candidateSize);

			if(candidateSize>0)
      {
				*ptrIndices = new IndexType[K];
				IndexType *indices = *ptrIndices;
				for(int k=0; k<candidateSize; k++)
        {
					IndexType idx = candidateNeighbors[k];
					if(!NGMethod<T>::isValid(idx))
            continue;
					indices[k] = candidateNeighbors[k];
					if(k >= (int)K)
            break;
				}
				numNeighbors = K;
			}

			delete[] candidateNeighbors;
		}

	};
}

/*
template struct ngl::NGLParams<float>;
template class ngl::NGLPointSet<float>;
template class ngl::NGMethod<float>;
template class ngl::KNNMethod<float>;

template struct ngl::NGLParams<double>;
template class ngl::NGLPointSet<double>;
template class ngl::NGMethod<double>;
template class ngl::KNNMethod<double>;
*/

#endif
