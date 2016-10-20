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
#ifndef EMPTYREGIONH
#define EMPTYREGIONH

#include <assert.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>

#include "nglGeometry.hpp"

namespace ngl
{
	template<typename T>
	class EdgeInfo
	{
	public:
		NGLPoint<T> p;
		NGLPoint<T> q;
		NGLPoint<T> midpoint;
		T len2, radius2;
		void initialize()
		{
			Geometry<T>::allocate(midpoint);
			Geometry<T>::allocate(p);
			Geometry<T>::allocate(q);
		}
		void compute(NGLPoint<T> &pin, NGLPoint<T> &qin)
		{
			Geometry<T>::set(p, pin);
			Geometry<T>::set(q, qin);
			Geometry<T>::interpolate(p, q, 0.5, midpoint);
			len2 = Geometry<T>::distanceL2sqr(p,q);
			radius2 = len2/4.0;
		}
		void destroy()
		{
			Geometry<T>::deallocate(midpoint);
			Geometry<T>::deallocate(p);
			Geometry<T>::deallocate(q);
		}
	};

	template<typename T>
	class EmptyRegionTest
	{
	public:
		virtual void initialize() = 0;
		virtual void destroy() = 0;
		virtual T contains(EdgeInfo<T> &edge, NGLPoint<T> &r) = 0;
		virtual ~EmptyRegionTest() { }
	};
	template<typename T>
	class UmbraTest
	{
	public:
		virtual void initialize() = 0;
		virtual void destroy() = 0;
		virtual T shadows(EdgeInfo<T> &edge, NGLPoint<T> &r) = 0;
		virtual ~UmbraTest() { }
	};

	template<typename T>
	class EmptyRegionMethod: public NGMethod<T>
	{
	protected:
		EmptyRegionTest<T> *test;
	public:
		EmptyRegionMethod(EmptyRegionTest<T> *test): NGMethod<T>()
		{
			this->test = test;
		}

		virtual ~EmptyRegionMethod() { }

		virtual void initialize()
    {
			NGMethod<T>::initialize();
		}
		virtual void destroy()
    {
		}
		virtual void getNeighbors(NGLPoint<T> &p, NGLPointSet<T> &points,
                              IndexType **ptrIndices, int &numNeighbors)
    {
			assert(test);
			std::vector<IndexType> neighbors;

			int candidateSize;
			IndexType *candidateNeighbors;

			//
			// Assumes that point set in 'points' has pre-computed a set of neighbors
			// (say, KNN with k = kMax) or if it doesn't, the subset equals the entire
			//  set
			points.getNeighbors(p, &candidateNeighbors, candidateSize);

			EdgeInfo<T> edgeInfo;
			edgeInfo.initialize();
			for(int k = 0; k < candidateSize; k++ )
			{
				IndexType idx = candidateNeighbors[k];
				if(!NGMethod<T>::isValid(idx))
          continue;

				// Pre-compute edge information
				edgeInfo.compute(p, points[idx]);

				if(edgeInfo.len2==0)
          continue;
				bool isRegionEmpty = true;
				for(int j = 0; j < candidateSize; j++ )
				{
					if(j==k)
            continue;
					IndexType idx2 = candidateNeighbors[j];
					if(!NGMethod<T>::isValid(idx2))
            continue;
					T testresult = test->contains(edgeInfo, points[idx2]);
					//DM: Use epsilon to ensure that the edge cases still
					//    fail accordingly (this should fix a peculiarity
					//    in the computation that was only manifested on
					//    Windows)
					if(testresult<=std::numeric_limits<T>::epsilon())
					{
						isRegionEmpty = false;
						break;
					}
				}
				if(isRegionEmpty)
				{
					neighbors.push_back(idx);
				}
			}

			edgeInfo.destroy();
			delete[] candidateNeighbors;

			numNeighbors = (int) neighbors.size();
			if(neighbors.size()>0)
      {
				*ptrIndices = new IndexType[(int) neighbors.size()];
				IndexType *indices = *ptrIndices;
				for(unsigned int k=0;k<neighbors.size();k++)
        {
					indices[k] = neighbors[k];
				}
			}
		}
		virtual void getNeighbors(IndexType queryIndex, NGLPointSet<T> &points,
                              IndexType **ptrIndices, int &numNeighbors)
    {
			assert(test);
			std::vector<IndexType> neighbors;

			int candidateSize;
			IndexType *candidateNeighbors;

			//
			// Assumes that point set in 'points' has pre-computed a set of neighbors
			// (say, KNN with k = kMax) or if it doesn't, the subset equals the entire
			//  set
			points.getNeighbors(queryIndex, &candidateNeighbors, candidateSize);
      NGLPoint<T> p = points[queryIndex];

			EdgeInfo<T> edgeInfo;
			edgeInfo.initialize();
			for(int k = 0; k < candidateSize; k++ )
			{
				IndexType idx = candidateNeighbors[k];
				if(!NGMethod<T>::isValid(idx)) continue;

				// Pre-compute edge information
				edgeInfo.compute(p, points[idx]);

				if(edgeInfo.len2==0) continue;
				bool isRegionEmpty = true;
				for(int j = 0; j < candidateSize; j++ )
				{
					if(j==k) continue;
					IndexType idx2 = candidateNeighbors[j];
					if(!NGMethod<T>::isValid(idx2)) continue;
					T testresult = test->contains(edgeInfo, points[idx2]);
					//DM: Use epsilon to ensure that the edge cases still
					//    fail accordingly (this should fix a peculiarity
					//    in the computation that was only manifested on
					//    Windows)
					if(testresult<=std::numeric_limits<T>::epsilon())
					{
						isRegionEmpty = false;
						break;
					}
				}
				if(isRegionEmpty)
				{
					neighbors.push_back(idx);
				}
			}

			edgeInfo.destroy();
			delete[] candidateNeighbors;

			numNeighbors = (int) neighbors.size();
			if(neighbors.size()>0)
      {
				*ptrIndices = new IndexType[(int) neighbors.size()];
				IndexType *indices = *ptrIndices;
				for(unsigned int k=0;k<neighbors.size();k++)
        {
					indices[k] = neighbors[k];
				}
			}
		}
	};

	template<typename T>
	class RelaxedEmptyRegionMethod: public EmptyRegionMethod<T>
	{
	public:
		RelaxedEmptyRegionMethod(EmptyRegionTest<T> *test)
      : EmptyRegionMethod<T>(test)
		{
		}

		virtual ~RelaxedEmptyRegionMethod() { }

		virtual void initialize()
    {
			EmptyRegionMethod<T>::initialize();
		}
		virtual void destroy()
    {
			EmptyRegionMethod<T>::destroy();
		}
		virtual void getNeighbors(NGLPoint<T> &p, NGLPointSet<T> &points,
                              IndexType **ptrIndices, int &numNeighbors)
    {
			assert(this->test);
			std::vector<IndexType> neighbors;

			int candidateSize;
			IndexType *candidateNeighbors;

			//
			// Assumes that point set in 'points' has pre-computed a set of neighbors
			// (say, KNN with k = kMax) Or if it doesn't, the subset equals the entire
			//  set
			// Assume these points are ordered by distance?
			// DM: Assume nothing, sort them now!
			points.getNeighbors(p, &candidateNeighbors, candidateSize);

			std::vector< std::pair<T,int> > sortedIndices;
			for (int k = 0; k < candidateSize; k++)
			{
				IndexType idx = candidateNeighbors[k];
				T distance = Geometry<T>::distanceL2sqr(p,points[idx]);
				sortedIndices.push_back(std::pair<T,int>(distance,idx));
			}
			std::sort(sortedIndices.begin(),sortedIndices.end());
			//DM: Now just put them back and be done with the vector
			for (int k = 0; k < candidateSize; k++)
				candidateNeighbors[k] = sortedIndices[k].second;

			EdgeInfo<T> edgeInfo;
			edgeInfo.initialize();
			std::vector<int> added;
			for(unsigned int k = 0; k < (unsigned int) candidateSize; k++ )
			{
				IndexType idx = candidateNeighbors[k];
				if(!NGMethod<T>::isValid(idx))
          continue;

				// Pre-compute edge information
				edgeInfo.compute(p, points[idx]);

				if(edgeInfo.len2==0)
          continue;

				bool isRegionEmpty = true;
				for(unsigned int i = 0; i < added.size(); i++ )
				{
					unsigned int j = added[i];
					if(j==k)
            continue;
					IndexType idx2 = candidateNeighbors[j];
					if(!NGMethod<T>::isValid(idx2))
            continue;
					T testresult = this->test->contains(edgeInfo, points[idx2]);
					//DM: Use epsilon to ensure that the edge cases still
					//    fail accordingly (this should fix a peculiarity
					//    in the computation that was only manifested on
					//    Windows)
					if(testresult<=std::numeric_limits<T>::epsilon())
					{
						isRegionEmpty = false;
						break;
					}
				}
				if(isRegionEmpty)
				{
					added.push_back(k);
					neighbors.push_back(idx);
				}
			}

			edgeInfo.destroy();
			delete[] candidateNeighbors;
			numNeighbors = (int) neighbors.size();
			if(neighbors.size()>0)
      {
				*ptrIndices = new IndexType[(int) neighbors.size()];
				IndexType *indices = *ptrIndices;
				for(unsigned int k=0;k<neighbors.size();k++)
        {
					indices[k] = neighbors[k];
				}
			}
		}

		virtual void getNeighbors(IndexType queryIndex, NGLPointSet<T> &points,
                              IndexType **ptrIndices, int &numNeighbors)
    {
			assert(this->test);
			std::vector<IndexType> neighbors;

			int candidateSize;
			IndexType *candidateNeighbors;

			//
			// Assumes that point set in 'points' has pre-computed a set of neighbors
			// (say, KNN with k = kMax) Or if it doesn't, the subset equals the entire
			//  set
			// Assume these points are ordered by distance?
			points.getNeighbors(queryIndex, &candidateNeighbors, candidateSize);
      NGLPoint<T> p = points[queryIndex];

			// DM: Assume nothing, sort them now!
			std::vector< std::pair<T,int> > sortedIndices;
			for (int k = 0; k < candidateSize; k++)
			{
				IndexType idx = candidateNeighbors[k];
				T distance = Geometry<T>::distanceL2sqr(p,points[idx]);
				sortedIndices.push_back(std::pair<T,int>(distance,idx));
			}
			std::sort(sortedIndices.begin(),sortedIndices.end());
			//DM: Now just put them back and be done with the vector
			for (int k = 0; k < candidateSize; k++)
				candidateNeighbors[k] = sortedIndices[k].second;

			EdgeInfo<T> edgeInfo;
			edgeInfo.initialize();
			std::vector<int> added;
			for(unsigned int k = 0; k < (unsigned int) candidateSize; k++ )
			{
				IndexType idx = candidateNeighbors[k];
				if(!NGMethod<T>::isValid(idx))
          continue;

				// Pre-compute edge information
				edgeInfo.compute(p, points[idx]);

				if(edgeInfo.len2==0)
          continue;

				bool isRegionEmpty = true;
				for(unsigned int i = 0; i < added.size(); i++ )
				{
					unsigned int j = added[i];
					if(j==k)
            continue;
					IndexType idx2 = candidateNeighbors[j];
					if(!NGMethod<T>::isValid(idx2))
            continue;
					T testresult = this->test->contains(edgeInfo, points[idx2]);
					//DM: Use epsilon to ensure that the edge cases still
					//    fail accordingly (this should fix a peculiarity
					//    in the computation that was only manifested on
					//    Windows)
					if(testresult<=std::numeric_limits<T>::epsilon())
					{
						isRegionEmpty = false;
						break;
					}
				}
				if(isRegionEmpty)
				{
					added.push_back(k);
					neighbors.push_back(idx);
				}
			}

			edgeInfo.destroy();
			delete[] candidateNeighbors;
			numNeighbors = (int) neighbors.size();
			if(neighbors.size()>0)
      {
				*ptrIndices = new IndexType[(int) neighbors.size()];
				IndexType *indices = *ptrIndices;
				for(unsigned int k=0;k<neighbors.size();k++)
        {
					indices[k] = neighbors[k];
				}
			}
		}
	};
}

/*
template class ngl::EdgeInfo<float>;
template class ngl::EmptyRegionTest<float>;
template class ngl::UmbraTest<float>;
template class ngl::RelaxedEmptyRegionMethod<float>;
template class ngl::EmptyRegionMethod<float>;

template class ngl::EdgeInfo<double>;
template class ngl::EmptyRegionTest<double>;
template class ngl::UmbraTest<double>;
template class ngl::RelaxedEmptyRegionMethod<double>;
template class ngl::EmptyRegionMethod<double>;
*/
#endif
