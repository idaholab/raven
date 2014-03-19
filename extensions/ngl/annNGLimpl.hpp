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
#ifndef ANNNGLIMPLHPP
#define ANNNGLIMPLHPP

#include <assert.h>
#include <vector>
#include <ANN/ANN.h>

#include "nglGeometry.hpp"

namespace ngl 
{		

	template<typename T>
	class ANNPointSet: public NGLPointSet<T>
	{
	protected:
		ANNpointArray dataPts;
		ANNpoint queryPt;
		ANNidxArray nnIdx;
		ANNdistArray dists;
		ANNkd_tree* kdTree;
		int K;
	public:
		ANNPointSet(): NGLPointSet<T>()
		{
		}
		
		ANNPointSet(T*datain, unsigned int numPoints): NGLPointSet<T>(datain, numPoints) 
		{
		}
		
		virtual void initialize(NGLParams<T>& params) 
		{
			K = params.iparam0;
			int dims = Geometry<T>::D;
			int numPts = this->numPoints;
			dataPts = annAllocPts(numPts, dims);
			
			for(int i = 0; i<numPts;i++) {
				for(int k=0;k<dims;k++) {
					dataPts[i][k] = this->pts[i][k];
				}
			}
			kdTree = new ANNkd_tree(dataPts, numPts, dims);
			
			queryPt = annAllocPt(dims);
			nnIdx = new ANNidx[K];
			dists = new ANNdist[K];
		}
		virtual void destroy() 
		{
			annDeallocPts(dataPts);
			annDeallocPt(queryPt);
			delete nnIdx;
			delete dists;
			delete kdTree;
		}
		virtual void getNeighbors(NGLPoint<T> &p, IndexType **ptrIndices, int &numNeighbors)
		{
			for(unsigned int k=0;k<Geometry<T>::D;k++) queryPt[k] = p[k];
			
			float epsilon = 0.0f;
			kdTree->annkSearch(queryPt, K, nnIdx, dists, epsilon);
			
			*ptrIndices = new IndexType[K];
			IndexType *indices = *ptrIndices;
			numNeighbors = K;
			for(int k=0;k<K;k++) {
				indices[k] = nnIdx[k];
			}
		}			
	};	
	
		
}

template class ngl::ANNPointSet<Precision>;

#endif
