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
#ifndef PREBUILTNGLIMPLHPP
#define PREBUILTNGLIMPLHPP

#include <assert.h>
#include <vector>
#include "nglGeometry.hpp"
#include <algorithm>

namespace ngl
{
  template<typename T>
  class prebuiltNGLPointSet: public NGLPointSet<T>
  {
  protected:
    int *neighborIndices;
    int *neighborOffsets;
    int kmax;

  public:
    prebuiltNGLPointSet()
     : NGLPointSet<T>()
    {
    }

    prebuiltNGLPointSet(T*datain, unsigned int numPoints,
                        std::vector<int> &edgeList)
      : NGLPointSet<T>(datain, numPoints)
    {
      int *neighborCounts = new int[numPoints];
      for(unsigned int i = 0; i < numPoints; i++)
        neighborCounts[i] = 0;

      for(unsigned int i = 0; i < edgeList.size(); i+=2)
      {
        int e1 = edgeList[i];
        int e2 = edgeList[i+1];
        neighborCounts[e1]++;
        neighborCounts[e2]++;
      }

      neighborOffsets = new int[numPoints];
      neighborOffsets[0] = neighborCounts[0];
      //Re-zero the neighborCounts while storing the offsets for each index
      neighborCounts[0] = 0;
      for(unsigned int i=1; i < numPoints; i++)
      {
        neighborOffsets[i] = neighborCounts[i] + neighborOffsets[i-1];
        neighborCounts[i] = 0;
      }

      neighborIndices = new int[neighborOffsets[numPoints-1]];
      for(unsigned int i = 0; i < edgeList.size(); i+=2)
      {
        int e1 = edgeList[i];
        int e2 = edgeList[i+1];

        int e1Offset = neighborCounts[e1];
        if(e1 > 0)
          e1Offset += neighborOffsets[e1-1];
        neighborIndices[e1Offset] = e2;
        neighborCounts[e1]++;

        int e2Offset = neighborCounts[e2];
        if(e2 > 0)
          e2Offset += neighborOffsets[e2-1];
        neighborIndices[e2Offset] = e1;
        neighborCounts[e2]++;
      }

      delete [] neighborCounts;
    }

    virtual void destroy()
    {
      delete [] neighborIndices;
      delete [] neighborOffsets;
    }

    virtual void getNeighbors(NGLPoint<T> &p, IndexType **ptrIndices,
                              int &numNeighbors)
    {
      //Find the point index otherwise return the full set of data by calling
      // the base class method
      int queryIndex = -1;

      if(queryIndex < 0)
        NGLPointSet<T>::getNeighbors(queryIndex, ptrIndices, numNeighbors);
      else
        getNeighbors(queryIndex, ptrIndices, numNeighbors);
    }

    //Overloading this function name so I don't have to search for the query
    // point if it already exists in the point set (which, for my purposes, it
    // should)
    virtual void getNeighbors(int queryIndex, IndexType **ptrIndices,
                              int &numNeighbors)
    {
      int offset = 0;
      if(queryIndex > 0)
        offset = neighborOffsets[queryIndex-1];
      numNeighbors = neighborOffsets[queryIndex] - offset;

      *ptrIndices = new IndexType[numNeighbors];
      IndexType *indices = *ptrIndices;
      for(int i = 0; i < numNeighbors; i++)
        indices[i] = neighborIndices[offset+i];
    }
  };
}

#endif
