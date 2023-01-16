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
#ifndef NGLIMPLHPP
#define NGLIMPLHPP
#include "ngl.hpp"

namespace ngl 
{
  //
  // Pass through empty region test for getting the ANN
  //
  template<typename T>
  class PassThrough: public EmptyRegionTest<T>
  {
  protected:
  public:
      virtual void initialize()
      {
      }
      virtual void destroy()
      {
      }
      virtual T contains(EdgeInfo<T> &edge, NGLPoint<T> &r)
      {
          return 1;
      }
  };

	//
	// Relative Neighbor
	//
	template<typename T>
	class RelativeNeighbor: public EmptyRegionTest<T>
	{
	protected:
	public:
		virtual void initialize()
		{
		}
		virtual void destroy() 
		{
		}
		virtual T contains(EdgeInfo<T> &edge, NGLPoint<T> &r) 
		{
			T d1 = Geometry<T>::distanceL2sqr(r, edge.p);
			T d2 = Geometry<T>::distanceL2sqr(r, edge.q);
			return std::max(d2 - edge.len2, d1 - edge.len2);
		}
	};
	
	//
	// Gabriel
	//
	template<typename T>
	class Gabriel: public EmptyRegionTest<T> 
	{
	public:
		virtual void initialize()
		{
		}
		virtual void destroy() 
		{
		}
		virtual T contains(EdgeInfo<T> &edge, NGLPoint<T> &r) 
		{
			T d1 = Geometry<T>::distanceL2sqr(r, edge.midpoint);
			return d1 - edge.radius2;
		}
	};

	//
	// Elliptic Gabriel
	//
	template<typename T>
	class EllipticGabriel: public EmptyRegionTest<T> 
	{
		NGLPoint<T> rp;
		NGLPoint<T> qp;
		NGLPoint<T> proj;
		T ratio;
	public:
		EllipticGabriel(float ratio) 
		{
			this->ratio = ratio;
		}
		virtual void initialize()
		{
			Geometry<T>::allocate(rp);
			Geometry<T>::allocate(qp);
			Geometry<T>::allocate(proj);
		}
		virtual void destroy() 
		{
			Geometry<T>::deallocate(rp);
			Geometry<T>::deallocate(qp);
			Geometry<T>::deallocate(proj);
		}
		virtual T contains(EdgeInfo<T> &edge, NGLPoint<T> &r) 
		{
			T axis1sqr = 1.0;
			T axis2sqr = ratio*ratio;
			Geometry<T>::subtract(r, edge.p, rp);
			Geometry<T>::subtract(edge.q, edge.p, qp);
			T t = Geometry<T>::dot(rp,qp)/Geometry<T>::dot(qp,qp);
			Geometry<T>::interpolate(edge.p, edge.q, t, proj);
			T dpsqr = 
				Geometry<T>::distanceL2sqr(proj, edge.midpoint)/axis1sqr + 
				Geometry<T>::distanceL2sqr(proj, r)/axis2sqr;
			return dpsqr - edge.radius2;
		}
	};
	
	
	//
	// Lune-based beta-skeleton
	//
	template<typename T>
	class BSkeleton: public EmptyRegionTest<T> 
	{
		NGLPoint<T> rp;
		NGLPoint<T> qp;
		NGLPoint<T> proj;
		T beta;
	public:
		BSkeleton(float beta) 
		{
			this->beta = beta;
		}
		virtual void initialize()
		{
			Geometry<T>::allocate(rp);
			Geometry<T>::allocate(qp);
			Geometry<T>::allocate(proj);
		}
		virtual void destroy() 
		{
			Geometry<T>::deallocate(rp);
			Geometry<T>::deallocate(qp);
			Geometry<T>::deallocate(proj);
		}
		virtual T contains(EdgeInfo<T> &edge, NGLPoint<T> &r) 
		{
			if(beta<1.0) {
				T r2 = edge.radius2/(beta*beta);
				T h2 = r2 - edge.radius2;
				T delta = sqrt(h2);

				Geometry<T>::subtract(r, edge.p, rp);
				Geometry<T>::subtract(edge.q, edge.p, qp);
				T t = Geometry<T>::dot(rp,qp)/Geometry<T>::dot(qp,qp);
				Geometry<T>::interpolate(edge.p, edge.q, t, proj);
				
				T dproj = sqrt(Geometry<T>::distanceL2sqr(r, proj));
				T dprojmidsqr = (Geometry<T>::distanceL2sqr(proj, edge.midpoint));
				T d2 = dprojmidsqr + (dproj + delta)*(dproj + delta);
				return d2 - r2;
			} else {
				NGLPoint<T> &c1 = rp;
				NGLPoint<T> &c2 = qp;
				Geometry<T>::interpolate(edge.p, edge.q, beta/2.0, c1);
				Geometry<T>::interpolate(edge.p, edge.q, 1.0 - beta/2.0, c2);
				T r2 = edge.radius2*beta*beta;
				T d1 = Geometry<T>::distanceL2sqr(r, c1);
				T d2 = Geometry<T>::distanceL2sqr(r, c2);
				
				return std::max(d2-r2, d1-r2);
			}
		}
	};
	
	
	//
	// Diamond
	//
	template<typename T>
	class Diamond: public EmptyRegionTest<T> 
	{
		NGLPoint<T> rp;
		NGLPoint<T> qp;
		NGLPoint<T> rq;
		T ratio;
	public:
		Diamond(float ratio) 
		{
			this->ratio = ratio;
		}
		virtual void initialize()
		{
			Geometry<T>::allocate(rp);
			Geometry<T>::allocate(qp);
			Geometry<T>::allocate(rq);
		}
		virtual void destroy() 
		{
			Geometry<T>::deallocate(rp);
			Geometry<T>::deallocate(qp);
			Geometry<T>::deallocate(rq);
		}
		virtual T contains(EdgeInfo<T> &edge, NGLPoint<T> &r) 
		{
			Geometry<T>::subtract(r, edge.p, rp);
			Geometry<T>::subtract(r, edge.q, rq);
			Geometry<T>::subtract(edge.q, edge.p, qp);
			
			T num1 = Geometry<T>::dot(rp,qp)*fabs(Geometry<T>::dot(rp,qp));
			T den1 = Geometry<T>::dot(qp,qp)*Geometry<T>::dot(rp,rp);

			T num2 = -Geometry<T>::dot(rq,qp)*fabs(Geometry<T>::dot(rq,qp));
			T den2 = Geometry<T>::dot(qp,qp)*Geometry<T>::dot(rq,rq);

			return ratio*ratio - min(num1/den1, num2/den2);
		}
	};
	
	template<typename T>
	void generalERgraph(NGLPointSet<T> &points, IndexType **indices, 
                      int &numEdges, NGLParams<T> params, 
                      EmptyRegionTest<T> *method)
	{
		assert(method);
		EmptyRegionMethod<T> *m = new EmptyRegionMethod<T>(method);
		points.initialize(params);		
		m->initialize();
		m->getNeighborGraph(points, indices, numEdges);
		delete m;
	}

	template<typename T>
	void getGabrielGraph(NGLPointSet<T> &points, IndexType **indices, 
                       int &numEdges, NGLParams<T> params) 
	{
		EmptyRegionTest<T> *method = new Gabriel<T>();
		method->initialize();
		generalERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}

	template<typename T>
	void getRelativeNeighborGraph(NGLPointSet<T> &points, IndexType **indices, 
                                int &numEdges, NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new RelativeNeighbor<T>();
		method->initialize();
		generalERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void getBSkeleton(NGLPointSet<T> &points, IndexType **indices, int &numEdges, 
                    NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new BSkeleton<T>(params.param1);
		method->initialize();
		generalERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void getEllipticGabrielGraph(NGLPointSet<T> &points, IndexType **indices,
                               int &numEdges, NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new EllipticGabriel<T>(params.param1);
		method->initialize();
		generalERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void getDiamondGraph(NGLPointSet<T> &points, IndexType **indices,
                       int &numEdges, NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new Diamond<T>(params.param1);
		method->initialize();
		generalERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void generalRelaxedERgraph(NGLPointSet<T> &points, IndexType **indices,
                             int &numEdges, NGLParams<T> params,
                             EmptyRegionTest<T> *method)
	{
		assert(method);
		RelaxedEmptyRegionMethod<T> *m = new RelaxedEmptyRegionMethod<T>(method);
		points.initialize(params);		
		m->initialize();
		m->getNeighborGraph(points, indices, numEdges);
		delete m;
	}
	
	template<typename T>
	void getRelaxedGabrielGraph(NGLPointSet<T> &points, IndexType **indices,
                              int &numEdges, NGLParams<T> params) 
	{
		EmptyRegionTest<T> *method = new Gabriel<T>();
		method->initialize();
		generalRelaxedERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void getRelaxedRelativeNeighborGraph(NGLPointSet<T> &points,
                                       IndexType **indices, int &numEdges,
                                       NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new RelativeNeighbor<T>();
		method->initialize();
		generalRelaxedERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void getRelaxedBSkeleton(NGLPointSet<T> &points, IndexType **indices,
                           int &numEdges, NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new BSkeleton<T>(params.param1);
		method->initialize();
		generalRelaxedERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void getRelaxedEllipticGabrielGraph(NGLPointSet<T> &points,
                                      IndexType **indices, int &numEdges,
                                      NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new EllipticGabriel<T>(params.param1);
		method->initialize();
		generalRelaxedERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
	template<typename T>
	void getRelaxedDiamondGraph(NGLPointSet<T> &points, IndexType **indices,
                              int &numEdges, NGLParams<T> params)
	{
		EmptyRegionTest<T> *method = new Diamond<T>(params.param1);
		method->initialize();
		generalRelaxedERgraph(points, indices, numEdges, params, method);
		method->destroy();
		delete method;
	}
	
  template<typename T>
  void getKNNGraph(NGLPointSet<T> &points, IndexType **indices, int &numEdges,
                   NGLParams<T> params)
  {
    EmptyRegionTest<T> *method = new PassThrough<T>();
    method->initialize();
    generalERgraph(points, indices, numEdges, params, method);
    method->destroy();
    delete method;
  }

	template<typename T>
	void getEpsilonGraph(NGLPointSet<T> &points, IndexType **indices,
                       int &numEdges, NGLParams<T> params);
}

#endif
