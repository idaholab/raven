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
#ifndef NGLGEOMETRYHPP
#define NGLGEOMETRYHPP

#include "math.h"

namespace ngl 
{
	typedef unsigned int IndexType;
	
	template<typename T>
	struct NGLPoint
	{
		T *data;
		inline T& operator[](int i) { return data[i]; }
		inline T operator *() { return *data; }
	};
	
	template<typename T>
	class Geometry 
  {
	 public:
		static unsigned int D;
		static void init(unsigned int d) 
		{
			D = d;
		}
		static NGLPoint<T>* allocate(int numPts) 
		{
			NGLPoint<T>* ptr = static_cast<NGLPoint<T>*>( new NGLPoint<T>[numPts] );
			for(int i = 0; i < numPts ; i++) 
			{
				allocate(ptr[i]);
			}
			return ptr;
		}
		static void allocate(NGLPoint<T> &p) 
		{
			p.data = static_cast<T*>( new T[D] );
		}
		static void deallocate(NGLPoint<T> &p) 
		{
			delete p.data;
		}
		inline static T distanceL2(NGLPoint<T> &a, NGLPoint<T> &b) 
		{
			return sqrt(distanceL2sqr(a,b));
		}
		inline static T distanceL2sqr(NGLPoint<T> &a, NGLPoint<T> &b) 
		{
			T dis2 = 0;
			for(unsigned int k=0;k<D;k++) 
			{
				dis2+=(a[k] - b[k])*(a[k] - b[k]);
			}
			return dis2;
		}
		static T distanceL0(NGLPoint<T> &a, NGLPoint<T> &b) 
		{
				return -1;
		}
		static T distanceLinf(NGLPoint<T> &a, NGLPoint<T> &b) 
		{
			return -1;
		}
		inline static void add(NGLPoint<T> &a, NGLPoint<T> &b, NGLPoint<T> &c) 
		{
			for(unsigned int k=0;k<D;k++) 
			{
				c[k] = a[k] + b[k];
			}
		}
		inline static void subtract(NGLPoint<T> &a, NGLPoint<T> &b, NGLPoint<T> &c) 
		{
			for(unsigned int k=0;k<D;k++) 
			{
				c[k] = a[k] - b[k];
			}
		}
		inline static T dot(NGLPoint<T> &a, NGLPoint<T> &b) 
		{
			T res = 0;
			for(unsigned int k=0;k<D;k++) 
			{
				res+=a[k]*b[k];
			}
			return res;
		}
		inline static T normalize(NGLPoint<T> &a) 
		{
			T adota = dot(a,a);
			T lena = sqrt(adota);
			if(lena>0) {
				for(unsigned int k=0;k<D;k++) 
				{
					a[k]/=lena;
				}
			}
			return lena;
		}
		inline static void set(NGLPoint<T> &dst, NGLPoint<T> &src) 
		{
			for(unsigned int k=0;k<D;k++) 
			{
				dst[k] = src[k];
			}
		}
		inline static void set(NGLPoint<T> &dst, T*src) 
		{
			for(unsigned int k=0;k<D;k++) 
			{
				dst[k] = src[k];
			}
		}
		inline static void muladd(NGLPoint<T> &a, NGLPoint<T> &b, T t, 
                              NGLPoint<T> &c) 
		{
			for(unsigned int k=0;k<D;k++) 
			{
				c[k] = a[k] + t*b[k];
			}
		}
		inline static void interpolate(NGLPoint<T> &a, NGLPoint<T> &b, T t, 
                                   NGLPoint<T> &c) 
		{
			for(unsigned int k=0;k<D;k++) 
			{
				c[k] = (1-t)*a[k] + t*b[k];
			}
		}
		inline static void mul(NGLPoint<T> &a, T x, NGLPoint<T> &b) 
		{
			for(unsigned int k=0;k<D;k++) 
			{
				b[k] = a[k]*x;
			}
		}
		
	};
	
	/*
	template<class T>
	struct NGL
	{
		typedef T* Point;
	};
	 */
}

/*
template struct ngl::NGLPoint<float>;
template class ngl::Geometry<float>;

template struct ngl::NGLPoint<double>;
template class ngl::Geometry<double>;
*/
#endif
