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

#include "DenseMatrix.h"

template <typename T>
DenseMatrix<T>::DenseMatrix()
{
  m = n = 0;
  a = NULL;
  fastAccess = NULL;
}

template <typename T>
DenseMatrix<T>::DenseMatrix(unsigned int nrows, unsigned int ncols, T *data)
{
  m = nrows;
  n = ncols;

  unsigned long l = n;
  l*=m;
  a = data;
  if(a == NULL)
  {
      a = new T[l];
  }
  createFastAccess();
  setupFastAccess();
}

template <typename T>
T & DenseMatrix<T>::operator()(unsigned int i, unsigned int j)
{
    //if(i>=m || j >= n) throw "Out of bounds";
  return fastAccess[j][i];
}

template <typename T>
void DenseMatrix<T>::Set(T val)
{
  for(int row = 0; row < M(); row++)
    for(int col = 0; col < N(); col++)
      fastAccess[col][row] = val;
}

template <typename T>
unsigned int DenseMatrix<T>::M()
{
  return m;
}

template <typename T>
unsigned int DenseMatrix<T>::N()
{
  return n;
}

template <typename T>
T * DenseMatrix<T>::data()
{
  return a;
}

template <typename T>
void DenseMatrix<T>::setDataPointer(T *data)
{
  a = data;
  setupFastAccess();
}

template <typename T>
void DenseMatrix<T>::deallocate()
{
  if(a != NULL)
  {
    delete[] a;
    delete[] fastAccess;
  }
}

template <typename T>
T ** DenseMatrix<T>::getColumnAccessor()
{
  return fastAccess;
}

template <typename T>    
void DenseMatrix<T>::createFastAccess()
{
  fastAccess = new T*[n];
}

template <typename T>
void DenseMatrix<T>::setupFastAccess()
{
  for(unsigned int i=0; i<n; i++)
  {
    fastAccess[i] = &a[i*m];
  }
}

template class DenseMatrix<double>;
template class DenseMatrix<float>;
template class DenseMatrix<int>;
