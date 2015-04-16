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

#include "DenseVector.h"

template <typename T>
DenseVector<T>::DenseVector()
{
  n = 0;
  a = NULL;
};

template <typename T>
DenseVector<T>::DenseVector(unsigned int nrows, T *data)
{
  n = nrows;
  a = data;
  if(a == NULL)
  {
    a = new T[n];
  }
}

template <typename T>
DenseVector<T>::~DenseVector() { }

template <typename T>
T & DenseVector<T>::operator()(unsigned int i)
{
  //if(i>=n) throw "Out of bounds";
  return a[i];
}

template <typename T>
T& DenseVector<T>::at(unsigned int i)
{
  return a[i];
}

template <typename T>
const T& DenseVector<T>::at(unsigned int i) const
{
  return a[i];
}

template <typename T>
void DenseVector<T>::setDataPointer(T *data)
{
  a = data;
}

template <typename T>
unsigned int DenseVector<T>::N()
{
  return n;
}

template <typename T>
T * DenseVector<T>::data()
{
  return a;
}

template <typename T>
void DenseVector<T>::Set(T val)
{
  for(int i = 0; i < N(); i++)
    a[i] = val;
}

template <typename T>
void DenseVector<T>::deallocate()
{
  if(a != NULL)
  {
    delete[] a;
  }
}

template class DenseVector<double>;
template class DenseVector<float>;
template class DenseVector<int>;
