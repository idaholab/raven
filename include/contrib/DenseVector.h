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

#ifndef DENSEVECTOR_H
#define DENSEVECTOR_H

#include <cstddef>

/**
 * Simple Matrix storage to abstract row and columnwise ordering
 */
template <typename T>
class DenseVector
{
 public:
  /**
   * Default Constructor
   * Initializes a NULL container object
   */
  DenseVector();

  /**
   * Constructor that will initialize a Vector of size nrows with the optional
   * values provided by data.
   * @param nrows integer specifying the size of the vector
   * @param data optional floating point array of values to store into the
   * vector.
   */
  DenseVector(unsigned int nrows, T *data=NULL);

  /**
   * Destructor will deallocate the internal storage of the vector data
   */
  virtual ~DenseVector();

  /**
   * Overloading the parentheses () operator to be used as an access operator
   * for the data stored internally in this object
   * @param i nonnegative integer representing the index to which we are
            exposing access.
   */
  virtual T &operator()(unsigned int i);

  /**
   * Another access operator for the data stored internally in this object
   * @param i nonnegative integer representing the index to which we are
            exposing access.
   */
  virtual T& at(unsigned int i);

  /**
   * Constant access operator for the data stored internally in this object
   * @param i nonnegative integer representing the index to which we are
            exposing access.
   */
  virtual const T& at(unsigned int i) const;

  /**
   * Sets the data for the vector.
   * @param val an array of floating point values.
   */
  virtual void Set(T val);

  /**
   * Sets the data pointer for the internal data storage.
   * @param data a pointer to an array of floating point values.
   */
  void setDataPointer(T *data);

  /**
   * Returns the size of the data array.
   */
  unsigned int N();

  /**
   * Returns a pointer to the raw internal data storage.
   */
  T *data();

  /**
   * Deallocates the array.
   */
  void deallocate();

 protected:
  /**
   * The internal data array.
   */
  T *a;

  /**
   * Variable storing the size of the data array.
   */
  unsigned int n;
};

#endif //DENSEVECTOR_H
