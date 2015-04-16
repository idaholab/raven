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

#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include <stddef.h>

/**
 * Simple Matrix storage to column major to use with lapack
 */
template <typename T>
class DenseMatrix
{
 public:
  /**
   * Default Constructor
   * Initializes a NULL container object
   */
  DenseMatrix();

  /**
   * Constructor that will initialize a Matrix of size nrows*ncols with the
     optional values provided by data.
   * @param nrows integer specifying the number of rows in the matrix
   * @param ncols integer specifying the number of columns in the matrix
   * @param data optional floating point array of values to store into the
   * flattened matrix.
   */
  DenseMatrix(unsigned int nrows, unsigned int ncols, T *data = NULL);

  /**
   * Overloading the parentheses () operator to be used as an access operator
   * for the data stored internally in this object
   * @param i nonnegative integer representing the row index to which we are
            exposing access.
   * @param j nonnegative integer representing the column index to which we are
            exposing access.
   */
  virtual T &operator()(unsigned int i, unsigned int j);

  /**
   * Sets the data for the vector.
   * @param val an array of floating point values.
   */
  virtual void Set(T val);

  /**
   * Gets the number of rows in the matrix
   */
  unsigned int M();

  /**
   * Gets the number of columns in the matrix
   */
  unsigned int N();

  /**
   * Returns a pointer to the raw internal data storage.
   */
  T *data();

  /**
   * Sets the data pointer for the internal data storage.
   * @param data a pointer to an array of floating point values.
   */
  void setDataPointer(T *data);

  /**
   * Deallocates the array.
   */
  void deallocate();

  /**
   * Returns a column accessor for the internal data.
   */
  T **getColumnAccessor();

 protected:
  /**
   * Access to data array
   */
  T *a;

  /**
   * Another accessor to data array
   */
  T **fastAccess;

  /*
   * M rows
   */
  unsigned int m;

  /*
   * N cols
   */
  unsigned int n;

 private:
  /**
   * Create an internal fast data accessor for the data object
   */
  void createFastAccess();

  /**
   * Sets an internal fast data accessor for the data object
   */
  void setupFastAccess();
};

#endif //DENSEMATRIX_H
