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

#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <vector>

/**
 * A singleton class for the encoding an element of the Union-Find data
 * structure.
 */
class Singleton
{
 public:
  /**
   * Constructor to initialize a Singleton as its own parent and of zero rank.
   * @param _id an integer defining the identifier of this Singleton.
   */
  Singleton(int _id) : id(_id), parent(_id), rank(0) {}
  int id;                                               /** Id of the element */
  int parent;                                  /** Id of the element's parent */
  int rank;                   /** The depth of this element in the Union-Find */
};

/**
 * A Union-Find data structure.
 * Used to store disjoint subsets of Singletons with two main functionalities:
 * finding the subset a Singleton belongs to and joining two disjoint sets.
 */
class UnionFind
{
 public:
  /**
   * Default Constructor acts as an empty initializer.
   */
  UnionFind();

  /**
   * Destructor will release the memory used to store the subsets of Singletons
   * stored by this object.
   */
  ~UnionFind();

  /**
   * A function to create and store a new Singleton as a new subset.
   * @param id an integer defining the new Singleton's identifier, should be
   *        unique.
   */
  void MakeSet(int id);

  /**
   * A function to find a specified identifier's representative by
   * recursively searching the parent nodes until a Singleton is its own parent.
   * @param id an integer identifier for which we want to find a representative.
   */
  int Find(int id);

  /**
   * A function to union two possibly disjoint subsets by first finding their
   * representatives and updating the parent of the lower rank to be the other
   * representative. In case of a tie, the lower identifier will be the parent.
   * This will also increment the rank of the determined parent.
   * @param x an integer identifier of a Singleton we want to merge.
   * @param y an integer identifier of a Singleton we want to merge.
   */
  void Union(int x, int y);

  /**
   * A function to count the number of disjoint sets in the Union-Find
   */
  int CountComponents();

  /**
   * A function to get the representatives of each disjoint set in the
   * Union-Find
   * @param reps a vector of integers that will be populated by this function
   *        with a list of identifiers each representing a disjoint set in the
   *        Union-Find
   */
  void GetComponentRepresentatives(std::vector<int> &reps);

  /**
   * A function to get all of the identifiers associated to a particular
   * representative.
   * @param rep an integer defining the identifier of the representative for
   *        the list of Singleton identifiers. If this value is not actually
   *        a representative of any subset, then the return list will be empty.
   * @param items a vector of integers that will be populated by this function
   *        with a list of identifiers associated to the subset with rep as its
   *        representative.
   */
  void GetComponentItems(int rep, std::vector<int> &items);

 private:
  /**
   * The list of singletons, keyed by their identifiers, use of a map allows for
   * arbitrary indices to be used and does not impose a strict ordering from
   * zero.
   */
  std::map<int, Singleton *> sets;
};

#endif //UNION_FIND_H
