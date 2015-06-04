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

#include "AMSC/UnionFind.h"

UnionFind::UnionFind() { }

UnionFind::~UnionFind()
{
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
    iter != sets.end();
    iter++) {
    delete iter->second;
  }
}

void UnionFind::MakeSet(int id)
{
  if(sets.find(id) != sets.end())
  {
    std::cerr << "ERROR: Singleton " << id << " already exists" << std::endl;
    return;
  }
  sets[id] = new Singleton(id);
}

int UnionFind::Find(int id)
{
  if(sets.find(id) == sets.end())
    MakeSet(id);

  if(sets[id]->parent == id)
    return id;
  else
    return sets[id]->parent = Find(sets[id]->parent);
}

void UnionFind::Union(int x, int y)
{
  int xRoot = Find(x);
  int yRoot = Find(y);
  if( xRoot == yRoot)
    return;

  if( sets[xRoot]->rank < sets[yRoot]->rank 
   || (sets[xRoot]->rank < sets[yRoot]->rank && xRoot < yRoot) )
  {
    sets[xRoot]->parent = yRoot;
    sets[yRoot]->rank = sets[yRoot]->rank + 1;
  }
  else
  {
    sets[yRoot]->parent = xRoot;
    sets[xRoot]->rank = sets[xRoot]->rank + 1;
  }
}

int UnionFind::CountComponents()
{
  int count = 0;
  std::set<int> roots;
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
      iter != sets.end();
      iter++)
  {
    int root = Find(iter->first);
    if( roots.find(root) == roots.end())
    {
      roots.insert(root);
      count++;
    }
  }
  return count;
}

void UnionFind::GetComponentRepresentatives(std::vector<int> &reps)
{
  int count = 0;
  std::set<int> roots;
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
      iter != sets.end();
      iter++)
  {
    int root = Find(iter->first);
    if( roots.find(root) == roots.end())
    {
      roots.insert(root);
      reps.push_back(root);
      count++;
    }
  }
}

void UnionFind::GetComponentItems(int rep, std::vector<int> &items)
{
  int count = 0;
  for(std::map<int, Singleton *>::iterator iter = sets.begin();
      iter != sets.end();
      iter++)
  {
    int root = Find(iter->first);
    if(rep == root)
    {
      items.push_back(iter->first);
      count++;
    }
  }
}
